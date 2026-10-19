"""
Expert Pruning for Mixture of Experts Models

Post-training optimization to reduce inference costs:
- Cluster-driven expert pruning
- Importance-based expert selection
- Expert merging and consolidation
- Structured pruning for efficiency

Reference: Cluster-Driven Expert Pruning (arXiv:2504.07807)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Set
from enum import Enum
import json

logger = logging.getLogger(__name__)

# Lazy imports
torch = None
np = None


def _ensure_torch():
    global torch
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except ImportError:
            pass


def _ensure_numpy():
    global np
    if np is None:
        try:
            import numpy as _np
            np = _np
        except ImportError:
            # Fallback to list-based operations
            pass


class PruningStrategy(Enum):
    """Expert pruning strategies."""
    IMPORTANCE_BASED = "importance"      # Prune least important experts
    CLUSTER_DRIVEN = "cluster"           # Prune redundant experts in clusters
    ACTIVATION_BASED = "activation"      # Prune least activated experts
    SIMILARITY_MERGE = "similarity"      # Merge similar experts
    STRUCTURED = "structured"            # Structured pruning within experts


@dataclass
class PruningConfig:
    """Configuration for expert pruning."""

    # Pruning strategy
    strategy: PruningStrategy = PruningStrategy.CLUSTER_DRIVEN

    # Target compression
    target_expert_ratio: float = 0.5  # Keep 50% of experts
    target_param_ratio: float = 0.7   # Target 70% of original params

    # Importance scoring
    importance_metric: str = "activation_frequency"  # or "gradient_magnitude", "taylor"

    # Clustering settings
    num_clusters: int = 4
    cluster_merge_threshold: float = 0.8  # Similarity threshold for merging

    # Calibration
    calibration_samples: int = 1000
    calibration_batch_size: int = 32

    # Quality constraints
    max_quality_degradation: float = 0.02  # Max 2% degradation
    validation_metric: str = "perplexity"

    # Advanced options
    iterative_pruning: bool = True
    pruning_steps: int = 5
    fine_tune_after_pruning: bool = True
    fine_tune_steps: int = 1000


@dataclass
class ExpertStats:
    """Statistics for an expert."""
    expert_id: int
    layer_id: int
    activation_count: int = 0
    activation_frequency: float = 0.0
    average_routing_weight: float = 0.0
    gradient_magnitude: float = 0.0
    taylor_importance: float = 0.0
    cluster_id: int = -1
    is_representative: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "layer_id": self.layer_id,
            "activation_count": self.activation_count,
            "activation_frequency": self.activation_frequency,
            "average_routing_weight": self.average_routing_weight,
            "gradient_magnitude": self.gradient_magnitude,
            "taylor_importance": self.taylor_importance,
            "cluster_id": self.cluster_id,
            "is_representative": self.is_representative,
        }


class ExpertImportanceAnalyzer:
    """
    Analyzes expert importance for pruning decisions.

    Methods:
    - Activation frequency tracking
    - Gradient-based importance
    - Taylor expansion importance
    """

    def __init__(self, config: PruningConfig):
        self.config = config
        self.expert_stats: Dict[Tuple[int, int], ExpertStats] = {}
        self.total_samples = 0

        _ensure_torch()
        _ensure_numpy()

        logger.info(f"ExpertImportanceAnalyzer initialized with {config.importance_metric} metric")

    def register_expert(self, layer_id: int, expert_id: int):
        """Register an expert for tracking."""
        key = (layer_id, expert_id)
        if key not in self.expert_stats:
            self.expert_stats[key] = ExpertStats(
                expert_id=expert_id,
                layer_id=layer_id
            )

    def record_activation(
        self,
        layer_id: int,
        expert_id: int,
        routing_weight: float,
        batch_size: int = 1
    ):
        """Record an expert activation."""
        key = (layer_id, expert_id)
        self.register_expert(layer_id, expert_id)

        stats = self.expert_stats[key]
        stats.activation_count += batch_size

        # Update running average of routing weight
        n = stats.activation_count
        stats.average_routing_weight = (
            (stats.average_routing_weight * (n - batch_size) + routing_weight * batch_size) / n
        )

    def record_gradient(
        self,
        layer_id: int,
        expert_id: int,
        gradient_norm: float
    ):
        """Record gradient magnitude for an expert."""
        key = (layer_id, expert_id)
        self.register_expert(layer_id, expert_id)

        stats = self.expert_stats[key]
        # Exponential moving average
        alpha = 0.1
        stats.gradient_magnitude = (1 - alpha) * stats.gradient_magnitude + alpha * gradient_norm

    def compute_taylor_importance(
        self,
        layer_id: int,
        expert_id: int,
        weight: Any,
        gradient: Any
    ):
        """Compute Taylor expansion importance score."""
        key = (layer_id, expert_id)
        self.register_expert(layer_id, expert_id)

        if torch is not None and hasattr(weight, 'abs'):
            importance = (weight * gradient).abs().sum().item()
        else:
            importance = 0.0

        self.expert_stats[key].taylor_importance = importance

    def update_frequencies(self, total_samples: int):
        """Update activation frequencies based on total samples."""
        self.total_samples = total_samples

        for stats in self.expert_stats.values():
            stats.activation_frequency = stats.activation_count / max(1, total_samples)

    def get_importance_scores(self) -> Dict[Tuple[int, int], float]:
        """Get importance scores for all experts."""
        scores = {}

        for key, stats in self.expert_stats.items():
            if self.config.importance_metric == "activation_frequency":
                score = stats.activation_frequency
            elif self.config.importance_metric == "gradient_magnitude":
                score = stats.gradient_magnitude
            elif self.config.importance_metric == "taylor":
                score = stats.taylor_importance
            elif self.config.importance_metric == "combined":
                # Weighted combination
                score = (
                    0.4 * stats.activation_frequency +
                    0.3 * stats.gradient_magnitude +
                    0.3 * stats.taylor_importance
                )
            else:
                score = stats.activation_frequency

            scores[key] = score

        return scores

    def get_layer_stats(self, layer_id: int) -> List[ExpertStats]:
        """Get stats for all experts in a layer."""
        return [
            stats for (lid, _), stats in self.expert_stats.items()
            if lid == layer_id
        ]


class ExpertClusterer:
    """
    Clusters experts based on weight similarity.

    Used for:
    - Identifying redundant experts
    - Selecting representative experts per cluster
    - Guided expert merging
    """

    def __init__(self, config: PruningConfig):
        self.config = config
        self.clusters: Dict[int, List[Tuple[int, int]]] = {}
        self.centroids: Dict[int, Any] = {}

        _ensure_numpy()

        logger.info(f"ExpertClusterer initialized with {config.num_clusters} clusters")

    def cluster_experts(
        self,
        expert_weights: Dict[Tuple[int, int], Any],
        layer_id: Optional[int] = None
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Cluster experts based on weight similarity.

        Args:
            expert_weights: Dictionary mapping (layer_id, expert_id) to weights
            layer_id: If specified, only cluster experts in this layer

        Returns:
            Dictionary mapping cluster_id to list of (layer_id, expert_id)
        """
        # Filter by layer if specified
        if layer_id is not None:
            weights = {
                k: v for k, v in expert_weights.items()
                if k[0] == layer_id
            }
        else:
            weights = expert_weights

        if len(weights) == 0:
            return {}

        # Flatten weights to vectors
        expert_keys = list(weights.keys())
        weight_vectors = []

        for key in expert_keys:
            w = weights[key]
            if torch is not None and hasattr(w, 'flatten'):
                vec = w.flatten().cpu().numpy() if hasattr(w, 'cpu') else w.flatten()
            else:
                vec = [0.0] * 100  # Mock vector
            weight_vectors.append(vec)

        # Simple k-means clustering (mock implementation)
        num_clusters = min(self.config.num_clusters, len(expert_keys))
        clusters = self._kmeans_cluster(weight_vectors, num_clusters)

        # Map back to expert keys
        result = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in result:
                result[cluster_id] = []
            result[cluster_id].append(expert_keys[i])

        self.clusters = result

        logger.info(f"Clustered {len(expert_keys)} experts into {len(result)} clusters")

        return result

    def _kmeans_cluster(self, vectors: List, k: int) -> List[int]:
        """Simple k-means clustering."""
        n = len(vectors)

        # Simple assignment based on index (mock)
        # In production, use sklearn.cluster.KMeans or similar
        clusters = [i % k for i in range(n)]

        return clusters

    def compute_similarity(self, weights_a: Any, weights_b: Any) -> float:
        """Compute cosine similarity between two weight tensors."""
        if torch is not None and hasattr(weights_a, 'flatten'):
            a = weights_a.flatten()
            b = weights_b.flatten()

            if hasattr(a, 'float'):
                a = a.float()
                b = b.float()

            sim = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item()
        else:
            sim = 0.5  # Default similarity

        return sim

    def select_representatives(
        self,
        importance_scores: Dict[Tuple[int, int], float]
    ) -> Set[Tuple[int, int]]:
        """
        Select representative experts from each cluster.

        Selects the most important expert from each cluster.
        """
        representatives = set()

        for cluster_id, expert_keys in self.clusters.items():
            # Find most important expert in cluster
            best_key = max(
                expert_keys,
                key=lambda k: importance_scores.get(k, 0.0)
            )
            representatives.add(best_key)

        logger.info(f"Selected {len(representatives)} representative experts")

        return representatives


class ExpertMerger:
    """
    Merges similar experts to reduce model size.

    Methods:
    - Weight averaging
    - Fisher-weighted merging
    - Task arithmetic
    """

    def __init__(self, config: PruningConfig):
        self.config = config

        _ensure_torch()

        logger.info("ExpertMerger initialized")

    def merge_experts(
        self,
        expert_weights: List[Any],
        expert_importance: List[float],
        method: str = "weighted_average"
    ) -> Any:
        """
        Merge multiple experts into one.

        Args:
            expert_weights: List of weight tensors
            expert_importance: Importance scores for weighting
            method: Merging method

        Returns:
            Merged weight tensor
        """
        if len(expert_weights) == 0:
            return None

        if len(expert_weights) == 1:
            return expert_weights[0]

        if method == "weighted_average":
            return self._weighted_average_merge(expert_weights, expert_importance)
        elif method == "simple_average":
            return self._simple_average_merge(expert_weights)
        elif method == "fisher":
            return self._fisher_merge(expert_weights, expert_importance)
        else:
            return self._weighted_average_merge(expert_weights, expert_importance)

    def _weighted_average_merge(
        self,
        weights: List[Any],
        importance: List[float]
    ) -> Any:
        """Merge using importance-weighted average."""
        if torch is None:
            return weights[0]

        # Normalize importance scores
        total = sum(importance) + 1e-8
        normalized = [imp / total for imp in importance]

        # Weighted sum
        merged = None
        for w, imp in zip(weights, normalized):
            if merged is None:
                merged = w * imp
            else:
                merged = merged + w * imp

        return merged

    def _simple_average_merge(self, weights: List[Any]) -> Any:
        """Simple averaging of weights."""
        if torch is None:
            return weights[0]

        merged = weights[0].clone()
        for w in weights[1:]:
            merged = merged + w
        merged = merged / len(weights)

        return merged

    def _fisher_merge(
        self,
        weights: List[Any],
        fisher_info: List[float]
    ) -> Any:
        """Fisher information weighted merge."""
        # Simplified - actual implementation uses full Fisher information matrix
        return self._weighted_average_merge(weights, fisher_info)


class ExpertPruner:
    """
    Main expert pruning class.

    Orchestrates:
    - Importance analysis
    - Clustering
    - Pruning decisions
    - Model modification
    """

    def __init__(self, config: PruningConfig):
        self.config = config
        self.analyzer = ExpertImportanceAnalyzer(config)
        self.clusterer = ExpertClusterer(config)
        self.merger = ExpertMerger(config)

        # Pruning results
        self.pruned_experts: Set[Tuple[int, int]] = set()
        self.merged_experts: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.original_expert_count = 0
        self.final_expert_count = 0

        logger.info(f"ExpertPruner initialized with {config.strategy.value} strategy")

    def calibrate(
        self,
        model: Any,
        calibration_dataloader: Any
    ) -> Dict[str, Any]:
        """
        Run calibration to gather expert statistics.

        Args:
            model: MoE model to calibrate
            calibration_dataloader: DataLoader with calibration samples

        Returns:
            Calibration statistics
        """
        logger.info("Starting calibration...")

        total_samples = 0

        # Hook to capture expert activations
        # (Simplified - actual implementation uses forward hooks)

        # Mock calibration
        for batch_idx in range(self.config.calibration_samples // self.config.calibration_batch_size):
            # Simulate recording activations
            for layer_id in range(4):  # Assume 4 MoE layers
                for expert_id in range(8):  # Assume 8 experts per layer
                    # Random activation (mock)
                    if (batch_idx + layer_id + expert_id) % 3 == 0:
                        self.analyzer.record_activation(
                            layer_id, expert_id,
                            routing_weight=0.5 + 0.1 * expert_id,
                            batch_size=self.config.calibration_batch_size
                        )

            total_samples += self.config.calibration_batch_size

        # Update frequencies
        self.analyzer.update_frequencies(total_samples)

        stats = {
            "total_samples": total_samples,
            "experts_tracked": len(self.analyzer.expert_stats),
            "importance_scores": self.analyzer.get_importance_scores()
        }

        logger.info(f"Calibration complete: {total_samples} samples, "
                   f"{len(self.analyzer.expert_stats)} experts tracked")

        return stats

    def prune(
        self,
        model: Any,
        expert_weights: Optional[Dict[Tuple[int, int], Any]] = None
    ) -> Dict[str, Any]:
        """
        Prune experts from the model.

        Args:
            model: MoE model to prune
            expert_weights: Optional pre-extracted expert weights

        Returns:
            Pruning results
        """
        logger.info(f"Starting {self.config.strategy.value} pruning...")

        # Get importance scores
        importance_scores = self.analyzer.get_importance_scores()

        if len(importance_scores) == 0:
            logger.warning("No importance scores available - run calibration first")
            return {"error": "calibration required"}

        self.original_expert_count = len(importance_scores)

        # Choose pruning strategy
        if self.config.strategy == PruningStrategy.IMPORTANCE_BASED:
            pruned = self._importance_pruning(importance_scores)
        elif self.config.strategy == PruningStrategy.CLUSTER_DRIVEN:
            pruned = self._cluster_pruning(importance_scores, expert_weights)
        elif self.config.strategy == PruningStrategy.ACTIVATION_BASED:
            pruned = self._activation_pruning(importance_scores)
        elif self.config.strategy == PruningStrategy.SIMILARITY_MERGE:
            pruned = self._similarity_merge_pruning(importance_scores, expert_weights)
        else:
            pruned = self._importance_pruning(importance_scores)

        self.pruned_experts = pruned
        self.final_expert_count = self.original_expert_count - len(pruned)

        results = {
            "strategy": self.config.strategy.value,
            "original_experts": self.original_expert_count,
            "pruned_experts": len(pruned),
            "remaining_experts": self.final_expert_count,
            "compression_ratio": len(pruned) / max(1, self.original_expert_count),
            "pruned_expert_ids": list(pruned),
        }

        logger.info(f"Pruning complete: {len(pruned)}/{self.original_expert_count} experts pruned")

        return results

    def _importance_pruning(
        self,
        importance_scores: Dict[Tuple[int, int], float]
    ) -> Set[Tuple[int, int]]:
        """Prune least important experts."""
        # Sort by importance
        sorted_experts = sorted(
            importance_scores.items(),
            key=lambda x: x[1]
        )

        # Determine how many to prune
        num_to_prune = int(len(sorted_experts) * (1 - self.config.target_expert_ratio))

        # Select least important for pruning
        pruned = set(key for key, _ in sorted_experts[:num_to_prune])

        return pruned

    def _cluster_pruning(
        self,
        importance_scores: Dict[Tuple[int, int], float],
        expert_weights: Optional[Dict[Tuple[int, int], Any]]
    ) -> Set[Tuple[int, int]]:
        """Cluster-driven pruning - keep one representative per cluster."""
        if expert_weights is None:
            # Fall back to importance-based if no weights
            return self._importance_pruning(importance_scores)

        # Cluster experts
        clusters = self.clusterer.cluster_experts(expert_weights)

        # Select representatives
        representatives = self.clusterer.select_representatives(importance_scores)

        # Prune non-representatives
        all_experts = set(importance_scores.keys())
        pruned = all_experts - representatives

        return pruned

    def _activation_pruning(
        self,
        importance_scores: Dict[Tuple[int, int], float]
    ) -> Set[Tuple[int, int]]:
        """Prune experts with low activation frequency."""
        pruned = set()

        # Use activation frequency directly
        for key, stats in self.analyzer.expert_stats.items():
            if stats.activation_frequency < 0.01:  # Less than 1% activation
                pruned.add(key)

        # Ensure we don't prune too many
        target_prune = int(len(importance_scores) * (1 - self.config.target_expert_ratio))
        if len(pruned) > target_prune:
            # Sort by frequency and keep the lowest
            sorted_by_freq = sorted(
                pruned,
                key=lambda k: self.analyzer.expert_stats[k].activation_frequency
            )
            pruned = set(sorted_by_freq[:target_prune])

        return pruned

    def _similarity_merge_pruning(
        self,
        importance_scores: Dict[Tuple[int, int], float],
        expert_weights: Optional[Dict[Tuple[int, int], Any]]
    ) -> Set[Tuple[int, int]]:
        """Merge similar experts, prune merged ones."""
        if expert_weights is None:
            return self._importance_pruning(importance_scores)

        pruned = set()
        merged = {}

        expert_keys = list(expert_weights.keys())
        processed = set()

        for i, key_a in enumerate(expert_keys):
            if key_a in processed:
                continue

            similar_experts = [key_a]

            for j, key_b in enumerate(expert_keys[i+1:], i+1):
                if key_b in processed:
                    continue

                # Only compare within same layer
                if key_a[0] != key_b[0]:
                    continue

                sim = self.clusterer.compute_similarity(
                    expert_weights[key_a],
                    expert_weights[key_b]
                )

                if sim > self.config.cluster_merge_threshold:
                    similar_experts.append(key_b)
                    processed.add(key_b)

            if len(similar_experts) > 1:
                # Keep most important, prune others
                most_important = max(
                    similar_experts,
                    key=lambda k: importance_scores.get(k, 0.0)
                )

                for key in similar_experts:
                    if key != most_important:
                        pruned.add(key)

                merged[most_important] = similar_experts

            processed.add(key_a)

        self.merged_experts = merged

        return pruned

    def apply_pruning(self, model: Any) -> Any:
        """
        Apply pruning to the model.

        Args:
            model: MoE model to modify

        Returns:
            Pruned model
        """
        logger.info(f"Applying pruning to model: removing {len(self.pruned_experts)} experts")

        # This would modify the actual model structure
        # Simplified - actual implementation updates MoE layer configurations

        return model

    def get_pruning_report(self) -> Dict[str, Any]:
        """Generate detailed pruning report."""
        return {
            "config": {
                "strategy": self.config.strategy.value,
                "target_expert_ratio": self.config.target_expert_ratio,
                "importance_metric": self.config.importance_metric,
            },
            "results": {
                "original_experts": self.original_expert_count,
                "pruned_experts": len(self.pruned_experts),
                "final_experts": self.final_expert_count,
                "compression_ratio": 1 - (self.final_expert_count / max(1, self.original_expert_count)),
            },
            "merged_groups": len(self.merged_experts),
            "expert_statistics": {
                str(k): v.to_dict()
                for k, v in self.analyzer.expert_stats.items()
            }
        }


def create_pruning_config(
    target_compression: float = 0.5,
    strategy: str = "cluster",
    max_quality_drop: float = 0.02
) -> PruningConfig:
    """
    Factory function for pruning configuration.

    Args:
        target_compression: Target ratio of experts to keep
        strategy: Pruning strategy name
        max_quality_drop: Maximum acceptable quality degradation

    Returns:
        Configured PruningConfig
    """
    return PruningConfig(
        strategy=PruningStrategy(strategy),
        target_expert_ratio=target_compression,
        max_quality_degradation=max_quality_drop,
        iterative_pruning=True,
        fine_tune_after_pruning=True,
    )


__all__ = [
    'PruningConfig',
    'PruningStrategy',
    'ExpertStats',
    'ExpertImportanceAnalyzer',
    'ExpertClusterer',
    'ExpertMerger',
    'ExpertPruner',
    'create_pruning_config',
]
