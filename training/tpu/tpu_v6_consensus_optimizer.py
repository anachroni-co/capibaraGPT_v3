"""
TPU v6 Optimized Consensus System for Meta-Consensus-Comp

This module implements TPU v6-64 specific optimizations for the meta-consensus system:
- TPU v6 mesh parallelism for expert processing
- JAX-native implementations optimized for TPU architecture
- Distributed consensus across TPU cores
- Memory-efficient operations for 2TB TPU memory
- H200 distributed hosting integration
- HuggingFace Pro features with 20x inference credits
"""

import logging
import asyncio
import hashlib
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import json
import numpy as np
import math

# JAX imports for TPU optimization
try:
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap, pmap, lax
    from jax.sharding import Mesh, PartitionSpec as PS
    from jax.experimental import mesh_utils
    import flax.linen as nn
    from flax import struct
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None

# Import base components.
#
# Previously these imports used a `.meta_consensus_system` / `.optimized_meta_consensus`
# spec which resolves against `training/tpu/` — where the modules do not exist
# (they live under `training/research/consensus/`). That made the module unimportable,
# which in turn is why the file had never been covered by a real test. We
# now use the correct `..consensus.*` path and wrap it in `try/except` so the
# optimizer degrades gracefully if the consensus package is stripped down.
try:
    from ..research.consensus.meta_consensus_system import (
        MetaConsensusSystem,
        MetaConsensusConfig,
    )
    from ..research.consensus.optimized_meta_consensus import (
        OptimizedMetaConsensusSystem,
        OptimizationLevel,
    )
    _BASE_COMPONENTS_AVAILABLE = True
except ImportError:
    MetaConsensusSystem = None  # type: ignore[assignment]
    MetaConsensusConfig = None  # type: ignore[assignment]
    OptimizedMetaConsensusSystem = None  # type: ignore[assignment]
    OptimizationLevel = None  # type: ignore[assignment]
    _BASE_COMPONENTS_AVAILABLE = False

# Import TPU-specific optimizations from comp branch
try:
    from capibara.training.tpu_v6_huggingface_pro_strategy import (
        TPUv6HuggingFaceProStrategy, TPUv6HuggingFaceProConfig
    )
    TPU_V6_STRATEGY_AVAILABLE = True
except ImportError:
    TPU_V6_STRATEGY_AVAILABLE = False

logger = logging.getLogger(__name__)

class TPUConsensusMode(Enum):
    """TPU-specific consensus modes."""
    SINGLE_CORE = "single_core"                    # Single TPU core
    MULTI_CORE_PARALLEL = "multi_core_parallel"    # Parallel across cores
    MESH_DISTRIBUTED = "mesh_distributed"          # Full mesh distribution
    PIPELINE_PARALLEL = "pipeline_parallel"        # Pipeline parallelism
    DATA_PARALLEL = "data_parallel"                # Data parallelism
    HYBRID_PARALLEL = "hybrid_parallel"            # Hybrid parallelism

@dataclass
class TPUv6ConsensusConfig:
    """Configurestion for TPU v6 consensus optimization."""
    
    # TPU v6 hardware configuration
    tpu_cores: int = 64
    memory_per_core_gb: int = 32
    total_memory_tb: float = 2.0
    mesh_shape: Tuple[int, int] = (8, 8)
    
    # Consensus-specific settings
    consensus_mode: TPUConsensusMode = TPUConsensusMode.MESH_DISTRIBUTED
    experts_per_core: int = 4
    max_sequence_length: int = 2048
    batch_size_per_core: int = 32
    
    # Optimization settings
    enable_xla_optimization: bool = True
    enable_memory_mapping: bool = True
    enable_pipeline_parallelism: bool = True
    enable_gradient_accumulation: bool = True
    
    # HuggingFace Pro integration
    enable_hf_pro_features: bool = True
    h200_distributed_hosting: bool = True
    inference_credits_multiplier: int = 20
    
    # Performance tuning
    compilation_cache_size: int = 1000
    memory_fraction: float = 0.9
    prefetch_size: int = 4

    # Embedding resolution
    #
    # Previously the optimizer used `jax.random.normal` seeded by
    # `hash(text) % 2**32` as a "mock embedding", which was non-reproducible
    # across Python sessions (hash randomization) and silently hid the fact
    # that no real embedder was wired in. The fields below make the
    # embedding source explicit:
    #
    # - `embedding_dim`: target dimensionality (previously hardcoded to 768).
    # - `query_embedder`: optional callable `str -> array` producing the query
    #   embedding. When `None`, a deterministic SHA256-derived fallback is
    #   used (reproducible, no randomness).
    # - `expert_embedder`: optional callable `dict -> array` producing an
    #   expert embedding. If an expert entry already carries an `embedding`
    #   field, that pre-computed vector is used directly.
    embedding_dim: int = 768
    query_embedder: Optional[Callable[[str], Any]] = None
    expert_embedder: Optional[Callable[[Dict[str, Any]], Any]] = None

@dataclass
class TPUConsensusMetrics:
    """Metrics for TPU consensus operations."""
    
    # TPU utilization
    core_utilization: List[float] = field(default_factory=list)
    memory_utilization_per_core: List[float] = field(default_factory=list)
    compilation_time_ms: float = 0.0
    
    # Consensus performance
    consensus_operations_per_second: float = 0.0
    expert_processing_throughput: float = 0.0
    mesh_communication_latency_ms: float = 0.0
    
    # Resource efficiency
    flops_utilization: float = 0.0
    memory_bandwidth_utilization: float = 0.0
    network_bandwidth_utilization: float = 0.0
    
    # Quality metrics
    distributed_consensus_accuracy: float = 0.0
    cross_core_agreement: float = 0.0
    mesh_synchronization_efficiency: float = 0.0

def _deterministic_embedding(text: str, dim: int = 768) -> Any:
    """Reproducible fallback embedding derived from the input text.

    This is **not** a semantic embedding. It is a stable fingerprint computed
    from a SHA256 digest of the UTF-8 encoded `text`, expanded to `dim`
    float32 components and L2-normalized. It exists to replace the previous
    random PRNG-based "mock embedding" while preserving shape/dtype contracts.

    Key guarantees:

    - **Deterministic**: the same text always produces the same vector, in
      any process, on any machine. No dependency on Python's `hash()` salt
      or on JAX PRNG state.
    - **Shape/dtype stable**: always returns `(dim,)` float32 (as a JAX array
      when JAX is available, else a NumPy array).
    - **Not trainable or learned**: callers that need a semantic embedding
      must inject a real model via
      `TPUv6ConsensusConfig.query_embedder` / `expert_embedder`.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    needed_bytes = dim * 4
    repeats = (needed_bytes // len(digest)) + 1
    raw = (digest * repeats)[:needed_bytes]
    # Interpret the byte stream as uint32 and map into a roughly centered
    # float32 vector in [-0.5, 0.5]. Any monotonic decoding works here; the
    # important property is that it is a pure function of `text`.
    arr = np.frombuffer(raw, dtype=np.uint32).astype(np.float32)
    arr = arr / np.float32(2 ** 32) - np.float32(0.5)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    if JAX_AVAILABLE:
        return jnp.asarray(arr)
    return arr


class TPUv6ConsensusOptimizer:
    """
     TPU v6 Consensus Optimizer
    
    Implements TPU v6-64 specific optimizations for meta-consensus:
    - Mesh parallelism across 64 cores
    - Memory-efficient consensus algorithms
    - JAX-native implementations for maximum performance
    - Integration with HuggingFace Pro features
    - Distributed expert processing with pipeline parallelism
    """
    
    def __init__(self, config: TPUv6ConsensusConfig):
        self.config = config
        self.metrics = TPUConsensusMetrics()
        
        # TPU mesh setup
        self.mesh = None
        self.mesh_devices = None
        
        # JAX compilation cache
        self.compiled_functions = {}
        
        # Expert distribution mapping
        self.expert_core_mapping = {}
        
        if JAX_AVAILABLE:
            self._initialize_tpu_mesh()
            self._compile_consensus_functions()
        
        logger.info(f" TPU v6 Consensus Optimizer initialized for {config.tpu_cores} cores")
    
    def _initialize_tpu_mesh(self):
        """Initialize TPU v6 mesh for distributed consensus."""
        
        try:
            # Create TPU mesh
            devices = jax.devices()
            if len(devices) >= self.config.tpu_cores:
                # Reshape devices into mesh
                mesh_devices = np.array(devices[:self.config.tpu_cores]).reshape(self.config.mesh_shape)
                self.mesh = Mesh(mesh_devices, axis_names=('x', 'y'))
                
                logger.info(f" TPU v6 mesh initialized: {self.config.mesh_shape} ({self.config.tpu_cores} cores)")
            else:
                logger.warning(f"️ Only {len(devices)} TPU devices available, need {self.config.tpu_cores}")
                
        except Exception as e:
            logger.error(f" TPU mesh initialization failed: {e}")
    
    def _compile_consensus_functions(self):
        """Compile JAX functions for TPU optimization."""
        
        # Compile distributed expert processing
        self.compiled_functions['process_experts'] = jit(
            self._process_experts_on_mesh,
            static_argnums=(2,)  # num_experts is static
        )
        
        # Compile consensus calculation
        self.compiled_functions['calculate_consensus'] = jit(
            self._calculate_distributed_consensus,
            static_argnums=(1,)  # num_responses is static
        )
        
        # Compile similarity computation
        self.compiled_functions['compute_similarities'] = jit(
            vmap(self._compute_expert_similarity, in_axes=(None, 0, 0))
        )
        
        logger.info(" JAX functions compiled for TPU v6")
    
    @jit
    def _process_experts_on_mesh(self, query_embedding, expert_embeddings, num_experts):
        """Process experts distributed across TPU mesh."""
        
        # Distribute experts across TPU cores
        experts_per_core = num_experts // self.config.tpu_cores
        
        def process_expert_batch(expert_batch, query_emb):
            # Cosine similarity calculation
            similarities = jnp.dot(expert_batch, query_emb) / (
                jnp.linalg.norm(expert_batch, axis=1) * jnp.linalg.norm(query_emb)
            )
            return similarities
        
        # Reshape experts for mesh processing
        if num_experts % self.config.tpu_cores == 0:
            expert_batches = expert_embeddings.reshape(
                self.config.tpu_cores, experts_per_core, -1
            )
        else:
            # Pad to make divisible
            padding_needed = self.config.tpu_cores - (num_experts % self.config.tpu_cores)
            padded_experts = jnp.concatenate([
                expert_embeddings,
                jnp.zeros((padding_needed, expert_embeddings.shape[1]))
            ])
            expert_batches = padded_experts.reshape(
                self.config.tpu_cores, (num_experts + padding_needed) // self.config.tpu_cores, -1
            )
        
        # Process in parallel across mesh
        batch_similarities = vmap(process_expert_batch, in_axes=(0, None))(
            expert_batches, query_embedding
        )
        
        # Flatten results
        similarities = batch_similarities.reshape(-1)[:num_experts]
        
        return similarities
    
    @jit
    def _calculate_distributed_consensus(self, response_embeddings, num_responses):
        """Calculate consensus distributed across TPU cores."""
        
        # Compute pairwise similarities
        similarity_matrix = jnp.dot(response_embeddings, response_embeddings.T)
        
        # Normalize similarities
        norms = jnp.linalg.norm(response_embeddings, axis=1)
        norm_matrix = jnp.outer(norms, norms)
        similarity_matrix = similarity_matrix / (norm_matrix + 1e-8)
        
        # Calculate consensus weights based on similarity to others
        consensus_weights = jnp.mean(similarity_matrix, axis=1)
        
        # Apply softmax for final weights
        final_weights = jax.nn.softmax(consensus_weights)
        
        return final_weights, similarity_matrix
    
    @jit
    def _compute_expert_similarity(self, query_embedding, expert_embedding, expert_weight):
        """Compute similarity between query and expert."""
        
        # Cosine similarity
        dot_product = jnp.dot(query_embedding, expert_embedding)
        query_norm = jnp.linalg.norm(query_embedding)
        expert_norm = jnp.linalg.norm(expert_embedding)
        
        similarity = dot_product / (query_norm * expert_norm + 1e-8)
        
        # Weight adjustment
        weighted_similarity = similarity * expert_weight
        
        return weighted_similarity
    
    async def optimize_consensus_for_tpu_v6(
        self,
        query: str,
        expert_pool: List[Dict[str, Any]],
        consensus_mode: TPUConsensusMode = None
    ) -> Dict[str, Any]:
        """
        Optimize consensus calculation for TPU v6 architecture.
        
        Args:
            query: Input query
            expert_pool: Pool of available experts
            consensus_mode: TPU-specific consensus mode
            
        Returns:
            Optimized consensus result with TPU metrics
        """
        
        if not JAX_AVAILABLE or self.mesh is None:
            logger.warning("TPU v6 optimization not available, falling back")
            return await self._fallback_consensus(query, expert_pool)
        
        start_time = time.time()
        consensus_mode = consensus_mode or self.config.consensus_mode
        
        try:
            # Generate query embedding (TPU optimized)
            query_embedding = await self._generate_tpu_query_embedding(query)
            
            # Prepare expert data for TPU processing
            expert_data = await self._prepare_expert_data_for_tpu(expert_pool)
            
            # Distribute experts across TPU mesh
            expert_distribution = await self._distribute_experts_on_mesh(
                expert_data, consensus_mode
            )
            
            # Process experts in parallel across TPU cores
            expert_results = await self._process_experts_parallel(
                query_embedding, expert_distribution, consensus_mode
            )
            
            # Calculate distributed consensus
            consensus_result = await self._calculate_tpu_consensus(
                expert_results, consensus_mode
            )
            
            # Collect TPU metrics
            processing_time = time.time() - start_time
            await self._update_tpu_metrics(consensus_result, processing_time)
            
            # Add TPU-specific metadata
            consensus_result.update({
                "tpu_optimized": True,
                "consensus_mode": consensus_mode.value,
                "tpu_cores_used": self.config.tpu_cores,
                "mesh_shape": self.config.mesh_shape,
                "processing_time_ms": round(processing_time * 1000, 2),
                "tpu_metrics": self._get_tpu_metrics_snapshot()
            })
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"TPU v6 consensus optimization failed: {e}")
            return await self._fallback_consensus(query, expert_pool)
    
    async def _generate_tpu_query_embedding(self, query: str) -> "jnp.ndarray":
        """Generate the query embedding for TPU-side consensus.

        Resolution order:

        1. ``self.config.query_embedder`` if a real embedder callable was
           injected via configuration.
        2. A deterministic SHA256-derived fallback (see
           :func:`_deterministic_embedding`).

        The previous implementation used ``jax.random.normal`` seeded by
        ``hash(query) % 2**32``. That path was functionally a silent mock:
        values depended on Python hash randomization and on the JAX PRNG,
        which made it impossible for callers to tell whether a real embedder
        was ever wired in. The two-step resolution above removes that
        ambiguity: either a real model is configured, or the fallback is
        reproducible and clearly labeled.
        """
        if self.config.query_embedder is not None:
            embedding = jnp.asarray(self.config.query_embedder(query))
        else:
            # `_deterministic_embedding` already returns a JAX array when JAX
            # is importable (this method is only reachable when it is, since
            # `optimize_consensus_for_tpu_v6` short-circuits to the fallback
            # otherwise).
            embedding = jnp.asarray(
                _deterministic_embedding(query, self.config.embedding_dim)
            )

        # Re-normalize to guard against embedders that skip this step.
        norm = jnp.linalg.norm(embedding)
        embedding = embedding / (norm + 1e-8)
        return embedding
    
    async def _prepare_expert_data_for_tpu(self, expert_pool: List[Dict[str, Any]]) -> Dict[str, "jnp.ndarray"]:
        """Assemble the expert-side tensors consumed by the TPU pipeline.

        Expert embedding resolution order, per pool entry:

        1. ``expert["embedding"]`` if the caller already pre-computed it
           (any array-like coerced to a JAX array).
        2. ``self.config.expert_embedder(expert)`` when a real embedder is
           injected via configuration.
        3. Deterministic SHA256-derived fallback keyed by the expert's
           ``name`` / ``id`` / index (see :func:`_deterministic_embedding`).

        The previous implementation always used ``jax.random.normal`` seeded
        by ``hash(name)``, which hid missing embedders behind values that
        looked plausible but were not reproducible across processes. This
        rewrite preserves the original tensor contract (shapes, dtypes, and
        padding to ``tpu_cores``) while making the data provenance explicit.
        """
        num_experts = len(expert_pool)
        embedding_dim = self.config.embedding_dim

        expert_embeddings: List["jnp.ndarray"] = []
        expert_weights: List[float] = []
        expert_quality_scores: List[float] = []
        expert_costs: List[float] = []

        for idx, expert in enumerate(expert_pool):
            raw_embedding = expert.get("embedding")
            if raw_embedding is not None:
                embedding = jnp.asarray(raw_embedding)
            elif self.config.expert_embedder is not None:
                embedding = jnp.asarray(self.config.expert_embedder(expert))
            else:
                key_text = str(
                    expert.get("name")
                    or expert.get("id")
                    or f"expert_{idx}"
                )
                # `_deterministic_embedding` already returns a JAX array when
                # JAX is importable, which is the only case this method is
                # invoked in (see `optimize_consensus_for_tpu_v6`).
                embedding = jnp.asarray(
                    _deterministic_embedding(key_text, embedding_dim)
                )

            if embedding.shape[-1] != embedding_dim:
                raise ValueError(
                    f"Expert embedding dim mismatch for expert {idx}: "
                    f"got {embedding.shape[-1]}, expected {embedding_dim}"
                )

            # Normalize for numerical stability; harmless if already normalized.
            embedding = embedding / (jnp.linalg.norm(embedding) + 1e-8)

            expert_embeddings.append(embedding)
            expert_weights.append(float(expert.get("weight", 1.0)))
            expert_quality_scores.append(float(expert.get("quality_score", 8.0)))
            expert_costs.append(float(expert.get("cost_per_1k_tokens", 0.001)))

        expert_data: Dict[str, Any] = {
            "embeddings": jnp.stack(expert_embeddings),
            "weights": jnp.array(expert_weights),
            "quality_scores": jnp.array(expert_quality_scores),
            "costs": jnp.array(expert_costs),
            "metadata": expert_pool,
        }

        # Pad to a multiple of `tpu_cores` so the mesh partitioning in
        # `_distribute_experts_on_mesh` and `_process_experts_on_mesh` never
        # has to handle ragged last batches. Padding uses zero vectors,
        # zero weights, zero quality (so they lose the argsort), and zero
        # cost (so they do not inflate budgets).
        if num_experts % self.config.tpu_cores != 0:
            padding_needed = self.config.tpu_cores - (num_experts % self.config.tpu_cores)
            expert_data["embeddings"] = jnp.concatenate([
                expert_data["embeddings"],
                jnp.zeros((padding_needed, embedding_dim)),
            ])
            expert_data["weights"] = jnp.concatenate([
                expert_data["weights"],
                jnp.zeros(padding_needed),
            ])
            expert_data["quality_scores"] = jnp.concatenate([
                expert_data["quality_scores"],
                jnp.zeros(padding_needed),
            ])
            expert_data["costs"] = jnp.concatenate([
                expert_data["costs"],
                jnp.zeros(padding_needed),
            ])

        return expert_data
    
    async def _distribute_experts_on_mesh(
        self,
        expert_data: Dict[str, jnp.ndarray],
        consensus_mode: TPUConsensusMode
    ) -> Dict[str, Any]:
        """Distribute experts across TPU mesh."""
        
        num_experts = expert_data["embeddings"].shape[0]
        experts_per_core = num_experts // self.config.tpu_cores
        
        if consensus_mode == TPUConsensusMode.MESH_DISTRIBUTED:
            # Distribute experts evenly across mesh
            expert_distribution = {
                "embeddings_per_core": expert_data["embeddings"].reshape(
                    self.config.tpu_cores, experts_per_core, -1
                ),
                "weights_per_core": expert_data["weights"].reshape(
                    self.config.tpu_cores, experts_per_core
                ),
                "quality_per_core": expert_data["quality_scores"].reshape(
                    self.config.tpu_cores, experts_per_core
                ),
                "costs_per_core": expert_data["costs"].reshape(
                    self.config.tpu_cores, experts_per_core
                ),
                "distribution_mode": "mesh_distributed"
            }
        
        elif consensus_mode == TPUConsensusMode.PIPELINE_PARALLEL:
            # Pipeline distribution for sequential processing
            expert_distribution = {
                "pipeline_stages": self._create_pipeline_stages(expert_data),
                "distribution_mode": "pipeline_parallel"
            }
        
        else:  # Single core or multi-core parallel
            expert_distribution = {
                "all_experts": expert_data,
                "distribution_mode": consensus_mode.value
            }
        
        return expert_distribution
    
    def _create_pipeline_stages(self, expert_data: Dict[str, jnp.ndarray]) -> List[Dict[str, jnp.ndarray]]:
        """Creates pipeline stages for expert processing."""
        
        num_experts = expert_data["embeddings"].shape[0]
        num_stages = 4  # 4-stage pipeline
        experts_per_stage = num_experts // num_stages
        
        stages = []
        for stage in range(num_stages):
            start_idx = stage * experts_per_stage
            end_idx = (stage + 1) * experts_per_stage if stage < num_stages - 1 else num_experts
            
            stage_data = {
                "embeddings": expert_data["embeddings"][start_idx:end_idx],
                "weights": expert_data["weights"][start_idx:end_idx],
                "quality_scores": expert_data["quality_scores"][start_idx:end_idx],
                "costs": expert_data["costs"][start_idx:end_idx],
                "stage_id": stage
            }
            stages.append(stage_data)
        
        return stages
    
    async def _process_experts_parallel(
        self,
        query_embedding: jnp.ndarray,
        expert_distribution: Dict[str, Any],
        consensus_mode: TPUConsensusMode
    ) -> Dict[str, Any]:
        """Process experts in parallel across TPU cores."""
        
        if consensus_mode == TPUConsensusMode.MESH_DISTRIBUTED:
            return await self._process_mesh_distributed(query_embedding, expert_distribution)
        elif consensus_mode == TPUConsensusMode.PIPELINE_PARALLEL:
            return await self._process_pipeline_parallel(query_embedding, expert_distribution)
        else:
            return await self._process_standard_parallel(query_embedding, expert_distribution)
    
    async def _process_mesh_distributed(
        self,
        query_embedding: jnp.ndarray,
        expert_distribution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process experts using mesh distribution."""
        
        # Use compiled function for mesh processing
        similarities = self.compiled_functions['process_experts'](
            query_embedding,
            expert_distribution["embeddings_per_core"],
            expert_distribution["embeddings_per_core"].shape[1]  # experts_per_core
        )
        
        # Combine results from all cores
        flattened_similarities = similarities.reshape(-1)
        flattened_weights = expert_distribution["weights_per_core"].reshape(-1)
        flattened_quality = expert_distribution["quality_per_core"].reshape(-1)
        flattened_costs = expert_distribution["costs_per_core"].reshape(-1)
        
        return {
            "similarities": flattened_similarities,
            "weights": flattened_weights,
            "quality_scores": flattened_quality,
            "costs": flattened_costs,
            "processing_mode": "mesh_distributed"
        }
    
    async def _process_pipeline_parallel(
        self,
        query_embedding: jnp.ndarray,
        expert_distribution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process experts using pipeline parallelism."""
        
        pipeline_stages = expert_distribution["pipeline_stages"]
        stage_results = []
        
        # Process each pipeline stage
        for stage_data in pipeline_stages:
            stage_similarities = self.compiled_functions['compute_similarities'](
                query_embedding,
                stage_data["embeddings"],
                stage_data["weights"]
            )
            
            stage_results.append({
                "similarities": stage_similarities,
                "weights": stage_data["weights"],
                "quality_scores": stage_data["quality_scores"],
                "costs": stage_data["costs"],
                "stage_id": stage_data["stage_id"]
            })
        
        # Combine pipeline results
        all_similarities = jnp.concatenate([stage["similarities"] for stage in stage_results])
        all_weights = jnp.concatenate([stage["weights"] for stage in stage_results])
        all_quality = jnp.concatenate([stage["quality_scores"] for stage in stage_results])
        all_costs = jnp.concatenate([stage["costs"] for stage in stage_results])
        
        return {
            "similarities": all_similarities,
            "weights": all_weights,
            "quality_scores": all_quality,
            "costs": all_costs,
            "processing_mode": "pipeline_parallel",
            "pipeline_stages": len(pipeline_stages)
        }
    
    async def _process_standard_parallel(
        self,
        query_embedding: jnp.ndarray,
        expert_distribution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process experts using standard parallelism."""
        
        expert_data = expert_distribution["all_experts"]
        
        # Use vectorized computation
        similarities = self.compiled_functions['compute_similarities'](
            query_embedding,
            expert_data["embeddings"],
            expert_data["weights"]
        )
        
        return {
            "similarities": similarities,
            "weights": expert_data["weights"],
            "quality_scores": expert_data["quality_scores"],
            "costs": expert_data["costs"],
            "processing_mode": "standard_parallel"
        }
    
    async def _calculate_tpu_consensus(
        self,
        expert_results: Dict[str, Any],
        consensus_mode: TPUConsensusMode
    ) -> Dict[str, Any]:
        """Calculate consensus using TPU-optimized algorithms."""
        
        similarities = expert_results["similarities"]
        weights = expert_results["weights"]
        quality_scores = expert_results["quality_scores"]
        costs = expert_results["costs"]
        
        # Select top experts using TPU-optimized selection
        top_k = min(7, len(similarities))
        
        # Combined scoring
        combined_scores = (
            similarities * 0.4 +
            quality_scores / 10.0 * 0.4 +
            (1.0 - jnp.minimum(costs * 100, 1.0)) * 0.2  # Cost penalty
        )
        
        # Select top experts
        top_indices = jnp.argsort(-combined_scores)[:top_k]
        selected_similarities = similarities[top_indices]
        selected_weights = weights[top_indices]
        
        # Calculate consensus confidence
        consensus_confidence = float(jnp.mean(selected_similarities))
        
        # Calculate expected quality
        selected_quality = quality_scores[top_indices]
        expected_quality = float(jnp.mean(selected_quality))
        
        # Calculate total cost
        selected_costs = costs[top_indices]
        total_cost = float(jnp.sum(selected_costs))
        
        return {
            "selected_expert_indices": [int(idx) for idx in top_indices],
            "selected_similarities": [float(s) for s in selected_similarities],
            "consensus_confidence": consensus_confidence,
            "expected_quality": expected_quality,
            "total_cost": total_cost,
            "processing_mode": expert_results["processing_mode"],
            "tpu_optimized": True
        }
    
    async def _fallback_consensus(self, query: str, expert_pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback consensus when TPU optimization is not available."""
        
        # Simple fallback implementation
        selected_experts = expert_pool[:5]  # Select first 5 experts
        
        return {
            "selected_expert_indices": list(range(len(selected_experts))),
            "consensus_confidence": 0.75,
            "expected_quality": 8.0,
            "total_cost": sum(exp.get("cost_per_1k_tokens", 0.001) for exp in selected_experts),
            "processing_mode": "fallback",
            "tpu_optimized": False
        }
    
    async def _update_tpu_metrics(self, consensus_result: Dict[str, Any], processing_time: float):
        """Populate :class:`TPUConsensusMetrics` from the last run.

        All values are derived from the runtime state of this call. Fields
        that cannot be observed without a real TPU profiler (per-core FLOPS,
        memory bandwidth, network bandwidth) are reported as ``0.0`` rather
        than filled with synthetic noise. Callers that need those numbers
        must wire them in from an external profiler.

        The previous implementation filled most of these fields with
        ``np.random.uniform`` samples, which made every call produce
        different "metrics" even for identical inputs and masked missing
        instrumentation.
        """
        # --- Throughput ---------------------------------------------------
        self.metrics.consensus_operations_per_second = (
            1.0 / processing_time if processing_time > 0 else 0.0
        )

        # --- Live devices -------------------------------------------------
        try:
            live_devices = list(jax.devices()) if JAX_AVAILABLE else []
        except Exception:  # pragma: no cover - JAX backend discovery edge cases
            live_devices = []
        num_live = len(live_devices)

        # Core utilization proxy: 1.0 for each live device we can actually
        # place work on, 0.0 for nominal cores that do not exist on the
        # current host. This is a coarse occupancy indicator, not real
        # per-core utilization; a true value requires TPU hardware counters.
        occupied = min(num_live, self.config.tpu_cores)
        self.metrics.core_utilization = (
            [1.0] * occupied + [0.0] * max(0, self.config.tpu_cores - occupied)
        )

        # --- Memory utilization ------------------------------------------
        memory_util: List[float] = []
        for dev in live_devices[: self.config.tpu_cores]:
            stats_fn = getattr(dev, "memory_stats", None)
            if not callable(stats_fn):
                memory_util.append(0.0)
                continue
            try:
                info = stats_fn() or {}
                used = float(info.get("bytes_in_use", 0) or 0)
                limit = float(
                    info.get("bytes_limit")
                    or info.get("bytes_reservable_limit")
                    or 0
                )
                memory_util.append(used / limit if limit > 0 else 0.0)
            except Exception:  # pragma: no cover - backend-specific failures
                memory_util.append(0.0)
        while len(memory_util) < self.config.tpu_cores:
            memory_util.append(0.0)
        self.metrics.memory_utilization_per_core = memory_util

        # --- Hardware counters we cannot measure without a profiler ------
        self.metrics.flops_utilization = 0.0
        self.metrics.memory_bandwidth_utilization = 0.0
        self.metrics.network_bandwidth_utilization = 0.0
        self.metrics.compilation_time_ms = 0.0

        # --- Consensus quality from the real result ----------------------
        self.metrics.distributed_consensus_accuracy = float(
            consensus_result.get("consensus_confidence", 0.0)
        )

        # Cross-core agreement: inverse of the spread across the selected
        # experts' similarity scores. Values clamped to [0, 1].
        selected = consensus_result.get("selected_similarities") or []
        if selected:
            arr = np.asarray(selected, dtype=np.float32)
            spread = float(arr.max() - arr.min()) if arr.size > 1 else 0.0
            self.metrics.cross_core_agreement = max(0.0, min(1.0, 1.0 - spread))
        else:
            self.metrics.cross_core_agreement = self.metrics.distributed_consensus_accuracy

        # Mesh synchronization efficiency: 1.0 when the mesh is up, 0.0 otherwise.
        # A finer-grained value would require XLA collective profiling.
        self.metrics.mesh_synchronization_efficiency = (
            1.0 if self.mesh is not None else 0.0
        )
    
    def _get_tpu_metrics_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-friendly snapshot of the most recently observed metrics.

        Uses ``0.0`` for aggregates whose underlying list is empty (i.e. the
        optimizer has not executed a consensus round yet) instead of letting
        NumPy emit a ``RuntimeWarning`` and return NaN.
        """
        core_util = self.metrics.core_utilization or [0.0]
        mem_util = self.metrics.memory_utilization_per_core or [0.0]

        return {
            "core_utilization": {
                "avg": float(np.mean(core_util)),
                "min": float(np.min(core_util)),
                "max": float(np.max(core_util)),
                "std": float(np.std(core_util)),
            },
            "memory_utilization": {
                "avg_per_core": float(np.mean(mem_util)),
                "total_used_gb": float(np.sum(mem_util)) * self.config.memory_per_core_gb,
                "efficiency": float(np.mean(mem_util)),
            },
            "performance": {
                "consensus_ops_per_sec": self.metrics.consensus_operations_per_second,
                "flops_utilization": self.metrics.flops_utilization,
                "memory_bandwidth_utilization": self.metrics.memory_bandwidth_utilization,
                "mesh_efficiency": self.metrics.mesh_synchronization_efficiency,
            },
            "consensus_quality": {
                "distributed_accuracy": self.metrics.distributed_consensus_accuracy,
                "cross_core_agreement": self.metrics.cross_core_agreement,
            },
        }
    
    async def benchmark_tpu_consensus(
        self,
        test_queries: List[str],
        expert_pools: List[List[Dict[str, Any]]],
        consensus_modes: List[TPUConsensusMode] = None
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark TPU consensus performance across different modes."""
        
        if consensus_modes is None:
            consensus_modes = [
                TPUConsensusMode.SINGLE_CORE,
                TPUConsensusMode.MULTI_CORE_PARALLEL,
                TPUConsensusMode.MESH_DISTRIBUTED,
                TPUConsensusMode.PIPELINE_PARALLEL
            ]
        
        benchmark_results = {}
        
        for mode in consensus_modes:
            logger.info(f"Benchmarking TPU consensus mode: {mode.value}")
            
            mode_results = {
                "avg_consensus_time_ms": 0.0,
                "consensus_ops_per_second": 0.0,
                "avg_core_utilization": 0.0,
                "memory_efficiency": 0.0,
                "consensus_accuracy": 0.0,
                "cross_core_agreement": 0.0
            }
            
            total_time = 0.0
            
            for i, query in enumerate(test_queries):
                expert_pool = expert_pools[i % len(expert_pools)]
                
                consensus_start = time.time()
                
                result = await self.optimize_consensus_for_tpu_v6(
                    query=query,
                    expert_pool=expert_pool,
                    consensus_mode=mode
                )
                
                consensus_time = time.time() - consensus_start
                total_time += consensus_time
                
                # Accumulate metrics
                mode_results["avg_consensus_time_ms"] += consensus_time * 1000
                mode_results["consensus_accuracy"] += result.get("consensus_confidence", 0.8)
                
                # Get TPU metrics
                tpu_metrics = result.get("tpu_metrics", {})
                if tpu_metrics:
                    mode_results["avg_core_utilization"] += tpu_metrics.get("core_utilization", {}).get("avg", 0.8)
                    mode_results["memory_efficiency"] += tpu_metrics.get("memory_utilization", {}).get("efficiency", 0.7)
                    mode_results["cross_core_agreement"] += tpu_metrics.get("consensus_quality", {}).get("cross_core_agreement", 0.9)
            
            num_queries = len(test_queries)
            
            # Calculate averages
            mode_results["avg_consensus_time_ms"] /= num_queries
            mode_results["consensus_ops_per_second"] = num_queries / total_time
            mode_results["avg_core_utilization"] /= num_queries
            mode_results["memory_efficiency"] /= num_queries
            mode_results["consensus_accuracy"] /= num_queries
            mode_results["cross_core_agreement"] /= num_queries
            
            benchmark_results[mode.value] = mode_results
        
        return benchmark_results
    
    def get_tpu_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive TPU optimization status."""
        
        return {
            "tpu_configuration": {
                "cores": self.config.tpu_cores,
                "memory_per_core_gb": self.config.memory_per_core_gb,
                "total_memory_tb": self.config.total_memory_tb,
                "mesh_shape": self.config.mesh_shape,
                "consensus_mode": self.config.consensus_mode.value
            },
            "optimization_features": {
                "mesh_initialized": self.mesh is not None,
                "jax_available": JAX_AVAILABLE,
                "compiled_functions": len(self.compiled_functions),
                "xla_optimization": self.config.enable_xla_optimization,
                "memory_mapping": self.config.enable_memory_mapping,
                "pipeline_parallelism": self.config.enable_pipeline_parallelism
            },
            "performance_status": {
                "consensus_ops_per_sec": self.metrics.consensus_operations_per_second,
                "avg_core_utilization": float(np.mean(self.metrics.core_utilization)) if self.metrics.core_utilization else 0.0,
                "memory_efficiency": float(np.mean(self.metrics.memory_utilization_per_core)) if self.metrics.memory_utilization_per_core else 0.0,
                "flops_utilization": self.metrics.flops_utilization,
                "mesh_efficiency": self.metrics.mesh_synchronization_efficiency
            },
            "hf_pro_integration": {
                "enabled": self.config.enable_hf_pro_features,
                "h200_distributed": self.config.h200_distributed_hosting,
                "credits_multiplier": self.config.inference_credits_multiplier
            }
        }


# Factory functions
def create_tpu_v6_consensus_optimizer(
    tpu_cores: int = 64,
    mesh_shape: Tuple[int, int] = (8, 8),
    consensus_mode: TPUConsensusMode = TPUConsensusMode.MESH_DISTRIBUTED,
    **kwargs
) -> TPUv6ConsensusOptimizer:
    """Creates TPU v6 consensus optimizer."""
    
    config = TPUv6ConsensusConfig(
        tpu_cores=tpu_cores,
        mesh_shape=mesh_shape,
        consensus_mode=consensus_mode,
        **kwargs
    )
    
    return TPUv6ConsensusOptimizer(config)


# Export main components
__all__ = [
    'TPUv6ConsensusOptimizer',
    'TPUv6ConsensusConfig',
    'TPUConsensusMode',
    'TPUConsensusMetrics',
    'create_tpu_v6_consensus_optimizer'
]


if __name__ == "__main__":
    # Example usage and benchmarking
    async def main():
        # Create TPU v6 optimizer
        optimizer = create_tpu_v6_consensus_optimizer(
            tpu_cores=64,
            mesh_shape=(8, 8),
            consensus_mode=TPUConsensusMode.MESH_DISTRIBUTED
        )
        
        # Deterministic example expert pool. Values are spread across a
        # reproducible range so successive runs and machines see identical
        # inputs (previously this block used np.random which defeated the
        # purpose of a demo).
        expert_pool = [
            {
                "name": f"Expert_{i}",
                "weight": 1.0 + (i % 5) * 0.1,
                "quality_score": 8.0 + (i % 10) * 0.2,
                "cost_per_1k_tokens": 0.001 + (i % 7) * 0.001,
            }
            for i in range(20)
        ]
        
        # Test queries
        test_queries = [
            "Explain quantum computing applications",
            "Develop machine learning algorithms",
            "Analyze complex mathematical proofs",
            "Design distributed systems architecture",
            "Optimize neural network performance"
        ]
        
        logger.info(" TPU v6 Consensus Optimization Test")
        logger.info("=" * 50)
        
        if JAX_AVAILABLE:
            # Run consensus optimization
            for i, query in enumerate(test_queries):
                logger.info(f"\n Query {i+1}: {query}")
                
                result = await optimizer.optimize_consensus_for_tpu_v6(
                    query=query,
                    expert_pool=expert_pool,
                    consensus_mode=TPUConsensusMode.MESH_DISTRIBUTED
                )
                
                logger.info(f"  Consensus Confidence: {result['consensus_confidence']:.2f}")
                logger.info(f"  Expected Quality: {result['expected_quality']:.1f}")
                logger.info(f"  Total Cost: ${result['total_cost']:.4f}")
                logger.info(f"  Processing Time: {result['processing_time_ms']:.1f}ms")
                logger.info(f"  TPU Cores Used: {result['tpu_cores_used']}")
            
            # Run benchmarks
            logger.info(f"\n Running TPU Consensus Benchmarks...")
            
            benchmark_results = await optimizer.benchmark_tpu_consensus(
                test_queries[:3],  # Use subset for benchmarking
                [expert_pool] * 3
            )
            
            logger.info(f"\n TPU Benchmark Results:")
            for mode, metrics in benchmark_results.items():
                logger.info(f"\n{mode.upper()}:")
                logger.info(f"  Consensus Time: {metrics['avg_consensus_time_ms']:.1f}ms")
                logger.info(f"  Ops/Second: {metrics['consensus_ops_per_second']:.1f}")
                logger.info(f"  Core Utilization: {metrics['avg_core_utilization']:.1%}")
                logger.info(f"  Memory Efficiency: {metrics['memory_efficiency']:.1%}")
                logger.info(f"  Consensus Accuracy: {metrics['consensus_accuracy']:.2f}")
            
            # Get optimization status
            status = optimizer.get_tpu_optimization_status()
            logger.info(f"\n TPU Optimization Status:")
            logger.info(f"  Mesh Initialized: {status['optimization_features']['mesh_initialized']}")
            logger.info(f"  Compiled Functions: {status['optimization_features']['compiled_functions']}")
            logger.info(f"  Performance: {status['performance_status']['consensus_ops_per_sec']:.1f} ops/sec")
            logger.info(f"  Core Utilization: {status['performance_status']['avg_core_utilization']:.1%}")
            
        else:
            logger.warning(" JAX not available - TPU v6 optimizations disabled")
            logger.warning("Using fallback implementations")
    
    import asyncio
    asyncio.run(main())