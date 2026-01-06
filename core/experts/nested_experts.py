"""
Nested Expert Hierarchy for CapibaraGPT

This module implements hierarchical expert nesting inspired by Nested Learning
(Behrouz et al., NeurIPS 2025). The core idea is to organize experts in a
multi-level hierarchy where each level updates at different frequencies,
preventing catastrophic forgetting and enabling efficient specialization.

Key Features:
- Three-level expert hierarchy (Meta, Domain, Micro)
- Different update frequencies per level (1000, 100, 10 steps)
- Hierarchical routing with parent-child knowledge transfer
- Load balancing across all hierarchy levels
- Expert specialization scoring and monitoring

Expert Levels:
- Meta-Experts (4 experts, slow updates every 1000 steps)
  → General knowledge, stable learning
- Domain-Experts (16 experts, medium updates every 100 steps)
  → Domain-specific knowledge, inherits from meta-experts
- Micro-Experts (64 experts, fast updates every 10 steps)
  → Task-specific knowledge, inherits from domain-experts

This implements the key Nested Learning principle of separating concerns
across time scales for different types of knowledge.

Reference:
    Behrouz, A.; Razaviyayn, M.; Mirrokni, V.; Zhong, P. (2025).
    "Nested Learning: The Illusion of Deep Learning Architectures"
    NeurIPS 2025
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import json

try:
    import numpy as np
except ImportError:
    # Fallback for systems without numpy
    class np:  # type: ignore
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if x else 0

        @staticmethod
        def argmax(x):
            return max(range(len(x)), key=lambda i: x[i]) if x else 0

        @staticmethod
        def softmax(x):
            if not x:
                return []
            exp_x = [2.71828 ** val for val in x]
            sum_exp = sum(exp_x)
            return [val / sum_exp for val in exp_x]

        @staticmethod
        def random():
            class _Random:
                @staticmethod
                def choice(arr):
                    import random
                    return random.choice(arr)

                @staticmethod
                def uniform(low, high):
                    import random
                    return random.uniform(low, high)

            return _Random()

logger = logging.getLogger(__name__)


@dataclass
class ExpertConfig:
    """Configuration for a single expert."""

    expert_id: int
    expert_type: str  # 'meta', 'domain', or 'micro'
    specialization: str  # What this expert specializes in
    hidden_size: int = 768
    update_freq: int = 10  # Update frequency in steps
    parent_id: Optional[int] = None  # Parent expert ID
    learning_rate: float = 1e-4
    capacity_factor: float = 1.0  # For load balancing


@dataclass
class ExpertState:
    """Runtime state of an expert."""

    usage_count: int = 0
    total_loss: float = 0.0
    specialization_score: float = 0.0
    last_update_step: int = 0
    performance_history: List[float] = field(default_factory=list)
    parent_context: Optional[Dict[str, Any]] = None
    active: bool = True


class Expert:
    """
    Base expert class with hierarchical capabilities.

    Each expert can have a parent expert from which it inherits knowledge,
    and child experts that inherit from it.
    """

    def __init__(self, config: ExpertConfig):
        self.config = config
        self.state = ExpertState()

        # Simulated parameters (in real implementation, these would be neural network weights)
        self.parameters = self._initialize_parameters()

        # Children experts
        self.children: List[int] = []

        logger.debug(
            f"Expert {config.expert_id} ({config.expert_type}:{config.specialization}) initialized"
        )

    def _initialize_parameters(self) -> Dict[str, Any]:
        """Initialize expert parameters."""
        # In real implementation, this would initialize neural network weights
        return {
            'weights': [0.5] * self.config.hidden_size,
            'bias': [0.0] * self.config.hidden_size,
            'specialization_weights': {},
        }

    def should_update(self, current_step: int) -> bool:
        """Check if expert should update at current step."""
        return current_step % self.config.update_freq == 0

    def forward(
        self,
        inputs: Any,
        parent_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through the expert.

        Args:
            inputs: Input data
            parent_context: Optional context from parent expert

        Returns:
            Expert output with metadata
        """
        # Store parent context
        if parent_context:
            self.state.parent_context = parent_context

        # Simulate forward pass (in real implementation, this would be neural network forward)
        output = {
            'expert_id': self.config.expert_id,
            'expert_type': self.config.expert_type,
            'specialization': self.config.specialization,
            'output': inputs,  # Simplified
            'confidence': 0.5 + self.state.specialization_score * 0.5,
            'used_parent_context': parent_context is not None,
        }

        self.state.usage_count += 1

        return output

    def update(
        self,
        loss: float,
        inputs: Any,
        current_step: int,
        learning_rate: Optional[float] = None
    ):
        """
        Update expert parameters.

        Args:
            loss: Loss value
            inputs: Training inputs
            current_step: Current training step
            learning_rate: Optional override for learning rate
        """
        if not self.should_update(current_step):
            return

        lr = learning_rate or self.config.learning_rate

        # Simulate parameter update (in real implementation, this would be gradient descent)
        # Update parameters based on loss
        # self.parameters['weights'] = [w - lr * grad for w, grad in ...]

        # Track state
        self.state.total_loss += loss
        self.state.last_update_step = current_step
        self.state.performance_history.append(loss)

        # Keep history bounded
        if len(self.state.performance_history) > 100:
            self.state.performance_history.pop(0)

        logger.debug(
            f"Expert {self.config.expert_id} updated at step {current_step}, loss={loss:.4f}"
        )

    def update_from_children(self, children_experts: List['Expert']):
        """
        Update this expert based on children's performance.

        This is a key Nested Learning feature - higher-level experts learn
        from their children's specializations.

        Args:
            children_experts: List of child experts
        """
        if not children_experts:
            return

        # Aggregate children performance
        children_losses = [
            child.state.total_loss / max(child.state.usage_count, 1)
            for child in children_experts
            if child.config.parent_id == self.config.expert_id
        ]

        if children_losses:
            avg_child_loss = sum(children_losses) / len(children_losses)

            # Update specialization score based on children performance
            # Better children = better specialization
            self.state.specialization_score = 1.0 / (1.0 + avg_child_loss)

            logger.debug(
                f"Expert {self.config.expert_id} updated from {len(children_losses)} children, "
                f"spec_score={self.state.specialization_score:.3f}"
            )

    def get_context(self) -> Dict[str, Any]:
        """Get context to pass to child experts."""
        return {
            'expert_id': self.config.expert_id,
            'specialization': self.config.specialization,
            'specialization_score': self.state.specialization_score,
            'parameters': self.parameters.copy(),  # In real impl, would be embeddings
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get expert statistics."""
        avg_loss = (
            self.state.total_loss / self.state.usage_count
            if self.state.usage_count > 0
            else 0.0
        )

        return {
            'expert_id': self.config.expert_id,
            'expert_type': self.config.expert_type,
            'specialization': self.config.specialization,
            'usage_count': self.state.usage_count,
            'avg_loss': avg_loss,
            'specialization_score': self.state.specialization_score,
            'last_update_step': self.state.last_update_step,
            'parent_id': self.config.parent_id,
            'num_children': len(self.children),
            'active': self.state.active,
        }


@dataclass
class HierarchicalRoutingConfig:
    """Configuration for hierarchical routing."""

    routing_strategy: str = 'confidence'  # 'confidence', 'random', 'round_robin', 'learned'
    temperature: float = 1.0  # Temperature for softmax routing
    top_k_meta: int = 2  # Number of meta experts to consider
    top_k_domain: int = 4  # Number of domain experts per meta
    top_k_micro: int = 2  # Number of micro experts per domain
    use_load_balancing: bool = True
    load_balance_weight: float = 0.01


class HierarchicalRouter:
    """
    Hierarchical router that routes through the expert hierarchy.

    Routing happens in stages:
    1. Select top-k meta-experts
    2. For each meta-expert, select top-k domain-experts
    3. For each domain-expert, select top-k micro-experts
    """

    def __init__(self, config: HierarchicalRoutingConfig):
        self.config = config
        self.routing_history = []
        self.routing_stats = defaultdict(int)

    def select_meta_expert(
        self,
        inputs: Any,
        meta_experts: List[Expert]
    ) -> List[int]:
        """
        Select meta-expert(s) for the input.

        Args:
            inputs: Input data
            meta_experts: List of meta-level experts

        Returns:
            List of selected meta-expert IDs
        """
        if not meta_experts:
            return []

        # Calculate routing scores
        scores = self._calculate_routing_scores(inputs, meta_experts)

        # Apply softmax with temperature
        probs = self._softmax_with_temperature(scores, self.config.temperature)

        # Select top-k
        top_k = min(self.config.top_k_meta, len(meta_experts))
        top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]

        selected_ids = [meta_experts[i].config.expert_id for i in top_indices]

        # Track routing
        for expert_id in selected_ids:
            self.routing_stats[f'meta_{expert_id}'] += 1

        return selected_ids

    def select_domain_expert(
        self,
        inputs: Any,
        domain_experts: List[Expert],
        parent_meta: int
    ) -> List[int]:
        """
        Select domain-expert(s) under a specific meta-expert.

        Args:
            inputs: Input data
            domain_experts: List of domain-level experts
            parent_meta: Parent meta-expert ID

        Returns:
            List of selected domain-expert IDs
        """
        # Filter to only children of this meta-expert
        candidate_experts = [
            expert for expert in domain_experts
            if expert.config.parent_id == parent_meta
        ]

        if not candidate_experts:
            return []

        # Calculate scores
        scores = self._calculate_routing_scores(inputs, candidate_experts)

        # Apply softmax
        probs = self._softmax_with_temperature(scores, self.config.temperature)

        # Select top-k
        top_k = min(self.config.top_k_domain, len(candidate_experts))
        top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]

        selected_ids = [candidate_experts[i].config.expert_id for i in top_indices]

        # Track routing
        for expert_id in selected_ids:
            self.routing_stats[f'domain_{expert_id}'] += 1

        return selected_ids

    def select_micro_expert(
        self,
        inputs: Any,
        micro_experts: List[Expert],
        parent_domain: int
    ) -> List[int]:
        """
        Select micro-expert(s) under a specific domain-expert.

        Args:
            inputs: Input data
            micro_experts: List of micro-level experts
            parent_domain: Parent domain-expert ID

        Returns:
            List of selected micro-expert IDs
        """
        # Filter to only children of this domain-expert
        candidate_experts = [
            expert for expert in micro_experts
            if expert.config.parent_id == parent_domain
        ]

        if not candidate_experts:
            return []

        # Calculate scores
        scores = self._calculate_routing_scores(inputs, candidate_experts)

        # Apply softmax
        probs = self._softmax_with_temperature(scores, self.config.temperature)

        # Select top-k
        top_k = min(self.config.top_k_micro, len(candidate_experts))
        top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]

        selected_ids = [candidate_experts[i].config.expert_id for i in top_indices]

        # Track routing
        for expert_id in selected_ids:
            self.routing_stats[f'micro_{expert_id}'] += 1

        return selected_ids

    def _calculate_routing_scores(
        self,
        inputs: Any,
        experts: List[Expert]
    ) -> List[float]:
        """
        Calculate routing scores for experts.

        Args:
            inputs: Input data
            experts: List of experts to score

        Returns:
            List of scores
        """
        scores = []

        for expert in experts:
            # Base score from specialization
            base_score = expert.state.specialization_score

            # Add load balancing term
            if self.config.use_load_balancing:
                usage_penalty = expert.state.usage_count * self.config.load_balance_weight
                base_score -= usage_penalty

            # Ensure non-negative
            scores.append(max(0.0, base_score))

        # Normalize
        total = sum(scores)
        if total > 0:
            scores = [s / total for s in scores]
        else:
            # Uniform if all zeros
            scores = [1.0 / len(experts)] * len(experts)

        return scores

    def _softmax_with_temperature(
        self,
        scores: List[float],
        temperature: float
    ) -> List[float]:
        """Apply softmax with temperature."""
        if not scores:
            return []

        # Scale by temperature
        scaled = [s / temperature for s in scores]

        # Softmax
        exp_scores = [2.71828 ** s for s in scaled]
        sum_exp = sum(exp_scores)

        if sum_exp == 0:
            return [1.0 / len(scores)] * len(scores)

        return [e / sum_exp for e in exp_scores]

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total_routes = sum(self.routing_stats.values())

        return {
            'total_routing_decisions': total_routes,
            'routing_distribution': dict(self.routing_stats),
            'meta_usage': {
                k: v for k, v in self.routing_stats.items()
                if k.startswith('meta_')
            },
            'domain_usage': {
                k: v for k, v in self.routing_stats.items()
                if k.startswith('domain_')
            },
            'micro_usage': {
                k: v for k, v in self.routing_stats.items()
                if k.startswith('micro_')
            },
        }


@dataclass
class NestedExpertConfig:
    """Configuration for nested expert hierarchy."""

    # Number of experts at each level
    num_meta_experts: int = 4
    num_domain_experts: int = 16
    num_micro_experts: int = 64

    # Update frequencies (KEY to Nested Learning)
    meta_update_freq: int = 1000  # Slow, stable learning
    domain_update_freq: int = 100  # Medium updates
    micro_update_freq: int = 10  # Fast adaptation

    # Expert sizes
    meta_hidden_size: int = 768
    domain_hidden_size: int = 512
    micro_hidden_size: int = 256

    # Learning rates
    meta_learning_rate: float = 1e-5  # Slow learning
    domain_learning_rate: float = 1e-4  # Medium learning
    micro_learning_rate: float = 1e-3  # Fast learning

    # Routing config
    routing_config: HierarchicalRoutingConfig = field(
        default_factory=HierarchicalRoutingConfig
    )

    # Specializations
    meta_specializations: List[str] = field(default_factory=lambda: [
        'reasoning',
        'creative',
        'factual',
        'technical',
    ])

    domain_specializations: List[str] = field(default_factory=lambda: [
        'mathematics', 'logic', 'problem_solving', 'analysis',  # reasoning
        'artistic', 'storytelling', 'imagination', 'expression',  # creative
        'knowledge', 'retrieval', 'accuracy', 'citation',  # factual
        'coding', 'engineering', 'scientific', 'systems',  # technical
    ])


class NestedExpertHierarchy:
    """
    Nested Expert Hierarchy implementing multi-level expert organization.

    This implements the key Nested Learning concept where experts are organized
    in a hierarchy with different update frequencies at each level:

    Level 0 (Meta): 4 experts, update every 1000 steps
       ↓
    Level 1 (Domain): 16 experts (4 per meta), update every 100 steps
       ↓
    Level 2 (Micro): 64 experts (4 per domain), update every 10 steps

    This separation of time scales prevents catastrophic forgetting while
    enabling both stable long-term knowledge (meta) and fast task adaptation (micro).
    """

    def __init__(self, config: Optional[NestedExpertConfig] = None):
        self.config = config or NestedExpertConfig()

        # Create expert hierarchy
        self.meta_experts: List[Expert] = []
        self.domain_experts: List[Expert] = []
        self.micro_experts: List[Expert] = []

        # Initialize experts
        self._initialize_hierarchy()

        # Hierarchical router
        self.router = HierarchicalRouter(self.config.routing_config)

        # State tracking
        self.current_step = 0
        self.update_history = []

        logger.info("🏗️ Nested Expert Hierarchy initialized")
        logger.info(f"   Meta-Experts: {len(self.meta_experts)} (freq={self.config.meta_update_freq})")
        logger.info(f"   Domain-Experts: {len(self.domain_experts)} (freq={self.config.domain_update_freq})")
        logger.info(f"   Micro-Experts: {len(self.micro_experts)} (freq={self.config.micro_update_freq})")

    def _initialize_hierarchy(self):
        """Initialize the expert hierarchy."""
        expert_id_counter = 0

        # Level 0: Meta-Experts
        for i in range(self.config.num_meta_experts):
            specialization = self.config.meta_specializations[
                i % len(self.config.meta_specializations)
            ]

            expert = Expert(ExpertConfig(
                expert_id=expert_id_counter,
                expert_type='meta',
                specialization=specialization,
                hidden_size=self.config.meta_hidden_size,
                update_freq=self.config.meta_update_freq,
                parent_id=None,
                learning_rate=self.config.meta_learning_rate,
            ))

            self.meta_experts.append(expert)
            expert_id_counter += 1

        # Level 1: Domain-Experts
        domains_per_meta = self.config.num_domain_experts // self.config.num_meta_experts

        for meta_idx, meta_expert in enumerate(self.meta_experts):
            for i in range(domains_per_meta):
                domain_idx = meta_idx * domains_per_meta + i
                specialization = self.config.domain_specializations[
                    domain_idx % len(self.config.domain_specializations)
                ]

                expert = Expert(ExpertConfig(
                    expert_id=expert_id_counter,
                    expert_type='domain',
                    specialization=specialization,
                    hidden_size=self.config.domain_hidden_size,
                    update_freq=self.config.domain_update_freq,
                    parent_id=meta_expert.config.expert_id,
                    learning_rate=self.config.domain_learning_rate,
                ))

                self.domain_experts.append(expert)
                meta_expert.children.append(expert.config.expert_id)
                expert_id_counter += 1

        # Level 2: Micro-Experts
        micros_per_domain = self.config.num_micro_experts // self.config.num_domain_experts

        for domain_expert in self.domain_experts:
            for i in range(micros_per_domain):
                # Generate task-specific specializations
                specialization = f"{domain_expert.config.specialization}_task_{i}"

                expert = Expert(ExpertConfig(
                    expert_id=expert_id_counter,
                    expert_type='micro',
                    specialization=specialization,
                    hidden_size=self.config.micro_hidden_size,
                    update_freq=self.config.micro_update_freq,
                    parent_id=domain_expert.config.expert_id,
                    learning_rate=self.config.micro_learning_rate,
                ))

                self.micro_experts.append(expert)
                domain_expert.children.append(expert.config.expert_id)
                expert_id_counter += 1

        logger.info(f"Hierarchy initialized with {expert_id_counter} total experts")

    def forward(self, inputs: Any, step: int) -> Dict[str, Any]:
        """
        Forward pass through the hierarchical expert system.

        Args:
            inputs: Input data
            step: Current step

        Returns:
            Aggregated output from selected experts
        """
        self.current_step = step

        # Stage 1: Select meta-experts
        selected_meta_ids = self.router.select_meta_expert(inputs, self.meta_experts)

        if not selected_meta_ids:
            return {'error': 'No meta-experts selected'}

        all_outputs = []
        routing_path = {
            'meta': [],
            'domain': [],
            'micro': []
        }

        # For each selected meta-expert
        for meta_id in selected_meta_ids:
            meta_expert = self._get_expert_by_id(meta_id, self.meta_experts)
            if not meta_expert:
                continue

            # Execute meta-expert
            meta_output = meta_expert.forward(inputs, parent_context=None)
            meta_context = meta_expert.get_context()

            routing_path['meta'].append(meta_id)

            # Stage 2: Select domain-experts under this meta-expert
            selected_domain_ids = self.router.select_domain_expert(
                inputs,
                self.domain_experts,
                parent_meta=meta_id
            )

            # For each selected domain-expert
            for domain_id in selected_domain_ids:
                domain_expert = self._get_expert_by_id(domain_id, self.domain_experts)
                if not domain_expert:
                    continue

                # Execute domain-expert with meta context
                domain_output = domain_expert.forward(inputs, parent_context=meta_context)
                domain_context = domain_expert.get_context()

                routing_path['domain'].append(domain_id)

                # Stage 3: Select micro-experts under this domain-expert
                selected_micro_ids = self.router.select_micro_expert(
                    inputs,
                    self.micro_experts,
                    parent_domain=domain_id
                )

                # For each selected micro-expert
                for micro_id in selected_micro_ids:
                    micro_expert = self._get_expert_by_id(micro_id, self.micro_experts)
                    if not micro_expert:
                        continue

                    # Execute micro-expert with domain context
                    micro_output = micro_expert.forward(inputs, parent_context=domain_context)

                    routing_path['micro'].append(micro_id)

                    # Collect output
                    all_outputs.append({
                        'micro_output': micro_output,
                        'domain_output': domain_output,
                        'meta_output': meta_output,
                        'path': {
                            'meta': meta_id,
                            'domain': domain_id,
                            'micro': micro_id,
                        }
                    })

        # Aggregate outputs
        aggregated = self._aggregate_outputs(all_outputs)
        aggregated['routing_path'] = routing_path
        aggregated['num_experts_used'] = len(all_outputs)

        return aggregated

    def update_experts(self, step: int, loss: float, inputs: Any):
        """
        Update experts based on their update frequencies.

        This is the KEY to Nested Learning - different levels update at different rates.

        Args:
            step: Current training step
            loss: Loss value
            inputs: Training inputs
        """
        self.current_step = step

        update_record = {
            'step': step,
            'loss': loss,
            'updated_levels': []
        }

        # Micro-experts: Update frequently (every 10 steps)
        if step > 0 and step % self.config.micro_update_freq == 0:
            for expert in self.micro_experts:
                expert.update(loss, inputs, step)
            update_record['updated_levels'].append('micro')
            logger.debug(f"Step {step}: Updated {len(self.micro_experts)} micro-experts")

        # Domain-experts: Update less frequently (every 100 steps)
        if step > 0 and step % self.config.domain_update_freq == 0:
            for expert in self.domain_experts:
                # Update from children first
                children = [e for e in self.micro_experts if e.config.parent_id == expert.config.expert_id]
                expert.update_from_children(children)

                # Then update parameters
                expert.update(loss, inputs, step)

            update_record['updated_levels'].append('domain')
            logger.info(f"Step {step}: Updated {len(self.domain_experts)} domain-experts")

        # Meta-experts: Update rarely (every 1000 steps)
        if step > 0 and step % self.config.meta_update_freq == 0:
            for expert in self.meta_experts:
                # Update from children first
                children = [e for e in self.domain_experts if e.config.parent_id == expert.config.expert_id]
                expert.update_from_children(children)

                # Then update parameters
                expert.update(loss, inputs, step)

            update_record['updated_levels'].append('meta')
            logger.info(f"Step {step}: Updated {len(self.meta_experts)} meta-experts")

        if update_record['updated_levels']:
            self.update_history.append(update_record)

    def _get_expert_by_id(self, expert_id: int, expert_list: List[Expert]) -> Optional[Expert]:
        """Get expert by ID from a list."""
        for expert in expert_list:
            if expert.config.expert_id == expert_id:
                return expert
        return None

    def _aggregate_outputs(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate outputs from multiple experts."""
        if not outputs:
            return {'output': None, 'confidence': 0.0}

        # Simple averaging (in real implementation, would be more sophisticated)
        avg_confidence = sum(o['micro_output']['confidence'] for o in outputs) / len(outputs)

        return {
            'output': outputs[0]['micro_output']['output'],  # Use first for simplicity
            'confidence': avg_confidence,
            'num_paths': len(outputs),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the hierarchy."""
        meta_stats = [e.get_statistics() for e in self.meta_experts]
        domain_stats = [e.get_statistics() for e in self.domain_experts]
        micro_stats = [e.get_statistics() for e in self.micro_experts]

        return {
            'current_step': self.current_step,
            'hierarchy_structure': {
                'num_meta': len(self.meta_experts),
                'num_domain': len(self.domain_experts),
                'num_micro': len(self.micro_experts),
                'total_experts': len(self.meta_experts) + len(self.domain_experts) + len(self.micro_experts),
            },
            'update_frequencies': {
                'meta': self.config.meta_update_freq,
                'domain': self.config.domain_update_freq,
                'micro': self.config.micro_update_freq,
            },
            'expert_statistics': {
                'meta': meta_stats,
                'domain': domain_stats,
                'micro': micro_stats,
            },
            'routing_statistics': self.router.get_routing_statistics(),
            'total_updates': len(self.update_history),
        }

    def get_hierarchy_visualization(self) -> str:
        """Get a text visualization of the hierarchy."""
        lines = []
        lines.append("=" * 60)
        lines.append("NESTED EXPERT HIERARCHY")
        lines.append("=" * 60)

        for meta_expert in self.meta_experts:
            lines.append(f"\n📊 Meta-Expert {meta_expert.config.expert_id}: {meta_expert.config.specialization}")
            lines.append(f"   Usage: {meta_expert.state.usage_count}, Score: {meta_expert.state.specialization_score:.3f}")

            # Find domain children
            domain_children = [e for e in self.domain_experts if e.config.parent_id == meta_expert.config.expert_id]

            for domain_expert in domain_children:
                lines.append(f"   ├─ 🎯 Domain {domain_expert.config.expert_id}: {domain_expert.config.specialization}")
                lines.append(f"   │  Usage: {domain_expert.state.usage_count}, Score: {domain_expert.state.specialization_score:.3f}")

                # Find micro children
                micro_children = [e for e in self.micro_experts if e.config.parent_id == domain_expert.config.expert_id]

                for i, micro_expert in enumerate(micro_children):
                    is_last = i == len(micro_children) - 1
                    connector = "└─" if is_last else "├─"
                    lines.append(f"   │  {connector} ⚡ Micro {micro_expert.config.expert_id}: {micro_expert.config.specialization}")
                    lines.append(f"   │     Usage: {micro_expert.state.usage_count}")

        lines.append("=" * 60)

        return "\n".join(lines)


# Factory function
def create_nested_expert_hierarchy(
    config: Optional[NestedExpertConfig] = None
) -> NestedExpertHierarchy:
    """Create a nested expert hierarchy instance."""
    return NestedExpertHierarchy(config)


# Global instance
_global_nested_experts: Optional[NestedExpertHierarchy] = None


def get_global_nested_experts() -> NestedExpertHierarchy:
    """Get the global nested expert hierarchy instance."""
    global _global_nested_experts
    if _global_nested_experts is None:
        _global_nested_experts = create_nested_expert_hierarchy()
    return _global_nested_experts


def main():
    """Test the nested expert hierarchy."""
    logging.basicConfig(level=logging.INFO)
    logger.info("🏗️ Nested Expert Hierarchy - Testing Mode")

    # Create hierarchy
    hierarchy = create_nested_expert_hierarchy()

    # Print visualization
    print(hierarchy.get_hierarchy_visualization())

    # Simulate some forward passes
    logger.info("\nSimulating forward passes...")
    for step in range(50):
        # Simulate input
        inputs = f"input_{step}"

        # Forward pass
        output = hierarchy.forward(inputs, step)

        # Update experts
        hierarchy.update_experts(step, loss=0.5 - step * 0.01, inputs=inputs)

        if step % 10 == 0:
            logger.info(f"\nStep {step}:")
            logger.info(f"  Experts used: {output['num_experts_used']}")
            logger.info(f"  Confidence: {output['confidence']:.3f}")
            logger.info(f"  Routing path: {output['routing_path']}")

    # Final statistics
    stats = hierarchy.get_statistics()
    logger.info(f"\n📊 Final Statistics:")
    logger.info(f"  Total experts: {stats['hierarchy_structure']['total_experts']}")
    logger.info(f"  Total updates: {stats['total_updates']}")

    routing_stats = stats['routing_statistics']
    logger.info(f"\n🔀 Routing Statistics:")
    logger.info(f"  Total routing decisions: {routing_stats['total_routing_decisions']}")

    # Print updated visualization
    print("\n" + hierarchy.get_hierarchy_visualization())


if __name__ == "__main__":
    main()
