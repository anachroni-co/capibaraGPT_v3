"""
Self-Modifying Router for CapibaraGPT

This module implements a self-modifying router inspired by Nested Learning
(Behrouz et al., NeurIPS 2025). The router learns not only routing decisions
but also learns to modify its own routing logic based on observed performance.

Key Features:
- Two-level nested optimization (routing policy + meta-routing policy)
- Different update frequencies (every step vs every 100 steps)
- Self-modification of routing parameters and strategies
- Performance trend analysis
- Automatic adaptation to changing conditions
- Structural policy changes based on meta-learning

Router Levels:
- Level 1: Routing Policy (updates every step)
  → Makes routing decisions
  → Learns from immediate feedback
- Level 2: Meta-Routing Policy (updates every 100 steps)
  → Analyzes routing trends
  → Modifies Level 1 policy structure and parameters
  → Optimizes routing strategy itself

This implements the key Nested Learning principle of learning at multiple
time scales, where the router learns both what to route AND how to route.

Reference:
    Behrouz, A.; Razaviyayn, M.; Mirrokni, V.; Zhong, P. (2025).
    "Nested Learning: The Illusion of Deep Learning Architectures"
    NeurIPS 2025
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
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
        def std(x):
            if not x:
                return 0
            mean = sum(x) / len(x)
            variance = sum((val - mean) ** 2 for val in x) / len(x)
            return variance ** 0.5

        @staticmethod
        def polyfit(x, y, deg):
            # Simple linear regression for deg=1
            if deg == 1 and len(x) == len(y) and len(x) >= 2:
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x[i] * y[i] for i in range(n))
                sum_xx = sum(x[i] ** 2 for i in range(n))

                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
                intercept = (sum_y - slope * sum_x) / n
                return [slope, intercept]
            return [0, 0]

        @staticmethod
        def clip(x, min_val, max_val):
            return max(min_val, min(max_val, x))

logger = logging.getLogger(__name__)


@dataclass
class RoutingPolicyConfig:
    """Configuration for a routing policy."""

    policy_id: str
    num_routes: int = 32  # Number of possible routes
    hidden_dim: int = 256  # Internal representation size

    # Routing parameters
    temperature: float = 1.0  # Softmax temperature
    top_k: int = 4  # Number of routes to select
    exploration_rate: float = 0.1  # Exploration vs exploitation

    # Learning parameters
    learning_rate: float = 1e-3
    update_freq: int = 1  # Update every step

    # Performance tracking
    track_performance: bool = True
    performance_window: int = 100


@dataclass
class MetaRoutingPolicyConfig:
    """Configuration for meta-routing policy."""

    # Meta-learning parameters
    meta_learning_rate: float = 1e-4
    meta_update_freq: int = 100  # Update every 100 steps

    # Analysis windows
    trend_analysis_window: int = 100
    performance_comparison_window: int = 50

    # Modification thresholds
    poor_performance_threshold: float = 0.5  # Below this triggers changes
    excellent_performance_threshold: float = 0.9  # Above this stabilizes

    # Modification strategies
    enable_temperature_adaptation: bool = True
    enable_top_k_adaptation: bool = True
    enable_exploration_adaptation: bool = True
    enable_structural_changes: bool = True

    # Bounds for adaptations
    temperature_range: Tuple[float, float] = (0.1, 2.0)
    top_k_range: Tuple[int, int] = (1, 8)
    exploration_range: Tuple[float, float] = (0.0, 0.3)


class RoutingPolicy:
    """
    Level 1: Routing policy that makes routing decisions.

    This learns to route based on inputs and feedback.
    """

    def __init__(self, config: RoutingPolicyConfig):
        self.config = config

        # Simulated routing network parameters
        # In real implementation, these would be neural network weights
        self.route_weights = [1.0 / config.num_routes] * config.num_routes
        self.route_biases = [0.0] * config.num_routes

        # Performance tracking per route
        self.route_performance = defaultdict(list)
        self.route_usage_count = defaultdict(int)

        # Recent decisions
        self.recent_decisions = deque(maxlen=config.performance_window)

        logger.debug(f"RoutingPolicy '{config.policy_id}' initialized with {config.num_routes} routes")

    def select_routes(
        self,
        inputs: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Select routes for the given input.

        Args:
            inputs: Input data
            context: Optional context information

        Returns:
            Tuple of (selected_route_ids, routing_metadata)
        """
        # Calculate routing scores
        scores = self._calculate_routing_scores(inputs, context)

        # Apply temperature scaling
        scaled_scores = [s / self.config.temperature for s in scores]

        # Softmax
        exp_scores = [2.71828 ** s for s in scaled_scores]
        sum_exp = sum(exp_scores)
        probabilities = [e / sum_exp for e in exp_scores]

        # Select top-k routes
        route_prob_pairs = list(enumerate(probabilities))
        route_prob_pairs.sort(key=lambda x: x[1], reverse=True)

        # Apply exploration
        import random
        if random.random() < self.config.exploration_rate:
            # Exploration: random selection
            selected_routes = random.sample(range(self.config.num_routes), self.config.top_k)
        else:
            # Exploitation: top-k selection
            selected_routes = [idx for idx, _ in route_prob_pairs[:self.config.top_k]]

        # Update usage counts
        for route_id in selected_routes:
            self.route_usage_count[route_id] += 1

        # Routing metadata
        metadata = {
            'probabilities': probabilities,
            'selected_routes': selected_routes,
            'exploration': random.random() < self.config.exploration_rate,
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
        }

        # Record decision
        self.recent_decisions.append({
            'routes': selected_routes,
            'scores': scores,
            'timestamp': datetime.now().isoformat()
        })

        return selected_routes, metadata

    def _calculate_routing_scores(
        self,
        inputs: Any,
        context: Optional[Dict[str, Any]]
    ) -> List[float]:
        """Calculate routing scores for each route."""
        # Simplified scoring (in real implementation, would use neural network)
        scores = []
        for i in range(self.config.num_routes):
            # Base score from weights
            score = self.route_weights[i] + self.route_biases[i]

            # Adjust based on recent performance
            if i in self.route_performance and self.route_performance[i]:
                recent_perf = self.route_performance[i][-10:]
                avg_perf = sum(recent_perf) / len(recent_perf)
                score += avg_perf * 0.5  # Performance bonus

            scores.append(score)

        return scores

    def update(
        self,
        selected_routes: List[int],
        performance_feedback: float,
        step: int
    ):
        """
        Update routing policy based on feedback.

        Args:
            selected_routes: Routes that were selected
            performance_feedback: Performance score [0, 1]
            step: Current training step
        """
        if step % self.config.update_freq != 0:
            return

        # Update performance for selected routes
        for route_id in selected_routes:
            self.route_performance[route_id].append(performance_feedback)

            # Keep history bounded
            if len(self.route_performance[route_id]) > self.config.performance_window:
                self.route_performance[route_id].pop(0)

        # Update weights (simplified gradient update)
        lr = self.config.learning_rate
        for route_id in selected_routes:
            # Increase weight if good performance, decrease if poor
            adjustment = lr * (performance_feedback - 0.5)  # Center around 0.5
            self.route_weights[route_id] += adjustment

            # Keep weights positive
            self.route_weights[route_id] = max(0.01, self.route_weights[route_id])

        # Normalize weights
        total_weight = sum(self.route_weights)
        if total_weight > 0:
            self.route_weights = [w / total_weight for w in self.route_weights]

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing policy statistics."""
        return {
            'policy_id': self.config.policy_id,
            'num_routes': self.config.num_routes,
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
            'exploration_rate': self.config.exploration_rate,
            'total_decisions': len(self.recent_decisions),
            'route_usage': dict(self.route_usage_count),
            'avg_route_performance': {
                route_id: np.mean(perfs) if perfs else 0
                for route_id, perfs in self.route_performance.items()
            },
        }


class MetaRoutingPolicy:
    """
    Level 2: Meta-routing policy that modifies the routing policy.

    This learns how to optimize the routing policy itself by analyzing
    trends and making structural changes.
    """

    def __init__(self, config: MetaRoutingPolicyConfig):
        self.config = config

        # Track meta-level performance
        self.meta_performance_history = []
        self.modification_history = []

        # Current meta-state
        self.current_strategy = 'balanced'  # balanced, aggressive, conservative
        self.last_meta_update_step = 0

        logger.info("MetaRoutingPolicy initialized")

    def analyze_routing_trends(
        self,
        routing_policy: RoutingPolicy,
        current_step: int
    ) -> Dict[str, Any]:
        """
        Analyze routing performance trends.

        Args:
            routing_policy: The routing policy to analyze
            current_step: Current training step

        Returns:
            Analysis results with trend information
        """
        stats = routing_policy.get_statistics()

        analysis = {
            'step': current_step,
            'timestamp': datetime.now().isoformat(),
            'trends': {},
            'issues': [],
            'recommendations': []
        }

        # Analyze route usage distribution
        usage_counts = list(stats['route_usage'].values())
        if usage_counts:
            usage_variance = np.std(usage_counts) if len(usage_counts) > 1 else 0
            usage_mean = np.mean(usage_counts)

            analysis['trends']['usage_variance'] = usage_variance
            analysis['trends']['usage_mean'] = usage_mean

            # Check for imbalance
            if usage_variance > usage_mean * 0.5:
                analysis['issues'].append('high_usage_imbalance')
                analysis['recommendations'].append('increase_exploration')

        # Analyze performance trends
        all_performances = []
        for route_id, perfs in routing_policy.route_performance.items():
            if perfs:
                all_performances.extend(perfs[-self.config.trend_analysis_window:])

        if len(all_performances) >= 10:
            # Calculate trend
            x = list(range(len(all_performances)))
            y = all_performances

            try:
                trend_coef = np.polyfit(x, y, 1)[0]
                analysis['trends']['performance_trend'] = trend_coef

                if trend_coef < -0.001:
                    analysis['issues'].append('declining_performance')
                    analysis['recommendations'].append('increase_temperature')
                elif trend_coef > 0.001:
                    analysis['issues'].append('improving_performance')
                    analysis['recommendations'].append('continue_current_strategy')
            except:
                pass

            # Overall performance level
            avg_performance = np.mean(all_performances)
            analysis['trends']['avg_performance'] = avg_performance

            if avg_performance < self.config.poor_performance_threshold:
                analysis['issues'].append('poor_overall_performance')
                analysis['recommendations'].append('major_restructure')
            elif avg_performance > self.config.excellent_performance_threshold:
                analysis['issues'].append('excellent_performance')
                analysis['recommendations'].append('reduce_exploration')

        return analysis

    def suggest_modifications(
        self,
        analysis: Dict[str, Any],
        routing_policy: RoutingPolicy
    ) -> Dict[str, Any]:
        """
        Suggest modifications to the routing policy based on analysis.

        Args:
            analysis: Analysis results from analyze_routing_trends
            routing_policy: Current routing policy

        Returns:
            Dictionary of suggested modifications
        """
        modifications = {
            'temperature': None,
            'top_k': None,
            'exploration_rate': None,
            'structural_changes': [],
        }

        recommendations = analysis.get('recommendations', [])

        # Temperature adaptation
        if self.config.enable_temperature_adaptation:
            if 'increase_temperature' in recommendations:
                new_temp = min(
                    routing_policy.config.temperature * 1.2,
                    self.config.temperature_range[1]
                )
                modifications['temperature'] = new_temp
            elif 'decrease_temperature' in recommendations:
                new_temp = max(
                    routing_policy.config.temperature * 0.8,
                    self.config.temperature_range[0]
                )
                modifications['temperature'] = new_temp

        # Top-k adaptation
        if self.config.enable_top_k_adaptation:
            if 'increase_diversity' in recommendations:
                new_k = min(
                    routing_policy.config.top_k + 1,
                    self.config.top_k_range[1]
                )
                modifications['top_k'] = new_k
            elif 'decrease_diversity' in recommendations:
                new_k = max(
                    routing_policy.config.top_k - 1,
                    self.config.top_k_range[0]
                )
                modifications['top_k'] = new_k

        # Exploration rate adaptation
        if self.config.enable_exploration_adaptation:
            if 'increase_exploration' in recommendations:
                new_rate = min(
                    routing_policy.config.exploration_rate + 0.05,
                    self.config.exploration_range[1]
                )
                modifications['exploration_rate'] = new_rate
            elif 'reduce_exploration' in recommendations:
                new_rate = max(
                    routing_policy.config.exploration_rate - 0.05,
                    self.config.exploration_range[0]
                )
                modifications['exploration_rate'] = new_rate

        # Structural changes
        if self.config.enable_structural_changes:
            if 'major_restructure' in recommendations:
                modifications['structural_changes'].append('reset_poor_routes')

            if 'high_usage_imbalance' in analysis.get('issues', []):
                modifications['structural_changes'].append('rebalance_weights')

        return modifications

    def apply_modifications(
        self,
        modifications: Dict[str, Any],
        routing_policy: RoutingPolicy,
        step: int
    ):
        """
        Apply suggested modifications to the routing policy.

        Args:
            modifications: Modifications to apply
            routing_policy: Routing policy to modify
            step: Current step
        """
        applied = []

        # Apply temperature change
        if modifications['temperature'] is not None:
            old_temp = routing_policy.config.temperature
            routing_policy.config.temperature = modifications['temperature']
            applied.append(f"temperature: {old_temp:.3f} → {modifications['temperature']:.3f}")

        # Apply top-k change
        if modifications['top_k'] is not None:
            old_k = routing_policy.config.top_k
            routing_policy.config.top_k = modifications['top_k']
            applied.append(f"top_k: {old_k} → {modifications['top_k']}")

        # Apply exploration rate change
        if modifications['exploration_rate'] is not None:
            old_rate = routing_policy.config.exploration_rate
            routing_policy.config.exploration_rate = modifications['exploration_rate']
            applied.append(f"exploration: {old_rate:.3f} → {modifications['exploration_rate']:.3f}")

        # Apply structural changes
        for change in modifications['structural_changes']:
            if change == 'reset_poor_routes':
                # Reset routes with poor performance
                for route_id, perfs in routing_policy.route_performance.items():
                    if perfs and np.mean(perfs) < 0.3:
                        routing_policy.route_weights[route_id] = 1.0 / routing_policy.config.num_routes
                        routing_policy.route_biases[route_id] = 0.0
                applied.append("reset_poor_routes")

            elif change == 'rebalance_weights':
                # Rebalance route weights
                total = sum(routing_policy.route_weights)
                if total > 0:
                    routing_policy.route_weights = [
                        w / total for w in routing_policy.route_weights
                    ]
                applied.append("rebalance_weights")

        # Record modification
        if applied:
            self.modification_history.append({
                'step': step,
                'modifications': applied,
                'timestamp': datetime.now().isoformat()
            })

            logger.info(f"Step {step}: Applied meta-modifications: {', '.join(applied)}")


@dataclass
class SelfModifyingRouterConfig:
    """Configuration for self-modifying router."""

    # Routing policy config
    routing_policy_config: RoutingPolicyConfig = field(
        default_factory=lambda: RoutingPolicyConfig(policy_id="default_routing_policy")
    )

    # Meta-routing policy config
    meta_routing_policy_config: MetaRoutingPolicyConfig = field(
        default_factory=MetaRoutingPolicyConfig
    )

    # Global settings
    enable_self_modification: bool = True
    track_detailed_stats: bool = True


class SelfModifyingRouter:
    """
    Self-Modifying Router implementing Nested Learning.

    This router learns at two levels:
    - Level 1: Routing decisions (fast, every step)
    - Level 2: Routing policy optimization (slow, every 100 steps)

    The meta-routing policy learns to modify the routing policy's structure
    and parameters based on observed performance trends.
    """

    def __init__(self, config: Optional[SelfModifyingRouterConfig] = None):
        self.config = config or SelfModifyingRouterConfig()

        # Level 1: Routing policy
        self.routing_policy = RoutingPolicy(self.config.routing_policy_config)

        # Level 2: Meta-routing policy
        self.meta_routing_policy = MetaRoutingPolicy(
            self.config.meta_routing_policy_config
        )

        # State
        self.current_step = 0
        self.total_routing_decisions = 0
        self.meta_updates_performed = 0

        logger.info("🔀 Self-Modifying Router initialized")
        logger.info(f"   Level 1 (Routing): Updates every {self.config.routing_policy_config.update_freq} steps")
        logger.info(f"   Level 2 (Meta-Routing): Updates every {self.config.meta_routing_policy_config.meta_update_freq} steps")

    def route(
        self,
        inputs: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Perform routing decision.

        Args:
            inputs: Input data to route
            context: Optional routing context

        Returns:
            Tuple of (selected_routes, routing_info)
        """
        self.total_routing_decisions += 1

        # Level 1: Make routing decision
        selected_routes, metadata = self.routing_policy.select_routes(inputs, context)

        # Add meta-information
        metadata['step'] = self.current_step
        metadata['meta_updates_count'] = self.meta_updates_performed

        return selected_routes, metadata

    def update_with_feedback(
        self,
        selected_routes: List[int],
        performance_feedback: float,
        step: int
    ):
        """
        Update router with performance feedback.

        Args:
            selected_routes: Routes that were selected
            performance_feedback: Performance score [0, 1]
            step: Current training step
        """
        self.current_step = step

        # Level 1: Update routing policy
        self.routing_policy.update(selected_routes, performance_feedback, step)

        # Level 2: Meta-update if it's time
        if (self.config.enable_self_modification and
            step % self.config.meta_routing_policy_config.meta_update_freq == 0):
            self._perform_meta_update(step)

    def _perform_meta_update(self, step: int):
        """Perform meta-level update (Level 2)."""
        # Analyze routing trends
        analysis = self.meta_routing_policy.analyze_routing_trends(
            self.routing_policy,
            step
        )

        # Get modification suggestions
        modifications = self.meta_routing_policy.suggest_modifications(
            analysis,
            self.routing_policy
        )

        # Apply modifications
        self.meta_routing_policy.apply_modifications(
            modifications,
            self.routing_policy,
            step
        )

        self.meta_updates_performed += 1

        logger.debug(f"Meta-update {self.meta_updates_performed} completed at step {step}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive router statistics."""
        routing_stats = self.routing_policy.get_statistics()

        return {
            'current_step': self.current_step,
            'total_routing_decisions': self.total_routing_decisions,
            'meta_updates_performed': self.meta_updates_performed,
            'level1_routing_policy': routing_stats,
            'level2_meta_policy': {
                'modification_count': len(self.meta_routing_policy.modification_history),
                'recent_modifications': self.meta_routing_policy.modification_history[-5:],
                'current_strategy': self.meta_routing_policy.current_strategy,
            },
            'self_modification_enabled': self.config.enable_self_modification,
        }

    def get_modification_history(self) -> List[Dict[str, Any]]:
        """Get history of meta-level modifications."""
        return self.meta_routing_policy.modification_history.copy()


# Factory function
def create_self_modifying_router(
    config: Optional[SelfModifyingRouterConfig] = None
) -> SelfModifyingRouter:
    """Create a self-modifying router instance."""
    return SelfModifyingRouter(config)


# Global instance
_global_self_modifying_router: Optional[SelfModifyingRouter] = None


def get_global_self_modifying_router() -> SelfModifyingRouter:
    """Get the global self-modifying router instance."""
    global _global_self_modifying_router
    if _global_self_modifying_router is None:
        _global_self_modifying_router = create_self_modifying_router()
    return _global_self_modifying_router


def main():
    """Test the self-modifying router."""
    logging.basicConfig(level=logging.INFO)
    logger.info("🔀 Self-Modifying Router - Testing Mode")

    # Create router
    router = create_self_modifying_router()

    # Simulate routing with varying performance
    for step in range(500):
        # Simulate input
        inputs = f"input_{step}"

        # Route
        selected_routes, metadata = router.route(inputs)

        # Simulate performance feedback
        # Performance improves over time but with noise
        base_performance = 0.3 + step * 0.001
        noise = (hash(str(step)) % 20 - 10) / 100
        performance = np.clip(base_performance + noise, 0, 1)

        # Update with feedback
        router.update_with_feedback(selected_routes, performance, step)

        # Log progress
        if step % 100 == 0:
            stats = router.get_statistics()
            logger.info(f"\nStep {step}:")
            logger.info(f"  Routing decisions: {stats['total_routing_decisions']}")
            logger.info(f"  Meta-updates: {stats['meta_updates_performed']}")
            logger.info(f"  Temperature: {stats['level1_routing_policy']['temperature']:.3f}")
            logger.info(f"  Top-k: {stats['level1_routing_policy']['top_k']}")
            logger.info(f"  Exploration: {stats['level1_routing_policy']['exploration_rate']:.3f}")

    # Final statistics
    final_stats = router.get_statistics()
    logger.info(f"\n📊 Final Statistics:")
    logger.info(f"  Total routing decisions: {final_stats['total_routing_decisions']}")
    logger.info(f"  Total meta-updates: {final_stats['meta_updates_performed']}")

    # Modification history
    modifications = router.get_modification_history()
    logger.info(f"\n🔧 Modification History ({len(modifications)} modifications):")
    for mod in modifications[-5:]:
        logger.info(f"  Step {mod['step']}: {', '.join(mod['modifications'])}")


if __name__ == "__main__":
    main()
