"""
Nested Meta-Loop System for CapibaraGPT

This module implements Nested Learning as proposed by Ali Behrouz (Google Research).
The core idea is to treat a single ML model not as one continuous process, but as a
system of interconnected, multi-level learning problems that are optimized simultaneously
at different time scales.

Key Features:
- Multi-level nested optimization (3 levels)
- Different update frequencies per level to avoid catastrophic forgetting
- Hierarchical hyperparameter optimization
- Cross-level feedback and consolidation
- Memory-efficient meta-learning

Levels:
- Level 0: Model parameters (updated every step - handled by training loop)
- Level 1: Hyperparameters (updated every 50 steps)
- Level 2: Meta-hyperparameters (updated every 200 steps)

Reference:
    Behrouz, A.; Razaviyayn, M.; Mirrokni, V.; Zhong, P. (2025).
    "Nested Learning: The Illusion of Deep Learning Architectures"
    NeurIPS 2025
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from capibara.core.meta_loop import (
    MetaLoop,
    MetaLoopConfig,
    MetaLoopState,
    MetaOptimizer,
    ArchitectureAdaptor
)

logger = logging.getLogger(__name__)


@dataclass
class NestedMetaLoopConfig:
    """Configuration for the nested meta-loop system."""

    # Level 1: Hyperparameter optimization (from original MetaLoop)
    level1_config: MetaLoopConfig = field(default_factory=MetaLoopConfig)

    # Level 2: Meta-hyperparameter optimization
    meta_meta_learning_rate: float = 1e-5  # Slower learning at meta-level
    meta_evaluation_interval: int = 200  # Meta-level evaluation frequency
    meta_update_interval: int = 500  # Meta-level update frequency

    # Meta-hyperparameter search space (optimizes the meta-loop itself)
    meta_search_space: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'meta_learning_rate': (1e-6, 1e-3),  # Meta-loop's learning rate
        'evaluation_frequency': (25, 100),  # How often to evaluate
        'meta_update_frequency': (100, 500),  # How often to meta-update
        'adaptation_window': (50, 200),  # Window size for adaptation
    })

    # Nested Learning specific parameters
    enable_cross_level_feedback: bool = True  # Allow higher levels to guide lower
    enable_memory_consolidation: bool = True  # Consolidate learning across levels
    catastrophic_forgetting_threshold: float = 0.85  # Retain 85% of old performance

    # Update frequencies (key to Nested Learning)
    level0_update_freq: int = 1  # Model params (every step)
    level1_update_freq: int = 50  # Hyperparams
    level2_update_freq: int = 200  # Meta-hyperparams


class NestedMetaLoopState:
    """Extended state tracking for nested meta-loop."""

    def __init__(self):
        # Level 1 state (hyperparameter level)
        self.level1_state = MetaLoopState()

        # Level 2 state (meta-hyperparameter level)
        self.level2_performance_history = []
        self.level2_meta_config_history = []
        self.level2_best_config = None
        self.level2_best_performance = float('-inf')

        # Cross-level tracking
        self.cross_level_feedback = []
        self.consolidation_history = []

        # Catastrophic forgetting detection
        self.task_performance_baselines = {}  # Track performance per task
        self.forgetting_events = []

        # Global step counter
        self.global_step = 0

    def update_level2_performance(self, meta_performance: float, meta_config: Dict[str, Any]):
        """Update Level 2 (meta-hyperparameter) performance history."""
        self.level2_performance_history.append({
            'step': self.global_step,
            'meta_performance': meta_performance,
            'timestamp': datetime.now().isoformat()
        })

        self.level2_meta_config_history.append({
            'step': self.global_step,
            'meta_config': meta_config.copy(),
            'meta_performance': meta_performance
        })

        if meta_performance > self.level2_best_performance:
            self.level2_best_performance = meta_performance
            self.level2_best_config = meta_config.copy()
            logger.info(f"🎯 Level 2: New best meta-configuration: {meta_performance:.4f}")

    def detect_catastrophic_forgetting(
        self,
        task_id: str,
        current_performance: float,
        threshold: float = 0.85
    ) -> bool:
        """
        Detects if catastrophic forgetting is occurring.

        Args:
            task_id: Identifier for the task
            current_performance: Current performance on the task
            threshold: Minimum acceptable retention (default 85%)

        Returns:
            True if catastrophic forgetting detected
        """
        if task_id not in self.task_performance_baselines:
            # First time seeing this task, establish baseline
            self.task_performance_baselines[task_id] = current_performance
            return False

        baseline = self.task_performance_baselines[task_id]
        retention = current_performance / baseline if baseline > 0 else 1.0

        if retention < threshold:
            # Catastrophic forgetting detected!
            forgetting_event = {
                'step': self.global_step,
                'task_id': task_id,
                'baseline': baseline,
                'current': current_performance,
                'retention': retention,
                'timestamp': datetime.now().isoformat()
            }
            self.forgetting_events.append(forgetting_event)
            logger.warning(
                f"⚠️ Catastrophic forgetting detected for task '{task_id}': "
                f"{retention:.1%} retention (threshold: {threshold:.1%})"
            )
            return True

        return False

    def get_level2_recent_performance(self, window: int = 5) -> List[float]:
        """Get recent meta-level performance."""
        return [p['meta_performance'] for p in self.level2_performance_history[-window:]]


class MetaMetaOptimizer:
    """
    Level 2 optimizer that optimizes the meta-loop's own hyperparameters.
    This is the key innovation from Nested Learning.
    """

    def __init__(self, config: NestedMetaLoopConfig):
        self.config = config
        self.search_space = config.meta_search_space
        self.current_meta_params = self._initialize_meta_params()
        self.meta_gradient_estimates = {}

    def _initialize_meta_params(self) -> Dict[str, float]:
        """Initialize meta-hyperparameters at the center of search space."""
        meta_params = {}
        for param_name, (min_val, max_val) in self.search_space.items():
            # Start at geometric mean for log-scale parameters
            if 'rate' in param_name or 'lr' in param_name:
                meta_params[param_name] = np.sqrt(min_val * max_val)
            else:
                meta_params[param_name] = (min_val + max_val) / 2
        return meta_params

    def optimize_meta_loop_config(
        self,
        meta_performance_history: List[Dict]
    ) -> Dict[str, float]:
        """
        Optimize the meta-loop's own configuration based on its performance.

        Args:
            meta_performance_history: History of meta-loop performance

        Returns:
            Updated meta-hyperparameters
        """
        if len(meta_performance_history) < 2:
            return self.current_meta_params.copy()

        # Estimate meta-gradients
        meta_gradients = self._estimate_meta_gradients(meta_performance_history)

        # Update meta-hyperparameters
        new_meta_params = {}
        for param_name in self.current_meta_params:
            if param_name in meta_gradients:
                gradient = meta_gradients[param_name]
                update = self.config.meta_meta_learning_rate * gradient

                # Apply update with bounds
                new_value = self.current_meta_params[param_name] + update
                min_val, max_val = self.search_space[param_name]
                new_value = np.clip(new_value, min_val, max_val)
                new_meta_params[param_name] = new_value

                logger.debug(
                    f"Level 2 update: {param_name} = {new_value:.6f} "
                    f"(gradient: {gradient:.6f})"
                )
            else:
                new_meta_params[param_name] = self.current_meta_params[param_name]

        self.current_meta_params = new_meta_params
        return new_meta_params.copy()

    def _estimate_meta_gradients(self, history: List[Dict]) -> Dict[str, float]:
        """Estimate performance gradients w.r.t. meta-hyperparameters."""
        gradients = {}

        if len(history) < 2:
            return gradients

        # Use recent history for gradient estimation
        recent_history = history[-10:]

        for param_name in self.search_space:
            param_values = [
                h['meta_config'].get(param_name, self.current_meta_params[param_name])
                for h in recent_history
            ]
            performances = [h['meta_performance'] for h in recent_history]

            if len(set(param_values)) > 1:  # Only if there's variation
                # Use correlation as gradient approximation
                correlation = np.corrcoef(param_values, performances)[0, 1]
                if not np.isnan(correlation):
                    gradients[param_name] = correlation

        return gradients


class NestedMetaLoop:
    """
    Nested Meta-Loop System implementing multi-level optimization.

    This implements the core concept of Nested Learning where optimization
    happens at multiple time scales:
    - Fast updates: Model parameters (handled externally)
    - Medium updates: Hyperparameters (Level 1)
    - Slow updates: Meta-hyperparameters (Level 2)

    This structure helps avoid catastrophic forgetting by separating concerns
    across different time scales.
    """

    def __init__(self, config: Optional[NestedMetaLoopConfig] = None):
        self.config = config or NestedMetaLoopConfig()
        self.state = NestedMetaLoopState()

        # Level 1: Standard meta-loop (hyperparameter optimization)
        self.level1_meta_loop = MetaLoop(self.config.level1_config)

        # Level 2: Meta-meta optimizer (meta-hyperparameter optimization)
        self.level2_optimizer = MetaMetaOptimizer(self.config)

        # Cross-level coordination
        self.cross_level_coordinator = CrossLevelCoordinator(self.config)

        logger.info("🔄 Nested Meta-Loop System initialized")
        logger.info(f"   📊 Level 0 (Model params): Every {self.config.level0_update_freq} steps")
        logger.info(f"   📊 Level 1 (Hyperparams): Every {self.config.level1_update_freq} steps")
        logger.info(f"   📊 Level 2 (Meta-hyperparams): Every {self.config.level2_update_freq} steps")
        logger.info(f"   🛡️ Catastrophic forgetting threshold: {self.config.catastrophic_forgetting_threshold:.1%}")

    def step(
        self,
        current_performance: float,
        current_hyperparams: Dict[str, Any],
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a nested meta-loop step.

        Args:
            current_performance: Current model performance
            current_hyperparams: Current hyperparameters
            task_id: Optional task identifier for forgetting detection

        Returns:
            Dictionary with suggestions at different levels
        """
        self.state.global_step += 1
        step = self.state.global_step

        suggestions = {
            'level0': {},  # Model parameter suggestions (if any)
            'level1': {},  # Hyperparameter suggestions
            'level2': {},  # Meta-hyperparameter suggestions
        }

        # Check for catastrophic forgetting
        if task_id and self.config.enable_memory_consolidation:
            forgetting_detected = self.state.detect_catastrophic_forgetting(
                task_id,
                current_performance,
                self.config.catastrophic_forgetting_threshold
            )
            if forgetting_detected:
                suggestions['catastrophic_forgetting_alert'] = True
                suggestions['recommended_action'] = 'reduce_learning_rate'

        # Level 1: Hyperparameter optimization (medium frequency)
        if step % self.config.level1_update_freq == 0:
            level1_suggestions = self.level1_meta_loop.step(
                current_performance,
                current_hyperparams
            )
            suggestions['level1'] = level1_suggestions

            logger.debug(f"Step {step}: Level 1 update executed")

        # Level 2: Meta-hyperparameter optimization (slow frequency)
        if step % self.config.level2_update_freq == 0:
            level2_suggestions = self._level2_update()
            suggestions['level2'] = level2_suggestions

            logger.debug(f"Step {step}: Level 2 update executed")

        # Cross-level feedback
        if self.config.enable_cross_level_feedback and step % 100 == 0:
            cross_level_suggestions = self.cross_level_coordinator.coordinate(
                level1_state=self.level1_meta_loop.state,
                level2_state=self.state,
                current_step=step
            )
            if cross_level_suggestions:
                suggestions['cross_level'] = cross_level_suggestions

        return suggestions

    def _level2_update(self) -> Dict[str, Any]:
        """
        Perform Level 2 meta-update (optimize the meta-loop itself).

        Returns:
            Suggestions for meta-loop configuration changes
        """
        suggestions = {}

        # Calculate meta-performance (how well is the meta-loop doing?)
        meta_performance = self._calculate_meta_performance()

        # Update Level 2 state
        current_meta_config = {
            'meta_learning_rate': self.config.level1_config.meta_learning_rate,
            'evaluation_frequency': self.config.level1_config.evaluation_frequency,
            'meta_update_frequency': self.config.level1_config.meta_update_frequency,
            'adaptation_window': self.config.level1_config.adaptation_window,
        }

        self.state.update_level2_performance(meta_performance, current_meta_config)

        # Optimize meta-hyperparameters
        if len(self.state.level2_performance_history) >= 2:
            optimized_meta_config = self.level2_optimizer.optimize_meta_loop_config(
                self.state.level2_meta_config_history
            )

            # Apply optimized configuration to Level 1 meta-loop
            if 'meta_learning_rate' in optimized_meta_config:
                self.config.level1_config.meta_learning_rate = optimized_meta_config['meta_learning_rate']
                self.level1_meta_loop.config.meta_learning_rate = optimized_meta_config['meta_learning_rate']

            if 'evaluation_frequency' in optimized_meta_config:
                new_freq = int(optimized_meta_config['evaluation_frequency'])
                self.config.level1_config.evaluation_frequency = new_freq
                self.level1_meta_loop.config.evaluation_frequency = new_freq

            if 'meta_update_frequency' in optimized_meta_config:
                new_freq = int(optimized_meta_config['meta_update_frequency'])
                self.config.level1_config.meta_update_frequency = new_freq
                self.level1_meta_loop.config.meta_update_frequency = new_freq

            suggestions['optimized_meta_config'] = optimized_meta_config

            logger.info(
                f"🔧 Level 2: Meta-loop configuration optimized "
                f"(meta_performance: {meta_performance:.4f})"
            )

        return suggestions

    def _calculate_meta_performance(self) -> float:
        """
        Calculate how well the meta-loop is performing.

        Meta-performance is measured by:
        1. Rate of improvement in base performance
        2. Stability of performance (low variance)
        3. Absence of catastrophic forgetting events

        Returns:
            Meta-performance score [0, 1]
        """
        if len(self.state.level1_state.performance_history) < 10:
            return 0.5  # Neutral score with insufficient data

        recent_performances = [
            p['performance']
            for p in self.state.level1_state.performance_history[-20:]
        ]

        # Component 1: Improvement rate (30% weight)
        if len(recent_performances) >= 10:
            first_half = np.mean(recent_performances[:10])
            second_half = np.mean(recent_performances[10:])
            improvement_rate = (second_half - first_half) / (first_half + 1e-8)
            improvement_score = np.clip(improvement_rate * 10, 0, 1)  # Scale to [0, 1]
        else:
            improvement_score = 0.5

        # Component 2: Stability (40% weight)
        variance = np.var(recent_performances)
        stability_score = np.exp(-variance * 10)  # Lower variance = higher score

        # Component 3: No forgetting (30% weight)
        recent_forgetting_events = len([
            e for e in self.state.forgetting_events
            if self.state.global_step - e['step'] < 200
        ])
        forgetting_penalty = min(recent_forgetting_events * 0.2, 1.0)
        forgetting_score = 1.0 - forgetting_penalty

        # Weighted combination
        meta_performance = (
            0.3 * improvement_score +
            0.4 * stability_score +
            0.3 * forgetting_score
        )

        return np.clip(meta_performance, 0, 1)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the nested meta-loop."""
        level1_status = self.level1_meta_loop.get_status()

        return {
            'global_step': self.state.global_step,
            'level1': level1_status,
            'level2': {
                'best_meta_performance': self.state.level2_best_performance,
                'best_meta_config': self.state.level2_best_config,
                'current_meta_params': self.level2_optimizer.current_meta_params,
                'recent_meta_performance': self.state.get_level2_recent_performance(5),
            },
            'catastrophic_forgetting': {
                'total_events': len(self.state.forgetting_events),
                'recent_events': len([
                    e for e in self.state.forgetting_events
                    if self.state.global_step - e['step'] < 200
                ]),
                'tracked_tasks': len(self.state.task_performance_baselines),
            },
            'update_frequencies': {
                'level0': self.config.level0_update_freq,
                'level1': self.config.level1_update_freq,
                'level2': self.config.level2_update_freq,
            }
        }


class CrossLevelCoordinator:
    """Coordinates feedback and consolidation across nested levels."""

    def __init__(self, config: NestedMetaLoopConfig):
        self.config = config
        self.coordination_history = []

    def coordinate(
        self,
        level1_state: MetaLoopState,
        level2_state: NestedMetaLoopState,
        current_step: int
    ) -> Dict[str, Any]:
        """
        Coordinate between levels for better optimization.

        Args:
            level1_state: State from Level 1 (hyperparameters)
            level2_state: State from Level 2 (meta-hyperparameters)
            current_step: Current training step

        Returns:
            Cross-level coordination suggestions
        """
        suggestions = {}

        # Check if Level 1 is struggling
        if len(level1_state.performance_history) >= 20:
            recent_trend = self._analyze_trend(
                [p['performance'] for p in level1_state.performance_history[-20:]]
            )

            if recent_trend < -0.05:  # Declining performance
                suggestions['level1_struggling'] = True
                suggestions['recommendation'] = 'trigger_level2_update_early'
                logger.info("📉 Cross-level: Level 1 struggling, suggesting early Level 2 intervention")

        # Check if we should consolidate learning
        if self.config.enable_memory_consolidation:
            if current_step % 500 == 0:
                consolidation_signal = self._should_consolidate(level1_state, level2_state)
                if consolidation_signal:
                    suggestions['consolidate_memory'] = True
                    logger.info("💾 Cross-level: Triggering memory consolidation")

        return suggestions

    def _analyze_trend(self, values: List[float]) -> float:
        """Analyze trend in values using linear regression."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

    def _should_consolidate(
        self,
        level1_state: MetaLoopState,
        level2_state: NestedMetaLoopState
    ) -> bool:
        """Determine if memory consolidation should occur."""
        # Consolidate if we've achieved stable good performance
        if len(level1_state.performance_history) >= 10:
            recent_perf = [p['performance'] for p in level1_state.performance_history[-10:]]
            if np.mean(recent_perf) > 0.8 and np.var(recent_perf) < 0.01:
                return True
        return False


# Factory functions
def create_nested_meta_loop(config: Optional[NestedMetaLoopConfig] = None) -> NestedMetaLoop:
    """Create a nested meta-loop instance."""
    return NestedMetaLoop(config)


# Global instance
_global_nested_meta_loop: Optional[NestedMetaLoop] = None


def get_global_nested_meta_loop() -> NestedMetaLoop:
    """Get the global nested meta-loop instance."""
    global _global_nested_meta_loop
    if _global_nested_meta_loop is None:
        _global_nested_meta_loop = create_nested_meta_loop()
    return _global_nested_meta_loop


def main():
    """Test the nested meta-loop system."""
    logger.info("🔄 Nested Meta-Loop System - Testing Mode")

    # Create nested meta-loop
    nested_loop = create_nested_meta_loop()

    # Simulate training with multiple tasks
    tasks = ['task_a', 'task_b', 'task_c']

    for step in range(500):
        # Simulate performance (with some tasks and noise)
        task_id = tasks[step % len(tasks)]
        base_performance = 0.6 + 0.3 * np.sin(step * 0.05)
        performance = base_performance + np.random.normal(0, 0.05)
        performance = np.clip(performance, 0, 1)

        # Current hyperparameters
        hyperparams = {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'dropout_rate': 0.1,
            'weight_decay': 0.01
        }

        # Execute nested meta-loop step
        suggestions = nested_loop.step(performance, hyperparams, task_id=task_id)

        # Log significant events
        if suggestions.get('level2'):
            logger.info(f"Step {step}: Level 2 suggestions = {suggestions['level2']}")
        if suggestions.get('catastrophic_forgetting_alert'):
            logger.warning(f"Step {step}: Catastrophic forgetting alert!")

    # Show final status
    status = nested_loop.get_status()
    logger.info(f"\nFinal Status:")
    logger.info(f"  Global step: {status['global_step']}")
    logger.info(f"  Level 1 best performance: {status['level1']['best_performance']:.4f}")
    logger.info(f"  Level 2 best meta-performance: {status['level2']['best_meta_performance']:.4f}")
    logger.info(f"  Catastrophic forgetting events: {status['catastrophic_forgetting']['total_events']}")
    logger.info(f"  Tracked tasks: {status['catastrophic_forgetting']['tracked_tasks']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
