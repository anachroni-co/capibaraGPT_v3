"""
Performance Adapter

Automatically adapts kernel selection and configurations based on
real-time performance metrics, optimizing latency, throughput,
and memory usage.
"""

import logging
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics

from .base_adapter import BaseAdapter, AdapterConfig
from .adapter_registry import register_adapter_decorator, AdapterType

logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """Monitored performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    TPU_UTILIZATION = "tpu_utilization"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"

class OptimizationGoal(Enum):
    """Optimization goals."""
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_MEMORY = "minimize_memory"
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"

@dataclass
class PerformanceThreshold:
    """Performance thresholds to trigger adaptations."""
    metric: PerformanceMetric
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    tolerance: float = 0.1  # Tolerance as percentage

@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot at a given moment."""
    timestamp: float
    metrics: Dict[PerformanceMetric, float]
    operation_type: str = ""
    batch_size: int = 1
    sequence_length: int = 0
    model_size: str = "unknown"

@dataclass
class AdaptationAction:
    """Performance adaptation action."""
    action_type: str
    target_component: str
    old_value: Any
    new_value: Any
    expected_improvement: Dict[PerformanceMetric, float]
    timestamp: float = field(default_factory=time.time)
    success: Optional[bool] = None
    actual_improvement: Dict[PerformanceMetric, float] = field(default_factory=dict)

class PerformanceMonitor:
    """Real-time performance metrics monitor."""
    
    def __init__(self, history_size: int = 1000, sampling_interval: float = 0.1):
        self.history_size = history_size
        self.sampling_interval = sampling_interval
        self.metrics_history: Dict[PerformanceMetric, deque] = {
            metric: deque(maxlen=history_size) for metric in PerformanceMetric
        }
        self.operation_metrics: Dict[str, Dict[PerformanceMetric, deque]] = defaultdict(
            lambda: {metric: deque(maxlen=100) for metric in PerformanceMetric}
        )
        self.monitoring_active = False
        self.monitoring_thread = None
        self.lock = threading.RLock()
        
    def start_monitoring(self):
        """Starts continuous metrics monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stops metrics monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")

    def record_metric(self, metric: PerformanceMetric, value: float, operation_type: str = ""):
        """Records a performance metric."""
        with self.lock:
            timestamp = time.time()
            self.metrics_history[metric].append((timestamp, value))

            if operation_type:
                self.operation_metrics[operation_type][metric].append((timestamp, value))

    def get_current_metrics(self) -> Dict[PerformanceMetric, float]:
        """Gets current metrics."""
        current_metrics = {}

        with self.lock:
            for metric in PerformanceMetric:
                history = self.metrics_history[metric]
                if history:
                    # Average of last 5 measurements
                    recent_values = [value for _, value in list(history)[-5:]]
                    current_metrics[metric] = statistics.mean(recent_values)
                else:
                    current_metrics[metric] = 0.0

        return current_metrics

    def get_metric_trend(self, metric: PerformanceMetric, window_size: int = 10) -> float:
        """Gets metric trend (slope)."""
        with self.lock:
            history = list(self.metrics_history[metric])

            if len(history) < 2:
                return 0.0

            recent_history = history[-window_size:]
            if len(recent_history) < 2:
                return 0.0

            # Calculate slope using simple linear regression
            n = len(recent_history)
            sum_x = sum(i for i in range(n))
            sum_y = sum(value for _, value in recent_history)
            sum_xy = sum(i * value for i, (_, value) in enumerate(recent_history))
            sum_x2 = sum(i * i for i in range(n))

            if n * sum_x2 - sum_x * sum_x == 0:
                return 0.0

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope

    def get_operation_metrics(self, operation_type: str) -> Dict[PerformanceMetric, float]:
        """Gets metrics specific to an operation type."""
        with self.lock:
            if operation_type not in self.operation_metrics:
                return {metric: 0.0 for metric in PerformanceMetric}
            
            op_metrics = {}
            for metric in PerformanceMetric:
                history = self.operation_metrics[operation_type][metric]
                if history:
                    recent_values = [value for _, value in list(history)[-5:]]
                    op_metrics[metric] = statistics.mean(recent_values)
                else:
                    op_metrics[metric] = 0.0
            
            return op_metrics
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.warning(f"Error in monitoring loop: {e}")
                time.sleep(1.0)

    def _collect_system_metrics(self):
        """Collects system metrics."""
        try:
            # Memory metrics
            memory_usage = self._get_memory_usage()
            self.record_metric(PerformanceMetric.MEMORY_USAGE, memory_usage)

            # CPU metrics
            cpu_usage = self._get_cpu_usage()
            self.record_metric(PerformanceMetric.CPU_UTILIZATION, cpu_usage)

            # GPU metrics (if available)
            gpu_usage = self._get_gpu_usage()
            if gpu_usage is not None:
                self.record_metric(PerformanceMetric.GPU_UTILIZATION, gpu_usage)

        except Exception as e:
            logger.debug(f"Error collecting system metrics: {e}")

    def _get_memory_usage(self) -> float:
        """Gets process memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_percent()
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Gets process CPU usage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _get_gpu_usage(self) -> Optional[float]:
        """Gets GPU usage if available."""
        try:
            # Try pynvml for NVIDIA GPUs
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return info.gpu
        except ImportError:
            return None
        except Exception:
            return None

class PerformanceOptimizer:
    """Optimizer that makes adaptation decisions based on metrics."""
    
    def __init__(self, goal: OptimizationGoal = OptimizationGoal.BALANCED):
        self.goal = goal
        self.thresholds: List[PerformanceThreshold] = []
        self.adaptation_history: deque = deque(maxlen=100)
        self.optimization_strategies: Dict[OptimizationGoal, Callable] = {
            OptimizationGoal.MINIMIZE_LATENCY: self._optimize_for_latency,
            OptimizationGoal.MAXIMIZE_THROUGHPUT: self._optimize_for_throughput,
            OptimizationGoal.MINIMIZE_MEMORY: self._optimize_for_memory,
            OptimizationGoal.BALANCED: self._optimize_balanced,
            OptimizationGoal.COST_OPTIMIZED: self._optimize_for_cost
        }

        # Set up default thresholds
        self._setup_default_thresholds()

    def _setup_default_thresholds(self):
        """Sets up default performance thresholds."""
        self.thresholds = [
            PerformanceThreshold(PerformanceMetric.LATENCY, max_value=100.0, tolerance=0.2),
            PerformanceThreshold(PerformanceMetric.MEMORY_USAGE, max_value=80.0, tolerance=0.1),
            PerformanceThreshold(PerformanceMetric.ERROR_RATE, max_value=0.05, tolerance=0.01),
            PerformanceThreshold(PerformanceMetric.CACHE_HIT_RATE, min_value=0.7, tolerance=0.1)
        ]

    def analyze_performance(self, current_metrics: Dict[PerformanceMetric, float]) -> List[AdaptationAction]:
        """Analyzes metrics and suggests adaptation actions."""
        actions = []

        # Check threshold violations
        for threshold in self.thresholds:
            metric_value = current_metrics.get(threshold.metric, 0.0)

            if self._violates_threshold(metric_value, threshold):
                action = self._create_adaptation_action(threshold, metric_value)
                if action:
                    actions.append(action)

        # Apply specific optimization strategy
        strategy_actions = self.optimization_strategies[self.goal](current_metrics)
        actions.extend(strategy_actions)

        return actions

    def _violates_threshold(self, value: float, threshold: PerformanceThreshold) -> bool:
        """Checks if a value violates a threshold."""
        if threshold.min_value is not None:
            if value < threshold.min_value * (1 - threshold.tolerance):
                return True
        
        if threshold.max_value is not None:
            if value > threshold.max_value * (1 + threshold.tolerance):
                return True
        
        if threshold.target_value is not None:
            deviation = abs(value - threshold.target_value) / threshold.target_value
            if deviation > threshold.tolerance:
                return True
        
        return False
    
    def _create_adaptation_action(self, threshold: PerformanceThreshold, current_value: float) -> Optional[AdaptationAction]:
        """Creates an adaptation action for a threshold violation."""
        if threshold.metric == PerformanceMetric.LATENCY:
            return AdaptationAction(
                action_type="reduce_batch_size",
                target_component="kernel_executor",
                old_value="current_batch",
                new_value="reduced_batch",
                expected_improvement={PerformanceMetric.LATENCY: -20.0}
            )
        elif threshold.metric == PerformanceMetric.MEMORY_USAGE:
            return AdaptationAction(
                action_type="enable_gradient_checkpointing",
                target_component="model",
                old_value=False,
                new_value=True,
                expected_improvement={PerformanceMetric.MEMORY_USAGE: -30.0}
            )
        
        return None
    
    def _optimize_for_latency(self, metrics: Dict[PerformanceMetric, float]) -> List[AdaptationAction]:
        """Optimization strategy to minimize latency."""
        actions = []
        
        if metrics.get(PerformanceMetric.LATENCY, 0) > 50.0:
            actions.append(AdaptationAction(
                action_type="switch_to_low_latency_kernel",
                target_component="kernel_backend",
                old_value="default",
                new_value="low_latency",
                expected_improvement={PerformanceMetric.LATENCY: -40.0}
            ))
        
        return actions
    
    def _optimize_for_throughput(self, metrics: Dict[PerformanceMetric, float]) -> List[AdaptationAction]:
        """Optimization strategy to maximize throughput."""
        actions = []
        
        if metrics.get(PerformanceMetric.THROUGHPUT, 0) < 100.0:
            actions.append(AdaptationAction(
                action_type="increase_batch_size",
                target_component="kernel_executor",
                old_value="current_batch",
                new_value="increased_batch",
                expected_improvement={PerformanceMetric.THROUGHPUT: 25.0}
            ))
        
        return actions
    
    def _optimize_for_memory(self, metrics: Dict[PerformanceMetric, float]) -> List[AdaptationAction]:
        """Optimization strategy to minimize memory usage."""
        actions = []
        
        if metrics.get(PerformanceMetric.MEMORY_USAGE, 0) > 70.0:
            actions.append(AdaptationAction(
                action_type="enable_memory_efficient_attention",
                target_component="attention_kernel",
                old_value="standard",
                new_value="memory_efficient",
                expected_improvement={PerformanceMetric.MEMORY_USAGE: -25.0}
            ))
        
        return actions
    
    def _optimize_balanced(self, metrics: Dict[PerformanceMetric, float]) -> List[AdaptationAction]:
        """Balanced optimization strategy."""
        actions = []

        # Combine strategies with weights
        latency_actions = self._optimize_for_latency(metrics)
        throughput_actions = self._optimize_for_throughput(metrics)
        memory_actions = self._optimize_for_memory(metrics)

        # Select balanced actions
        all_actions = latency_actions + throughput_actions + memory_actions

        # Filter conflicting actions and select most beneficial ones
        return self._select_balanced_actions(all_actions)

    def _optimize_for_cost(self, metrics: Dict[PerformanceMetric, float]) -> List[AdaptationAction]:
        """Optimization strategy to minimize costs."""
        actions = []

        # Prefer more economical backends
        actions.append(AdaptationAction(
            action_type="switch_to_cost_effective_backend",
            target_component="kernel_backend",
            old_value="premium",
            new_value="cost_effective",
            expected_improvement={PerformanceMetric.LATENCY: 10.0}  # Small trade-off
        ))

        return actions

    def _select_balanced_actions(self, actions: List[AdaptationAction]) -> List[AdaptationAction]:
        """Selects balanced actions avoiding conflicts."""
        # Simple implementation: select up to 2 non-conflicting actions
        selected = []

        for action in actions[:2]:  # Limit to 2 actions per cycle
            if not self._conflicts_with_selected(action, selected):
                selected.append(action)

        return selected

    def _conflicts_with_selected(self, action: AdaptationAction, selected: List[AdaptationAction]) -> bool:
        """Checks if an action conflicts with already selected ones."""
        for selected_action in selected:
            if action.target_component == selected_action.target_component:
                return True
        return False

@register_adapter_decorator(
    adapter_type=AdapterType.PERFORMANCE_OPTIMIZATION,
    priority=95,
    capabilities=["real_time_monitoring", "automatic_adaptation", "multi_metric_optimization"],
    metadata={"version": "1.0", "supports_continuous_optimization": True}
)
class PerformanceAdapter(BaseAdapter):
    """
    Performance adapter that monitors metrics in real-time
    and automatically adapts system configuration.
    """

    def __init__(self,
                 config: Optional[AdapterConfig] = None,
                 optimization_goal: OptimizationGoal = OptimizationGoal.BALANCED):
        super().__init__(config)
        self.optimization_goal = optimization_goal
        self.monitor = PerformanceMonitor()
        self.optimizer = PerformanceOptimizer(optimization_goal)
        self.adaptation_callbacks: Dict[str, Callable] = {}
        self.auto_adaptation_enabled = True
        self.adaptation_interval = 5.0  # seconds
        self.last_adaptation_time = 0.0

    def _initialize_impl(self) -> bool:
        """Initializes the performance adapter."""
        try:
            # Start monitoring
            self.monitor.start_monitoring()

            # Register default adaptation callbacks
            self._register_default_callbacks()

            self.logger.info(f"Performance adapter initialized with goal: {self.optimization_goal.value}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize performance adapter: {e}")
            return False

    def _execute_impl(self, operation_type: str = "", *args, **kwargs) -> Dict[str, Any]:
        """Executes performance analysis and adaptation."""
        start_time = time.time()

        try:
            # Get current metrics
            current_metrics = self.monitor.get_current_metrics()

            # Record operation metrics if specified
            if operation_type:
                execution_time = (time.time() - start_time) * 1000
                self.monitor.record_metric(PerformanceMetric.LATENCY, execution_time, operation_type)

            # Check if it's time to adapt
            if self._should_adapt():
                adaptation_actions = self.optimizer.analyze_performance(current_metrics)

                if adaptation_actions:
                    self._execute_adaptations(adaptation_actions)
                    self.last_adaptation_time = time.time()

            return {
                'current_metrics': current_metrics,
                'adaptation_active': self.auto_adaptation_enabled,
                'last_adaptation': self.last_adaptation_time,
                'operation_type': operation_type
            }

        except Exception as e:
            self.logger.error(f"Error in performance analysis: {e}")
            return {'error': str(e)}

    def _should_adapt(self) -> bool:
        """Determines if it's time to execute adaptations."""
        if not self.auto_adaptation_enabled:
            return False

        current_time = time.time()
        return (current_time - self.last_adaptation_time) >= self.adaptation_interval

    def _execute_adaptations(self, actions: List[AdaptationAction]):
        """Executes adaptation actions."""
        for action in actions:
            try:
                callback = self.adaptation_callbacks.get(action.action_type)
                if callback:
                    success = callback(action)
                    action.success = success
                    
                    if success:
                        self.logger.info(f"Successfully executed adaptation: {action.action_type}")
                    else:
                        self.logger.warning(f"Failed to execute adaptation: {action.action_type}")
                else:
                    self.logger.warning(f"No callback registered for adaptation: {action.action_type}")
                    action.success = False
                
                # Registrar acción en historial
                self.optimizer.adaptation_history.append(action)
                
            except Exception as e:
                self.logger.error(f"Error executing adaptation {action.action_type}: {e}")
                action.success = False
    
    def _register_default_callbacks(self):
        """Registers default adaptation callbacks."""
        self.adaptation_callbacks.update({
            "reduce_batch_size": self._adapt_batch_size,
            "increase_batch_size": self._adapt_batch_size,
            "switch_to_low_latency_kernel": self._adapt_kernel_backend,
            "switch_to_cost_effective_backend": self._adapt_kernel_backend,
            "enable_gradient_checkpointing": self._adapt_memory_optimization,
            "enable_memory_efficient_attention": self._adapt_attention_mechanism
        })

    def _adapt_batch_size(self, action: AdaptationAction) -> bool:
        """Adapts batch size."""
        # Simulated implementation - in a real case would interact with the system
        self.logger.info(f"Adapting batch size: {action.action_type}")
        return True

    def _adapt_kernel_backend(self, action: AdaptationAction) -> bool:
        """Adapts kernel backend."""
        self.logger.info(f"Adapting kernel backend: {action.new_value}")
        return True

    def _adapt_memory_optimization(self, action: AdaptationAction) -> bool:
        """Adapts memory optimizations."""
        self.logger.info(f"Adapting memory optimization: {action.action_type}")
        return True

    def _adapt_attention_mechanism(self, action: AdaptationAction) -> bool:
        """Adapts attention mechanism."""
        self.logger.info(f"Adapting attention mechanism: {action.new_value}")
        return True

    def register_adaptation_callback(self, action_type: str, callback: Callable[[AdaptationAction], bool]):
        """Registers a custom callback for an adaptation type."""
        self.adaptation_callbacks[action_type] = callback
        self.logger.info(f"Registered callback for adaptation type: {action_type}")

    def set_optimization_goal(self, goal: OptimizationGoal):
        """Changes the optimization goal."""
        self.optimization_goal = goal
        self.optimizer.goal = goal
        self.logger.info(f"Changed optimization goal to: {goal.value}")

    def enable_auto_adaptation(self):
        """Enables automatic adaptation."""
        self.auto_adaptation_enabled = True
        self.logger.info("Auto-adaptation enabled")

    def disable_auto_adaptation(self):
        """Disables automatic adaptation."""
        self.auto_adaptation_enabled = False
        self.logger.info("Auto-adaptation disabled")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generates a complete performance report."""
        current_metrics = self.monitor.get_current_metrics()

        # Calculate trends
        trends = {}
        for metric in PerformanceMetric:
            trends[metric.value] = self.monitor.get_metric_trend(metric)

        # Adaptation statistics
        adaptation_stats = {
            'total_adaptations': len(self.optimizer.adaptation_history),
            'successful_adaptations': sum(1 for a in self.optimizer.adaptation_history if a.success),
            'recent_adaptations': [
                {
                    'action_type': a.action_type,
                    'timestamp': a.timestamp,
                    'success': a.success
                } for a in list(self.optimizer.adaptation_history)[-5:]
            ]
        }

        return {
            'current_metrics': {k.value: v for k, v in current_metrics.items()},
            'metric_trends': trends,
            'optimization_goal': self.optimization_goal.value,
            'auto_adaptation_enabled': self.auto_adaptation_enabled,
            'adaptation_stats': adaptation_stats,
            'thresholds': [
                {
                    'metric': t.metric.value,
                    'min_value': t.min_value,
                    'max_value': t.max_value,
                    'target_value': t.target_value
                } for t in self.optimizer.thresholds
            ]
        }

    def cleanup(self):
        """Cleans up adapter resources."""
        self.monitor.stop_monitoring()
        self.logger.info("Performance adapter cleanup completed")


# Global adapter instance
performance_adapter = PerformanceAdapter()

# Utility functions
def monitor_operation_performance(operation_type: str):
    """Decorator to monitor operation performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                performance_adapter.monitor.record_metric(
                    PerformanceMetric.LATENCY, execution_time, operation_type
                )
                return result
            except Exception as e:
                performance_adapter.monitor.record_metric(
                    PerformanceMetric.ERROR_RATE, 1.0, operation_type
                )
                raise
        return wrapper
    return decorator

def get_performance_summary():
    """Gets system performance summary."""
    if performance_adapter.get_status().value == "ready":
        return performance_adapter.get_performance_report()
    else:
        return {"status": "performance_adapter_not_ready"}