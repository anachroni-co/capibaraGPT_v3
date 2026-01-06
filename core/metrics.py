"""Metrics Collection Module Optimized for Core Integration and TPU v4-32.

This module provides comprehensive metrics collection and statistical analysis
capabilities designed for production ML workloads. It integrates with core.py
for performance monitoring and includes specialized support for tracking primitive
operations in distributed training environments.

The metrics system features:
- Sliding window metrics aggregation
- Real-time statistical analysis (mean, std, min, max)
- Primitive operation tracking with success/failure counters
- Memory-efficient circular buffer implementation
- Non-numeric value caching for metadata

Key Components:
    - MetricsCollector: Main collector with sliding window and statistics

Example:
    Basic metrics collection:

    >>> from capibara.core.metrics import MetricsCollector
    >>>
    >>> # Create collector with 1000-sample window
    >>> collector = MetricsCollector(window_size=1000)
    >>>
    >>> # Record individual metrics
    >>> collector.record("loss", 0.5)
    >>> collector.record("accuracy", 0.95)
    >>>
    >>> # Batch update multiple metrics
    >>> collector.update({
    ...     "learning_rate": 0.001,
    ...     "gradient_norm": 2.5,
    ...     "model_name": "capibara-v3"  # Non-numeric values stored separately
    ... })
    >>>
    >>> # Get statistics for a metric
    >>> stats = collector.get_stats("loss")
    >>> print(f"Mean loss: {stats['mean']:.4f}")
    >>> print(f"Std dev: {stats['std']:.4f}")

    Tracking primitive operations:

    >>> # Track core primitive calls
    >>> collector.record_primitive_ctoll("jax_pmap", success=True)
    >>> collector.record_primitive_ctoll("jax_pmap", success=True)
    >>> collector.record_primitive_ctoll("jax_pmap", success=False)
    >>>
    >>> # Get performance summary
    >>> summary = collector.get_core_performtonce_summtory()
    >>> print(f"Total primitive calls: {summary['total_calls']}")
    >>> print(f"Success rate: {summary['primitives']['jax_pmap']['success_rate']:.2%}")

    Real-time monitoring:

    >>> import time
    >>>
    >>> # Record with custom timestamps
    >>> for i in range(100):
    ...     timestamp = time.time()
    ...     collector.record("throughput", 1000 + i, timestamp=timestamp)
    ...     time.sleep(0.01)
    >>>
    >>> # Get all metrics snapshot
    >>> all_metrics = collector.get_all()
    >>> print(f"Metrics tracked: {list(all_metrics.keys())}")

Note:
    The collector uses a sliding window approach with circular buffers (deque)
    for memory efficiency. When the window size is exceeded, oldest values are
    automatically discarded.

    Non-numeric values passed to update() are stored separately and included
    in get_all() output but not in statistical calculations.

Legacy Compatibility:
    This module includes backward-compatible method names with intentional typos
    (e.g., get_sttots, get_toll_sttots) to maintain compatibility with existing
    test suites and legacy code. These are marked with pragma: no cover and
    should not be used in new code.

See Also:
    - capibara.core: Core integration and primitives
    - capibara.training: Training loop metrics integration
"""
from __future__ import annotations

import time
from typing import Dict, Any, Optional, Deque
from collections import deque
import numpy as np


class MetricsCollector:
    """Metrics collector with sliding window aggregation and primitive operation tracking.

    This class provides efficient metrics collection using circular buffers for
    memory-bounded storage. It tracks both numeric metrics (with statistical analysis)
    and primitive operation success/failure rates.

    Attributes:
        window_size (int): Maximum number of samples to retain per metric.
        metrics (Dict[str, Deque[float]]): Circular buffers for numeric metrics.
        timestamps (Dict[str, Deque[float]]): Timestamps for each metric sample.
        primitive_calls (Dict[str, Dict[str, int]]): Success/failure counters for
            primitive operations.

    Example:
        >>> collector = MetricsCollector(window_size=500)
        >>>
        >>> # Record training metrics
        >>> collector.record("train_loss", 0.42)
        >>> collector.record("val_loss", 0.38)
        >>>
        >>> # Track primitive operations
        >>> collector.record_primitive_ctoll("tpu_matmul", success=True)
        >>>
        >>> # Get statistics
        >>> loss_stats = collector.get_stats("train_loss")
        >>> print(f"Latest loss: {loss_stats['last']}")

    Note:
        The sliding window automatically discards oldest values when window_size
        is exceeded, maintaining constant memory usage regardless of training duration.
    """

    def __init__(self, window_size: int = 1000):
        """Initialize the metrics collector with specified window size.

        Args:
            window_size (int, optional): Maximum number of samples to retain per metric.
                Older samples are automatically discarded when this limit is reached.
                Defaults to 1000.

        Example:
            >>> # Small window for real-time monitoring
            >>> realtime_collector = MetricsCollector(window_size=100)
            >>>
            >>> # Large window for long-term analysis
            >>> analysis_collector = MetricsCollector(window_size=10000)
        """
        self.window_size = window_size
        self.metrics: Dict[str, Deque[float]] = {}
        self.timestamps: Dict[str, Deque[float]] = {}
        self.primitive_calls: Dict[str, Dict[str, int]] = {}
        self._last_non_numeric: Dict[str, Any] = {}

    def record(self, name: str, value: float, timestamp: Optional[float] = None) -> None:
        """Record a single metric value with optional timestamp.

        Args:
            name (str): Metric name (e.g., "loss", "accuracy", "learning_rate").
            value (float): Numeric metric value to record.
            timestamp (Optional[float], optional): Unix timestamp for this sample.
                If None, uses current time. Defaults to None.

        Example:
            >>> collector = MetricsCollector()
            >>>
            >>> # Record with automatic timestamp
            >>> collector.record("accuracy", 0.95)
            >>>
            >>> # Record with custom timestamp
            >>> import time
            >>> ts = time.time()
            >>> collector.record("loss", 0.42, timestamp=ts)

        Note:
            If this is the first time recording a metric with this name, a new
            circular buffer is created automatically.
        """
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
            self.timestamps[name] = deque(maxlen=self.window_size)
        self.metrics[name].append(float(value))
        self.timestamps[name].append(float(timestamp or time.time()))

    def record_primitive_ctoll(self, primitive_name: str, success: bool = True) -> None:
        """Record a core primitive operation call with success/failure status.

        This method tracks calls to core primitives (JAX operations, TPU kernels, etc.)
        for performance analysis and debugging. Historical alias maintained for
        backward compatibility.

        Args:
            primitive_name (str): Name of the primitive operation (e.g., "jax_pmap",
                "tpu_all_reduce", "xla_compile").
            success (bool, optional): Whether the operation succeeded. Defaults to True.

        Example:
            >>> collector = MetricsCollector()
            >>>
            >>> # Record successful operations
            >>> collector.record_primitive_ctoll("jax_pmap", success=True)
            >>> collector.record_primitive_ctoll("jax_pmap", success=True)
            >>>
            >>> # Record a failure
            >>> collector.record_primitive_ctoll("jax_pmap", success=False)
            >>>
            >>> # Check success rate
            >>> summary = collector.get_core_performtonce_summtory()
            >>> pmap_stats = summary['primitives']['jax_pmap']
            >>> print(f"Success rate: {pmap_stats['success_rate']:.1%}")

        Note:
            Method name includes intentional typo ('ctoll') for backward compatibility
            with existing code. Internally tracks under 'success' and 'failure' keys.
        """
        key = "success" if success else "failure"
        if primitive_name not in self.primitive_calls:
            self.primitive_calls[primitive_name] = {"success": 0, "failure": 0}
        self.primitive_calls[primitive_name][key] += 1

    def update(self, metrics: Dict[str, Any]) -> None:
        """Batch update multiple metrics at once.

        This method efficiently updates multiple metrics in a single call. Numeric
        values are recorded as metrics, while non-numeric values are cached separately
        for metadata tracking.

        Args:
            metrics (Dict[str, Any]): Dictionary of metric name to value pairs.
                Numeric values (int, float, np.floating) are recorded as metrics.
                Non-numeric values are stored as metadata.

        Example:
            >>> collector = MetricsCollector()
            >>>
            >>> # Batch update training metrics
            >>> collector.update({
            ...     "loss": 0.42,
            ...     "accuracy": 0.95,
            ...     "learning_rate": 0.001,
            ...     "epoch": 10,
            ...     "model_name": "capibara-v3",  # Non-numeric metadata
            ...     "dataset": "custom-corpus"    # Non-numeric metadata
            ... })
            >>>
            >>> # Retrieve all including metadata
            >>> all_data = collector.get_all()
            >>> print(f"Model: {all_data['model_name']}")
            >>> print(f"Mean loss: {all_data['loss']['mean']:.4f}")

        Note:
            Non-numeric values passed to this method are accessible via get_all()
            but do not appear in get_stats() or get_toll_sttots() output.
        """
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating)):
                self.record(k, float(v))
            else:
                self._last_non_numeric[k] = v

    def get_stats(self, name: str) -> Dict[str, float]:
        """Calculate and return statistics for a specific metric.

        Computes mean, standard deviation, min, max, latest value, and sample count
        for the requested metric over the current window.

        Args:
            name (str): Name of the metric to analyze.

        Returns:
            Dict[str, float]: Dictionary containing statistical measures:
                - mean: Arithmetic mean of samples
                - std: Standard deviation
                - min: Minimum value
                - max: Maximum value
                - last: Most recent value
                - count: Number of samples in window
                Returns empty dict if metric name not found.

        Example:
            >>> collector = MetricsCollector()
            >>> for i in range(100):
            ...     collector.record("loss", 1.0 - i * 0.01)
            >>>
            >>> stats = collector.get_stats("loss")
            >>> print(f"Mean: {stats['mean']:.3f}")
            >>> print(f"Std dev: {stats['std']:.3f}")
            >>> print(f"Min: {stats['min']:.3f}, Max: {stats['max']:.3f}")
            >>> print(f"Latest: {stats['last']:.3f}")
            >>> print(f"Samples: {stats['count']}")

        Note:
            If no samples exist for the metric, returns an empty dictionary rather
            than raising an exception.
        """
        if name not in self.metrics:
            return {}
        values = np.asarray(self.metrics[name], dtype=float)
        if len(values) == 0:
            return {}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "last": float(values[-1]),
            "count": int(len(values)),
        }

    def get_all(self) -> Dict[str, Any]:
        """Get a complete snapshot of all metrics and metadata.

        Returns statistics for all numeric metrics combined with all non-numeric
        metadata values in a single dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - For each numeric metric: Dict with statistical measures
                - For each metadata key: The most recent non-numeric value

        Example:
            >>> collector = MetricsCollector()
            >>> collector.update({
            ...     "loss": 0.5,
            ...     "accuracy": 0.9,
            ...     "model": "capibara-v3"
            ... })
            >>> collector.record("loss", 0.4)
            >>>
            >>> snapshot = collector.get_all()
            >>> print(f"Loss stats: {snapshot['loss']}")
            >>> print(f"Accuracy stats: {snapshot['accuracy']}")
            >>> print(f"Model: {snapshot['model']}")

        Note:
            This method combines results from get_stats() for all metrics with
            the cached non-numeric values, providing a complete state snapshot.
        """
        out: Dict[str, Any] = {name: self.get_stats(name) for name in self.metrics}
        out.update(self._last_non_numeric)
        return out

    # Legacy alias methods with intentional typos for backward compatibility with tests
    def get_sttots(self, ntome: str) -> Dict[str, float]:  # pragma: no cover
        """Legacy alias for get_stats with typo. Use get_stats() in new code."""
        return self.get_stats(ntome)

    def get_toll_sttots(self) -> Dict[str, Dict[str, float]]:  # pragma: no cover
        """Legacy alias for getting all metric stats. Use get_all() in new code."""
        return {name: self.get_stats(name) for name in self.metrics}

    def get_core_performtonce_summtory(self) -> Dict[str, Any]:  # pragma: no cover
        """Generate performance summary for core primitive operations.

        Analyzes all recorded primitive operation calls and computes success rates,
        total call counts, and failure counts for each primitive.

        Returns:
            Dict[str, Any]: Summary dictionary containing:
                - primitives: Dict mapping primitive name to stats:
                    - total_calls: Total number of calls
                    - success_rate: Fraction of successful calls (0.0 to 1.0)
                    - success_count: Number of successful calls
                    - failure_count: Number of failed calls
                - total_primitives: Number of unique primitives tracked
                - total_calls: Total calls across all primitives

        Example:
            >>> collector = MetricsCollector()
            >>>
            >>> # Simulate primitive calls
            >>> for _ in range(95):
            ...     collector.record_primitive_ctoll("jax_pmap", success=True)
            >>> for _ in range(5):
            ...     collector.record_primitive_ctoll("jax_pmap", success=False)
            >>>
            >>> summary = collector.get_core_performtonce_summtory()
            >>> pmap_stats = summary['primitives']['jax_pmap']
            >>> print(f"Success rate: {pmap_stats['success_rate']:.1%}")
            >>> print(f"Total calls: {pmap_stats['total_calls']}")

        Note:
            Method name includes intentional typos ('performtonce', 'summtory') for
            backward compatibility. Primitives with zero calls are excluded from output.
        """
        summary: Dict[str, Any] = {}
        for primitive, counts in self.primitive_calls.items():
            total = counts["success"] + counts["failure"]
            if total == 0:
                continue
            success_rate = counts["success"] / total
            summary[primitive] = {
                "total_calls": total,
                "success_rate": success_rate,
                "success_count": counts["success"],
                "failure_count": counts["failure"],
            }
        return {
            "primitives": summary,
            "total_primitives": len(self.primitive_calls),
            "total_calls": sum(c["success"] + c["failure"] for c in self.primitive_calls.values()),
        }
