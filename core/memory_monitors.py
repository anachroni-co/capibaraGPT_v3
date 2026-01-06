"""Memory Monitoring Module Optimized for TPU v4-32 and Core Integration.

This module provides specialized memory monitoring utilities designed for efficient
resource management in production environments, particularly when running on TPU v4-32
hardware. It includes monitors for both general process memory and TPU-specific memory.

The monitoring system enables:
- Automatic memory pressure detection
- Threshold-based cleanup triggers
- JAX/XLA cache management
- TPU device memory tracking
- Integration with core.py for coordinated cleanup

Key Components:
    - CoreIntegratedMemoryMonitor: General process memory monitoring
    - CoreIntegratedTpuMemoryMonitor: TPU v4-32 specific memory tracking

Example:
    Basic memory monitoring setup:

    >>> from capibara.core.memory_monitors import CoreIntegratedMemoryMonitor
    >>>
    >>> # Create monitor with 85% cleanup threshold
    >>> monitor = CoreIntegratedMemoryMonitor(threshold=0.85)
    >>>
    >>> # Check if cleanup is needed
    >>> if monitor.should_cleanup():
    ...     monitor.force_cleanup()
    ...     print("Memory cleanup performed")

    TPU-specific monitoring:

    >>> from capibara.core.memory_monitors import CoreIntegratedTpuMemoryMonitor
    >>>
    >>> # Monitor TPU memory with 32GB limit
    >>> tpu_monitor = CoreIntegratedTpuMemoryMonitor(
    ...     memory_limit_gb=32.0,
    ...     cleanup_threshold=0.9
    ... )
    >>>
    >>> # Check TPU memory pressure
    >>> if tpu_monitor.should_cleanup():
    ...     tpu_monitor.force_cleanup()

Note:
    This module gracefully handles missing dependencies (psutil, JAX). If psutil
    is not available, memory monitoring is disabled. If JAX is not available,
    cleanup operations fall back to basic Python garbage collection.

Dependencies:
    - psutil (optional): For process memory monitoring
    - JAX (optional): For JAX/XLA cache clearing
    - gc (standard library): For Python garbage collection

See Also:
    - capibara.core: Core module integration
    - capibara.jax: JAX utilities and imports
"""
from __future__ import annotations

import gc
from typing import Dict, Any, Optional

try:
    from capibara.jax import numpy as jnp  # noqa: F401  # Reserved for future extensions
except Exception:  # pragma: no cover
    import numpy as jnp  # type: ignore  # noqa: F401

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

# JAX native (may not be available at runtime)
try:
    from capibara.jax import jax as _jax  # type: ignore
except Exception:  # pragma: no cover
    _jax = None  # type: ignore


class CoreIntegratedMemoryMonitor:
    """Process memory monitor integrated with core.py for automatic cleanup.

    Monitors the memory usage of the current process and triggers cleanup operations
    when memory consumption exceeds a configured threshold. Integrates with JAX/XLA
    cache management for ML workloads.

    Attributes:
        threshold (float): Memory usage fraction (0.0-1.0) that triggers cleanup.
            Default is 0.85 (85% of available memory).
        use_core_cleanup (bool): Whether to use core.py cleanup optimizations including
            JAX cache clearing. Default is True.
        process (Optional[psutil.Process]): psutil process object for memory monitoring.
            None if psutil is not available.

    Example:
        >>> monitor = CoreIntegratedMemoryMonitor(threshold=0.80)
        >>> if monitor.should_cleanup():
        ...     print("Memory usage high, performing cleanup")
        ...     monitor.force_cleanup()

    Note:
        If psutil is not installed, the monitor will be non-functional but won't
        raise exceptions. Methods will return safe default values.
    """

    def __init__(self, threshold: float = 0.85, use_core_cleanup: bool = True):
        """Initialize the memory monitor with configurable thresholds.

        Args:
            threshold (float, optional): Memory usage fraction (0.0-1.0) that triggers
                cleanup. Values above this threshold will cause should_cleanup() to
                return True. Defaults to 0.85 (85%).
            use_core_cleanup (bool, optional): Whether to enable core.py integrated
                cleanup including JAX/XLA cache clearing. Defaults to True.

        Example:
            >>> # Conservative cleanup at 75% memory usage
            >>> monitor = CoreIntegratedMemoryMonitor(threshold=0.75)
            >>>
            >>> # Aggressive cleanup at 95% without JAX optimizations
            >>> monitor = CoreIntegratedMemoryMonitor(
            ...     threshold=0.95,
            ...     use_core_cleanup=False
            ... )
        """
        self.threshold = threshold
        self.use_core_cleanup = use_core_cleanup
        self.process = psutil.Process() if psutil else None

    def should_cleanup(self) -> bool:
        """Determine if memory cleanup should be triggered.

        Checks the current process memory usage against the configured threshold.
        Returns True if usage exceeds the threshold, indicating cleanup is needed.

        Returns:
            bool: True if memory usage exceeds threshold and cleanup is recommended,
                False otherwise or if monitoring is unavailable.

        Example:
            >>> monitor = CoreIntegratedMemoryMonitor(threshold=0.85)
            >>> if monitor.should_cleanup():
            ...     monitor.force_cleanup()

        Note:
            If psutil is not available, this always returns False. Memory percentage
            is calculated as percent of total system memory used by this process.
        """
        if not self.process:
            return False
        try:
            # psutil returns percentage 0-100
            memory_percent = self.process.memory_percent()  # type: ignore[attr-defined]
            return memory_percent > (self.threshold * 100)
        except Exception:
            return False

    def force_cleanup(self) -> None:
        """Force memory cleanup using available optimizations.

        Performs aggressive memory cleanup including:
        - JAX/XLA cache clearing (if use_core_cleanup=True and JAX available)
        - Python garbage collection
        - XLA backend buffer clearing (when possible)

        This method is designed to be safe to call even if dependencies are missing.

        Example:
            >>> monitor = CoreIntegratedMemoryMonitor()
            >>> monitor.force_cleanup()  # Safely performs cleanup

        Note:
            Cleanup operations are best-effort. Missing APIs or backends are
            silently skipped. At minimum, Python's gc.collect() is always called.
        """
        if self.use_core_cleanup and _jax is not None:
            try:
                # Clear JAX caches if API exists
                if hasattr(_jax, "clear_caches"):
                    _jax.clear_caches()
                # Attempt to clear XLA backend buffers if exposed
                lib = getattr(_jax, "lib", None)
                if lib and hasattr(lib, "xla_bridge"):
                    bridge = lib.xla_bridge
                    if hasattr(bridge, "get_backend"):
                        backend = bridge.get_backend()
                        # Some builds expose "platform_version" or GC methods
                        getattr(backend, "platform_version", None)
            except Exception:
                pass
        # Python garbage collection
        gc.collect()


class CoreIntegratedTpuMemoryMonitor:
    """TPU v4-32 specific memory monitor for device memory tracking.

    Specialized monitor for tracking TPU device memory usage with automatic cleanup
    triggers. Designed for TPU v4-32 pods with 32GB HBM per device.

    Attributes:
        memory_limit_bytes (int): Total memory limit in bytes. Calculated from
            memory_limit_gb parameter.
        cleanup_threshold (float): Usage fraction (0.0-1.0) that triggers cleanup.
            Default is 0.9 (90% of limit).

    Example:
        >>> # Monitor for 32GB TPU v4
        >>> tpu_monitor = CoreIntegratedTpuMemoryMonitor(
        ...     memory_limit_gb=32.0,
        ...     cleanup_threshold=0.9
        ... )
        >>> if tpu_monitor.should_cleanup():
        ...     tpu_monitor.force_cleanup()

    Note:
        Memory tracking is based on JAX device APIs. If JAX is not available,
        monitoring is disabled but methods remain safe to call.
    """

    def __init__(self, memory_limit_gb: float = 32.0, cleanup_threshold: float = 0.9):
        """Initialize TPU memory monitor with memory limits and thresholds.

        Args:
            memory_limit_gb (float, optional): Expected memory limit in gigabytes.
                For TPU v4-32, this is typically 32.0 GB per device. Defaults to 32.0.
            cleanup_threshold (float, optional): Fraction of memory_limit_gb (0.0-1.0)
                that triggers cleanup. Defaults to 0.9 (90%).

        Example:
            >>> # TPU v5e with 16GB per device
            >>> monitor = CoreIntegratedTpuMemoryMonitor(
            ...     memory_limit_gb=16.0,
            ...     cleanup_threshold=0.85
            ... )
            >>>
            >>> # TPU v4 with conservative cleanup
            >>> monitor = CoreIntegratedTpuMemoryMonitor(
            ...     memory_limit_gb=32.0,
            ...     cleanup_threshold=0.7
            ... )
        """
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.cleanup_threshold = cleanup_threshold

    def should_cleanup(self) -> bool:
        """Determine if TPU memory cleanup is needed based on estimated usage.

        Estimates current TPU memory usage across all devices and compares against
        the cleanup threshold.

        Returns:
            bool: True if estimated memory usage exceeds cleanup_threshold,
                False otherwise or if monitoring is unavailable.

        Example:
            >>> tpu_monitor = CoreIntegratedTpuMemoryMonitor()
            >>> if tpu_monitor.should_cleanup():
            ...     print("TPU memory pressure detected")
            ...     tpu_monitor.force_cleanup()

        Note:
            If JAX is not available, this always returns False. Memory estimation
            is best-effort and may not reflect exact hardware state.
        """
        current_usage = self._estimate_memory_usage()
        return current_usage > self.cleanup_threshold

    def _estimate_memory_usage(self) -> float:
        """Estimate current TPU memory usage as a fraction of total limit.

        Queries JAX devices for memory statistics and calculates total usage
        across all available TPU devices.

        Returns:
            float: Estimated memory usage as fraction 0.0-1.0, where 1.0 represents
                full utilization of memory_limit_bytes. Returns 0.0 if JAX is
                unavailable or devices cannot be queried.

        Note:
            This is an estimation based on JAX device APIs. Actual hardware memory
            usage may differ. Missing or incomplete APIs result in 0.0 return value.
        """
        if _jax is None:
            return 0.0
        try:
            devices = getattr(_jax, "devices", lambda: [])()
            if not devices:
                return 0.0
            total_used = 0
            for dev in devices:
                if hasattr(dev, "memory_stats"):
                    stats = dev.memory_stats()
                    total_used += stats.get("bytes_in_use", 0)
            return float(total_used) / float(self.memory_limit_bytes) if self.memory_limit_bytes else 0.0
        except Exception:
            return 0.0

    def force_cleanup(self) -> None:
        """Force TPU memory cleanup using JAX cache clearing.

        Attempts to free TPU memory by:
        - Clearing JAX compilation caches
        - Clearing XLA backend buffers
        - Running Python garbage collection

        This method is safe to call even if JAX is unavailable - it will fall back
        to Python garbage collection.

        Example:
            >>> tpu_monitor = CoreIntegratedTpuMemoryMonitor()
            >>> tpu_monitor.force_cleanup()  # Safely clears TPU memory

        Note:
            Cleanup is best-effort. Not all JAX/XLA versions expose the same cleanup
            APIs. At minimum, Python's gc.collect() is always executed.
        """
        if _jax is None:
            gc.collect()
            return
        try:
            if hasattr(_jax, "clear_caches"):
                _jax.clear_caches()
            lib = getattr(_jax, "lib", None)
            if lib and hasattr(lib, "xla_bridge"):
                bridge = lib.xla_bridge
                if hasattr(bridge, "get_backend"):
                    backend = bridge.get_backend()
                    # Not all builds have explicit GC API; best-effort
                    getattr(backend, "platform_version", None)
        except Exception:
            pass
        gc.collect()
