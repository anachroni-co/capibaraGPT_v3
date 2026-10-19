"""
Memory Profiler Module for CapibaraGPT
=====================================

Advanced memory profiling utilities for detecting memory leaks,
tracking allocations, and generating detailed memory reports.

Features:
- Function-level memory profiling with decorators
- Memory leak detection
- Allocation tracking for tensors/arrays
- HTML/JSON report generation
- Integration with training loops
"""

import gc
import sys
import time
import json
import traceback
import functools
import threading
import weakref
from dataclasses import dataclass, field, asdict
from typing import (
    Dict, List, Optional, Any, Callable, TypeVar,
    Union, Tuple, Set
)
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    tracemalloc = None
    HAS_TRACEMALLOC = False

torch = None  # type: ignore
HAS_TORCH = False
_TORCH_IMPORT_ATTEMPTED = False

def _ensure_torch() -> None:
    """Lazily import torch only when needed.

    This avoids hard failures in CPU-only environments where torch (or its CUDA
    DLLs) may be missing.
    """
    global torch, HAS_TORCH, _TORCH_IMPORT_ATTEMPTED
    if _TORCH_IMPORT_ATTEMPTED:
        return
    _TORCH_IMPORT_ATTEMPTED = True
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            torch = None
            HAS_TORCH = False
            return

        import torch as _torch  # type: ignore
        torch = _torch
        HAS_TORCH = True
    except Exception:
        torch = None
        HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

import logging

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""
    timestamp: float
    rss_bytes: int  # Resident Set Size
    vms_bytes: int  # Virtual Memory Size
    heap_bytes: int  # Python heap
    gpu_bytes: int  # GPU memory (if available)
    num_objects: int  # Python object count
    label: str = ""

    @property
    def rss_mb(self) -> float:
        return self.rss_bytes / (1024 * 1024)

    @property
    def gpu_mb(self) -> float:
        return self.gpu_bytes / (1024 * 1024)


@dataclass
class AllocationRecord:
    """Record of a memory allocation."""
    size_bytes: int
    timestamp: float
    traceback: str
    dtype: str = ""
    shape: Tuple = ()
    device: str = "cpu"

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass
class FunctionProfile:
    """Memory profile for a function call."""
    function_name: str
    module: str
    call_count: int = 0
    total_allocated_bytes: int = 0
    peak_memory_bytes: int = 0
    total_time_seconds: float = 0.0
    memory_samples: List[int] = field(default_factory=list)


@dataclass
class LeakReport:
    """Report of a potential memory leak."""
    location: str
    growth_bytes: int
    growth_rate_bytes_per_sec: float
    sample_count: int
    confidence: float  # 0.0 to 1.0
    first_seen: float
    last_seen: float
    traceback: str = ""


class MemoryProfiler:
    """
    Advanced memory profiler for Python applications.

    Example:
        >>> profiler = MemoryProfiler()
        >>> profiler.start()
        >>>
        >>> # Your code here
        >>> result = expensive_operation()
        >>>
        >>> profiler.stop()
        >>> profiler.generate_report("memory_report.html")
    """

    _instance: Optional['MemoryProfiler'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for global profiler access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        sample_interval: float = 1.0,
        track_allocations: bool = True,
        detect_leaks: bool = True,
        max_snapshots: int = 10000
    ):
        """
        Initialize the memory profiler.

        Args:
            sample_interval: Seconds between memory samples
            track_allocations: Whether to track individual allocations
            detect_leaks: Whether to run leak detection
            max_snapshots: Maximum number of snapshots to retain
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.sample_interval = sample_interval
        self.track_allocations = track_allocations
        self.detect_leaks = detect_leaks
        self.max_snapshots = max_snapshots

        self._snapshots: List[MemorySnapshot] = []
        self._allocations: Dict[int, AllocationRecord] = {}
        self._function_profiles: Dict[str, FunctionProfile] = {}
        self._leak_candidates: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        self._leak_reports: List[LeakReport] = []

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process() if HAS_PSUTIL else None
        self._start_time: float = 0.0

        self._tensor_refs: weakref.WeakSet = weakref.WeakSet()

        self._initialized = True
        logger.info("MemoryProfiler initialized")

    def start(self) -> None:
        """Start memory profiling."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._snapshots.clear()
        self._leak_reports.clear()

        # Start tracemalloc if available
        if HAS_TRACEMALLOC and self.track_allocations:
            tracemalloc.start(25)  # 25 frames deep

        # Start sampling thread
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

        logger.info("Memory profiling started")

    def stop(self) -> None:
        """Stop memory profiling."""
        self._running = False

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        if HAS_TRACEMALLOC and tracemalloc.is_tracing():
            tracemalloc.stop()

        # Run final leak detection
        if self.detect_leaks:
            self._analyze_leaks()

        logger.info("Memory profiling stopped")

    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a manual memory snapshot."""
        snapshot = self._create_snapshot(label)
        self._snapshots.append(snapshot)

        # Trim old snapshots
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots = self._snapshots[-self.max_snapshots:]

        return snapshot

    def _create_snapshot(self, label: str = "") -> MemorySnapshot:
        """Create a memory snapshot."""
        rss_bytes = 0
        vms_bytes = 0
        heap_bytes = 0
        gpu_bytes = 0

        if self._process:
            try:
                mem_info = self._process.memory_info()
                rss_bytes = mem_info.rss
                vms_bytes = mem_info.vms
            except Exception:
                pass

        # Python heap via tracemalloc
        if HAS_TRACEMALLOC and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            heap_bytes = current

        # GPU memory
        _ensure_torch()
        if HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_bytes = torch.cuda.memory_allocated()
            except Exception:
                pass

        # Object count
        num_objects = len(gc.get_objects())

        return MemorySnapshot(
            timestamp=time.time(),
            rss_bytes=rss_bytes,
            vms_bytes=vms_bytes,
            heap_bytes=heap_bytes,
            gpu_bytes=gpu_bytes,
            num_objects=num_objects,
            label=label
        )

    def _sample_loop(self) -> None:
        """Background sampling loop."""
        while self._running:
            try:
                self.take_snapshot()
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.warning(f"Memory sampling error: {e}")

    def track_tensor(self, tensor: Any, name: str = "") -> None:
        """
        Track a tensor for memory analysis.

        Args:
            tensor: PyTorch tensor or NumPy array
            name: Optional name for identification
        """
        if not self.track_allocations:
            return

        try:
            size_bytes = 0
            dtype = ""
            shape = ()
            device = "cpu"

            _ensure_torch()
            if HAS_TORCH and isinstance(tensor, torch.Tensor):
                size_bytes = tensor.element_size() * tensor.nelement()
                dtype = str(tensor.dtype)
                shape = tuple(tensor.shape)
                device = str(tensor.device)
                self._tensor_refs.add(tensor)
            elif HAS_NUMPY and isinstance(tensor, np.ndarray):
                size_bytes = tensor.nbytes
                dtype = str(tensor.dtype)
                shape = tensor.shape

            record = AllocationRecord(
                size_bytes=size_bytes,
                timestamp=time.time(),
                traceback=self._get_short_traceback(),
                dtype=dtype,
                shape=shape,
                device=device
            )

            self._allocations[id(tensor)] = record

        except Exception as e:
            logger.debug(f"Failed to track tensor: {e}")

    def _get_short_traceback(self, limit: int = 5) -> str:
        """Get a shortened traceback string."""
        frames = traceback.extract_stack()[:-2][-limit:]
        return "\n".join(
            f"  {f.filename}:{f.lineno} in {f.name}"
            for f in frames
        )

    def _analyze_leaks(self) -> None:
        """Analyze snapshots for potential memory leaks."""
        if len(self._snapshots) < 10:
            return

        # Look for consistent memory growth
        rss_values = [s.rss_bytes for s in self._snapshots]
        times = [s.timestamp for s in self._snapshots]

        # Simple linear regression for growth rate
        n = len(rss_values)
        if n < 2:
            return

        sum_x = sum(times)
        sum_y = sum(rss_values)
        sum_xy = sum(t * r for t, r in zip(times, rss_values))
        sum_xx = sum(t * t for t in times)

        denom = n * sum_xx - sum_x * sum_x
        if denom == 0:
            return

        slope = (n * sum_xy - sum_x * sum_y) / denom

        # If growth rate exceeds threshold, report potential leak
        threshold_bytes_per_sec = 1024 * 1024  # 1 MB/sec

        if slope > threshold_bytes_per_sec:
            duration = times[-1] - times[0]
            growth = rss_values[-1] - rss_values[0]

            # Calculate confidence based on consistency
            increases = sum(1 for i in range(1, n) if rss_values[i] > rss_values[i-1])
            confidence = increases / (n - 1) if n > 1 else 0.0

            if confidence > 0.6:  # More than 60% of samples show growth
                self._leak_reports.append(LeakReport(
                    location="global",
                    growth_bytes=growth,
                    growth_rate_bytes_per_sec=slope,
                    sample_count=n,
                    confidence=confidence,
                    first_seen=times[0],
                    last_seen=times[-1]
                ))

        # Check GPU memory separately
        _ensure_torch()
        if HAS_TORCH and torch.cuda.is_available():
            gpu_values = [s.gpu_bytes for s in self._snapshots]
            gpu_growth = gpu_values[-1] - gpu_values[0] if gpu_values else 0

            if gpu_growth > 100 * 1024 * 1024:  # 100 MB growth
                self._leak_reports.append(LeakReport(
                    location="GPU",
                    growth_bytes=gpu_growth,
                    growth_rate_bytes_per_sec=gpu_growth / (times[-1] - times[0]) if times[-1] != times[0] else 0,
                    sample_count=n,
                    confidence=0.8,
                    first_seen=times[0],
                    last_seen=times[-1]
                ))

    def get_stats(self) -> Dict[str, Any]:
        """Get current profiling statistics."""
        stats = {
            "running": self._running,
            "snapshot_count": len(self._snapshots),
            "allocation_count": len(self._allocations),
            "leak_reports": len(self._leak_reports),
            "profiled_functions": len(self._function_profiles),
        }

        if self._snapshots:
            latest = self._snapshots[-1]
            stats["current_rss_mb"] = latest.rss_mb
            stats["current_gpu_mb"] = latest.gpu_mb
            stats["object_count"] = latest.num_objects

        if self._process:
            try:
                stats["cpu_percent"] = self._process.cpu_percent()
            except Exception:
                pass

        return stats

    def get_top_allocations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N memory allocations by size."""
        sorted_allocs = sorted(
            self._allocations.values(),
            key=lambda x: x.size_bytes,
            reverse=True
        )[:n]

        return [
            {
                "size_mb": a.size_mb,
                "dtype": a.dtype,
                "shape": a.shape,
                "device": a.device,
                "traceback": a.traceback
            }
            for a in sorted_allocs
        ]

    def get_leak_reports(self) -> List[Dict[str, Any]]:
        """Get all detected memory leak reports."""
        return [
            {
                "location": r.location,
                "growth_mb": r.growth_bytes / (1024 * 1024),
                "growth_rate_mb_per_sec": r.growth_rate_bytes_per_sec / (1024 * 1024),
                "confidence": r.confidence,
                "duration_seconds": r.last_seen - r.first_seen
            }
            for r in self._leak_reports
        ]

    def generate_report(
        self,
        output_path: Union[str, Path],
        format: str = "html"
    ) -> Path:
        """
        Generate a memory profiling report.

        Args:
            output_path: Path for the output report
            format: Report format ('html' or 'json')

        Returns:
            Path to generated report
        """
        output_path = Path(output_path)

        report_data = {
            "generated_at": datetime.now().isoformat(),
            "profiling_duration_seconds": time.time() - self._start_time if self._start_time else 0,
            "stats": self.get_stats(),
            "leak_reports": self.get_leak_reports(),
            "top_allocations": self.get_top_allocations(20),
            "function_profiles": {
                name: asdict(profile)
                for name, profile in self._function_profiles.items()
            },
            "memory_timeline": [
                {
                    "timestamp": s.timestamp,
                    "rss_mb": s.rss_mb,
                    "gpu_mb": s.gpu_mb,
                    "label": s.label
                }
                for s in self._snapshots[-1000:]  # Last 1000 samples
            ]
        }

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        else:
            self._generate_html_report(output_path, report_data)

        logger.info(f"Memory report generated: {output_path}")
        return output_path

    def _generate_html_report(self, output_path: Path, data: Dict) -> None:
        """Generate an HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Memory Profile Report - CapibaraGPT</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px;
                margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #4CAF50; }}
        .stat-label {{ color: #666; font-size: 14px; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; }}
        .danger {{ background: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; }}
        pre {{ background: #2d2d2d; color: #f8f8f2; padding: 15px;
              border-radius: 4px; overflow-x: auto; font-size: 12px; }}
        .chart {{ height: 300px; background: white; border-radius: 8px;
                 padding: 20px; margin: 15px 0; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>Memory Profile Report</h1>
        <p>Generated: {data['generated_at']}</p>
        <p>Profiling Duration: {data['profiling_duration_seconds']:.1f} seconds</p>

        <div class="card">
            <h2>Summary Statistics</h2>
            <div class="stat">
                <div class="stat-value">{data['stats'].get('current_rss_mb', 0):.1f}</div>
                <div class="stat-label">Current RSS (MB)</div>
            </div>
            <div class="stat">
                <div class="stat-value">{data['stats'].get('current_gpu_mb', 0):.1f}</div>
                <div class="stat-label">GPU Memory (MB)</div>
            </div>
            <div class="stat">
                <div class="stat-value">{data['stats'].get('object_count', 0):,}</div>
                <div class="stat-label">Python Objects</div>
            </div>
            <div class="stat">
                <div class="stat-value">{data['stats'].get('snapshot_count', 0):,}</div>
                <div class="stat-label">Samples Collected</div>
            </div>
        </div>

        {"".join(self._render_leak_warning(leak) for leak in data['leak_reports'])}

        <div class="card">
            <h2>Memory Timeline</h2>
            <div class="chart">
                <canvas id="memoryChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Top Allocations</h2>
            <table>
                <tr><th>Size (MB)</th><th>Type</th><th>Shape</th><th>Device</th><th>Location</th></tr>
                {"".join(self._render_allocation_row(a) for a in data['top_allocations'][:10])}
            </table>
        </div>

        <div class="card">
            <h2>Function Profiles</h2>
            <table>
                <tr><th>Function</th><th>Calls</th><th>Total Allocated</th><th>Peak Memory</th><th>Total Time</th></tr>
                {"".join(self._render_function_row(name, prof) for name, prof in data['function_profiles'].items())}
            </table>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('memoryChart').getContext('2d');
        const timeline = {json.dumps(data['memory_timeline'])};
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: timeline.map((_, i) => i),
                datasets: [{{
                    label: 'RSS (MB)',
                    data: timeline.map(t => t.rss_mb),
                    borderColor: '#4CAF50',
                    fill: false
                }}, {{
                    label: 'GPU (MB)',
                    data: timeline.map(t => t.gpu_mb),
                    borderColor: '#2196F3',
                    fill: false
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});
    </script>
</body>
</html>"""

        with open(output_path, 'w') as f:
            f.write(html)

    def _render_leak_warning(self, leak: Dict) -> str:
        """Render a leak warning HTML block."""
        severity = "danger" if leak['confidence'] > 0.8 else "warning"
        return f"""
        <div class="card {severity}">
            <h3>Potential Memory Leak Detected: {leak['location']}</h3>
            <p>Growth: {leak['growth_mb']:.2f} MB ({leak['growth_rate_mb_per_sec']:.2f} MB/sec)</p>
            <p>Confidence: {leak['confidence']*100:.0f}%</p>
        </div>
        """

    def _render_allocation_row(self, alloc: Dict) -> str:
        """Render an allocation table row."""
        return f"""
        <tr>
            <td>{alloc['size_mb']:.2f}</td>
            <td>{alloc['dtype']}</td>
            <td>{alloc['shape']}</td>
            <td>{alloc['device']}</td>
            <td><pre>{alloc['traceback'][:200]}</pre></td>
        </tr>
        """

    def _render_function_row(self, name: str, prof: Dict) -> str:
        """Render a function profile table row."""
        return f"""
        <tr>
            <td>{name}</td>
            <td>{prof['call_count']}</td>
            <td>{prof['total_allocated_bytes'] / (1024*1024):.2f} MB</td>
            <td>{prof['peak_memory_bytes'] / (1024*1024):.2f} MB</td>
            <td>{prof['total_time_seconds']:.3f}s</td>
        </tr>
        """


# Global profiler instance
_profiler: Optional[MemoryProfiler] = None


def get_profiler() -> MemoryProfiler:
    """Get or create the global memory profiler."""
    global _profiler
    if _profiler is None:
        _profiler = MemoryProfiler()
    return _profiler


def profile_memory(func: F) -> F:
    """
    Decorator to profile memory usage of a function.

    Example:
        >>> @profile_memory
        ... def train_step(batch):
        ...     # training code
        ...     return loss
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = get_profiler()
        func_name = f"{func.__module__}.{func.__qualname__}"

        # Get or create function profile
        if func_name not in profiler._function_profiles:
            profiler._function_profiles[func_name] = FunctionProfile(
                function_name=func.__name__,
                module=func.__module__ or ""
            )

        profile = profiler._function_profiles[func_name]

        # Take before snapshot
        gc.collect()
        before = profiler._create_snapshot(f"before:{func_name}")
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Take after snapshot
            end_time = time.perf_counter()
            gc.collect()
            after = profiler._create_snapshot(f"after:{func_name}")

            # Update profile
            profile.call_count += 1
            allocated = max(0, after.rss_bytes - before.rss_bytes)
            profile.total_allocated_bytes += allocated
            profile.peak_memory_bytes = max(profile.peak_memory_bytes, after.rss_bytes)
            profile.total_time_seconds += end_time - start_time
            profile.memory_samples.append(after.rss_bytes)

    return wrapper  # type: ignore


@contextmanager
def memory_profile_block(name: str):
    """
    Context manager to profile a code block.

    Example:
        >>> with memory_profile_block("data_loading"):
        ...     data = load_large_dataset()
    """
    profiler = get_profiler()

    gc.collect()
    before = profiler.take_snapshot(f"start:{name}")
    start_time = time.perf_counter()

    try:
        yield
    finally:
        end_time = time.perf_counter()
        gc.collect()
        after = profiler.take_snapshot(f"end:{name}")

        growth = after.rss_bytes - before.rss_bytes
        duration = end_time - start_time

        logger.info(
            f"Memory block '{name}': "
            f"{growth / (1024*1024):.2f} MB allocated in {duration:.3f}s"
        )


def check_for_leaks(threshold_mb: float = 100.0) -> List[Dict[str, Any]]:
    """
    Check for memory leaks based on current profiling data.

    Args:
        threshold_mb: Minimum growth to consider a leak

    Returns:
        List of leak reports
    """
    profiler = get_profiler()
    profiler._analyze_leaks()

    return [
        leak for leak in profiler.get_leak_reports()
        if leak['growth_mb'] > threshold_mb
    ]


def print_memory_summary() -> None:
    """Print a summary of current memory usage."""
    profiler = get_profiler()
    stats = profiler.get_stats()

    print("\n" + "=" * 50)
    print("MEMORY PROFILE SUMMARY")
    print("=" * 50)
    print(f"RSS Memory:     {stats.get('current_rss_mb', 0):.1f} MB")
    print(f"GPU Memory:     {stats.get('current_gpu_mb', 0):.1f} MB")
    print(f"Python Objects: {stats.get('object_count', 0):,}")
    print(f"Snapshots:      {stats.get('snapshot_count', 0)}")
    print(f"Leak Reports:   {stats.get('leak_reports', 0)}")
    print("=" * 50 + "\n")


class TrainingMemoryTracker:
    """
    Memory tracker specifically designed for training loops.

    Example:
        >>> tracker = TrainingMemoryTracker()
        >>> for epoch in range(num_epochs):
        ...     tracker.start_epoch(epoch)
        ...     for batch in dataloader:
        ...         tracker.start_step()
        ...         loss = train_step(batch)
        ...         tracker.end_step(loss=loss.item())
        ...     tracker.end_epoch()
        >>> tracker.report()
    """

    def __init__(self, log_interval: int = 100):
        """
        Initialize training memory tracker.

        Args:
            log_interval: Log memory every N steps
        """
        self.log_interval = log_interval
        self.profiler = get_profiler()

        self._epoch_data: List[Dict] = []
        self._step_data: List[Dict] = []
        self._current_epoch: int = 0
        self._current_step: int = 0
        self._step_start_time: float = 0.0
        self._epoch_start_time: float = 0.0
        self._epoch_start_memory: int = 0

    def start_epoch(self, epoch: int) -> None:
        """Mark the start of an epoch."""
        self._current_epoch = epoch
        self._current_step = 0
        self._epoch_start_time = time.time()

        gc.collect()
        snapshot = self.profiler.take_snapshot(f"epoch_{epoch}_start")
        self._epoch_start_memory = snapshot.rss_bytes

    def end_epoch(self) -> Dict[str, Any]:
        """Mark the end of an epoch and return stats."""
        gc.collect()
        snapshot = self.profiler.take_snapshot(f"epoch_{self._current_epoch}_end")

        epoch_data = {
            "epoch": self._current_epoch,
            "duration_seconds": time.time() - self._epoch_start_time,
            "memory_growth_mb": (snapshot.rss_bytes - self._epoch_start_memory) / (1024 * 1024),
            "final_rss_mb": snapshot.rss_mb,
            "final_gpu_mb": snapshot.gpu_mb,
            "steps": self._current_step
        }

        self._epoch_data.append(epoch_data)

        logger.info(
            f"Epoch {self._current_epoch} complete: "
            f"memory growth = {epoch_data['memory_growth_mb']:.2f} MB"
        )

        return epoch_data

    def start_step(self) -> None:
        """Mark the start of a training step."""
        self._step_start_time = time.time()

    def end_step(self, **metrics) -> None:
        """Mark the end of a training step."""
        self._current_step += 1

        if self._current_step % self.log_interval == 0:
            snapshot = self.profiler.take_snapshot(
                f"epoch_{self._current_epoch}_step_{self._current_step}"
            )

            step_data = {
                "epoch": self._current_epoch,
                "step": self._current_step,
                "rss_mb": snapshot.rss_mb,
                "gpu_mb": snapshot.gpu_mb,
                "step_time": time.time() - self._step_start_time,
                **metrics
            }

            self._step_data.append(step_data)

    def report(self, output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Generate a training memory report.

        Args:
            output_path: Optional path to save JSON report

        Returns:
            Report data dictionary
        """
        report = {
            "total_epochs": len(self._epoch_data),
            "total_steps": sum(e["steps"] for e in self._epoch_data),
            "epochs": self._epoch_data,
            "step_samples": self._step_data[-100:],  # Last 100 steps
            "memory_trend": self._calculate_trend(),
            "leak_detected": self._detect_training_leak()
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report

    def _calculate_trend(self) -> str:
        """Calculate memory trend across epochs."""
        if len(self._epoch_data) < 2:
            return "insufficient_data"

        memory_values = [e["final_rss_mb"] for e in self._epoch_data]
        growth = memory_values[-1] - memory_values[0]

        if growth > 100:
            return "increasing"
        elif growth < -100:
            return "decreasing"
        return "stable"

    def _detect_training_leak(self) -> bool:
        """Detect if there's a memory leak during training."""
        if len(self._epoch_data) < 3:
            return False

        # Check if memory consistently increases each epoch
        growths = [
            self._epoch_data[i]["final_rss_mb"] - self._epoch_data[i-1]["final_rss_mb"]
            for i in range(1, len(self._epoch_data))
        ]

        # If most epochs show growth, likely a leak
        positive_growths = sum(1 for g in growths if g > 10)
        return positive_growths / len(growths) > 0.7
