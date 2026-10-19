"""
Benchmark Runner Module
=======================

Core benchmark execution engine with timing, statistics, and result management.
"""

import gc
import time
import json
import statistics
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Dict, List, Optional, Any, Callable, TypeVar, Union,
    Tuple
)
from functools import wraps
from contextlib import contextmanager
import logging

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

torch = None  # type: ignore
HAS_TORCH = False


def _torch_import_works() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class TimingStats:
    """Statistical summary of timing measurements."""
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    std_ms: float
    iterations: int
    warmup_iterations: int


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    group: str
    timing: TimingStats
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    timestamp: str
    hardware_info: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    error_message: str = ""

    @property
    def throughput_ops_per_sec(self) -> float:
        """Calculate operations per second."""
        if self.timing.mean_ms > 0:
            return 1000.0 / self.timing.mean_ms
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['throughput_ops_per_sec'] = self.throughput_ops_per_sec
        return result


@dataclass
class BenchmarkSuite:
    """Collection of related benchmarks."""
    name: str
    description: str = ""
    benchmarks: List[Callable] = field(default_factory=list)
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    tags: List[str] = field(default_factory=list)


class BenchmarkRunner:
    """
    Main benchmark execution engine.

    Example:
        >>> runner = BenchmarkRunner()
        >>> runner.add_benchmark("matmul", matmul_benchmark, group="core")
        >>> results = runner.run()
        >>> runner.save_results("benchmark_results.json")
    """

    def __init__(
        self,
        warmup_iterations: int = 3,
        min_iterations: int = 5,
        max_iterations: int = 100,
        min_time_seconds: float = 1.0,
        max_time_seconds: float = 10.0,
        results_dir: Union[str, Path] = "benchmark_results"
    ):
        """
        Initialize benchmark runner.

        Args:
            warmup_iterations: Number of warmup runs before timing
            min_iterations: Minimum number of timed iterations
            max_iterations: Maximum number of timed iterations
            min_time_seconds: Minimum total benchmark time
            max_time_seconds: Maximum total benchmark time
            results_dir: Directory to store results
        """
        self.warmup_iterations = warmup_iterations
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.min_time_seconds = min_time_seconds
        self.max_time_seconds = max_time_seconds
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._benchmarks: Dict[str, Tuple[Callable, str, List[str]]] = {}
        self._suites: Dict[str, BenchmarkSuite] = {}
        self._results: List[BenchmarkResult] = []
        self._hardware_info = self._get_hardware_info()

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Collect hardware information."""
        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

        if HAS_PSUTIL:
            try:
                info["cpu_count"] = psutil.cpu_count()
                info["cpu_count_logical"] = psutil.cpu_count(logical=True)
                mem = psutil.virtual_memory()
                info["total_memory_gb"] = mem.total / (1024**3)
            except Exception:
                pass

        if _torch_import_works():
            try:
                import torch as _torch  # type: ignore
                info["torch_version"] = _torch.__version__
                info["cuda_available"] = _torch.cuda.is_available()
                if _torch.cuda.is_available():
                    info["cuda_device_count"] = _torch.cuda.device_count()
                    info["cuda_device_name"] = _torch.cuda.get_device_name(0)
            except Exception:
                pass

        # Try to get git commit
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                info["git_commit"] = result.stdout.strip()[:12]
        except Exception:
            pass

        return info

    def add_benchmark(
        self,
        name: str,
        func: Callable,
        group: str = "default",
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Register a benchmark function.

        Args:
            name: Unique benchmark name
            func: Benchmark function to run
            group: Group name for categorization
            tags: Optional tags for filtering
        """
        self._benchmarks[name] = (func, group, tags or [])
        logger.debug(f"Registered benchmark: {name} (group: {group})")

    def add_suite(self, suite: BenchmarkSuite) -> None:
        """
        Register a benchmark suite.

        Args:
            suite: BenchmarkSuite to register
        """
        self._suites[suite.name] = suite
        logger.debug(f"Registered suite: {suite.name}")

    def _measure_memory(self) -> float:
        """Get current memory usage in MB."""
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return 0.0

    def _run_single_benchmark(
        self,
        name: str,
        func: Callable,
        group: str,
        tags: List[str]
    ) -> BenchmarkResult:
        """Run a single benchmark and collect results."""
        gc.collect()
        memory_before = self._measure_memory()
        memory_peak = memory_before
        times_ms: List[float] = []
        error_message = ""
        passed = True

        try:
            # Warmup runs
            for _ in range(self.warmup_iterations):
                func()
                gc.collect()

            # Timed runs
            start_total = time.perf_counter()
            iteration = 0

            while True:
                gc.collect()

                start = time.perf_counter()
                func()
                end = time.perf_counter()

                times_ms.append((end - start) * 1000)
                iteration += 1

                # Track peak memory
                current_mem = self._measure_memory()
                memory_peak = max(memory_peak, current_mem)

                # Check termination conditions
                elapsed = time.perf_counter() - start_total
                if iteration >= self.min_iterations:
                    if elapsed >= self.min_time_seconds or iteration >= self.max_iterations:
                        break
                if elapsed >= self.max_time_seconds:
                    break

        except Exception as e:
            passed = False
            error_message = str(e)
            logger.error(f"Benchmark {name} failed: {e}")

        gc.collect()
        memory_after = self._measure_memory()

        # Calculate statistics
        if times_ms:
            timing = TimingStats(
                min_ms=min(times_ms),
                max_ms=max(times_ms),
                mean_ms=statistics.mean(times_ms),
                median_ms=statistics.median(times_ms),
                std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
                iterations=len(times_ms),
                warmup_iterations=self.warmup_iterations
            )
        else:
            timing = TimingStats(0, 0, 0, 0, 0, 0, self.warmup_iterations)

        return BenchmarkResult(
            name=name,
            group=group,
            timing=timing,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_peak_mb=memory_peak,
            timestamp=datetime.now().isoformat(),
            hardware_info=self._hardware_info,
            tags=tags,
            passed=passed,
            error_message=error_message
        )

    def run(
        self,
        filter_groups: Optional[List[str]] = None,
        filter_tags: Optional[List[str]] = None,
        filter_names: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """
        Run all registered benchmarks.

        Args:
            filter_groups: Only run benchmarks in these groups
            filter_tags: Only run benchmarks with these tags
            filter_names: Only run benchmarks with these names

        Returns:
            List of BenchmarkResult objects
        """
        self._results = []

        # Run individual benchmarks
        for name, (func, group, tags) in self._benchmarks.items():
            # Apply filters
            if filter_groups and group not in filter_groups:
                continue
            if filter_tags and not any(t in tags for t in filter_tags):
                continue
            if filter_names and name not in filter_names:
                continue

            logger.info(f"Running benchmark: {name}")
            result = self._run_single_benchmark(name, func, group, tags)
            self._results.append(result)

            # Log result
            status = "PASS" if result.passed else "FAIL"
            logger.info(
                f"  [{status}] {result.timing.mean_ms:.3f}ms "
                f"(+/- {result.timing.std_ms:.3f}ms, {result.timing.iterations} iterations)"
            )

        # Run suites
        for suite_name, suite in self._suites.items():
            if filter_groups and suite_name not in filter_groups:
                continue

            logger.info(f"Running suite: {suite_name}")

            # Setup
            if suite.setup:
                try:
                    suite.setup()
                except Exception as e:
                    logger.error(f"Suite {suite_name} setup failed: {e}")
                    continue

            # Run benchmarks in suite
            for bench_func in suite.benchmarks:
                bench_name = f"{suite_name}/{bench_func.__name__}"
                tags = suite.tags.copy()

                if filter_tags and not any(t in tags for t in filter_tags):
                    continue
                if filter_names and bench_name not in filter_names:
                    continue

                logger.info(f"Running benchmark: {bench_name}")
                result = self._run_single_benchmark(
                    bench_name, bench_func, suite_name, tags
                )
                self._results.append(result)

            # Teardown
            if suite.teardown:
                try:
                    suite.teardown()
                except Exception as e:
                    logger.warning(f"Suite {suite_name} teardown failed: {e}")

        return self._results

    def save_results(
        self,
        filename: Optional[str] = None,
        append_timestamp: bool = True
    ) -> Path:
        """
        Save benchmark results to JSON file.

        Args:
            filename: Output filename (default: benchmark_results.json)
            append_timestamp: Whether to append timestamp to filename

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = "benchmark_results.json"

        if append_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            filename = f"{stem}_{timestamp}{suffix}"

        output_path = self.results_dir / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "hardware_info": self._hardware_info,
            "config": {
                "warmup_iterations": self.warmup_iterations,
                "min_iterations": self.min_iterations,
                "max_iterations": self.max_iterations,
                "min_time_seconds": self.min_time_seconds,
                "max_time_seconds": self.max_time_seconds,
            },
            "results": [r.to_dict() for r in self._results],
            "summary": self._generate_summary()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Results saved to: {output_path}")
        return output_path

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self._results:
            return {}

        passed = sum(1 for r in self._results if r.passed)
        failed = len(self._results) - passed

        groups = {}
        for r in self._results:
            if r.group not in groups:
                groups[r.group] = []
            groups[r.group].append(r.timing.mean_ms)

        return {
            "total_benchmarks": len(self._results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self._results) if self._results else 0,
            "groups": {
                g: {
                    "count": len(times),
                    "mean_ms": statistics.mean(times) if times else 0,
                    "total_ms": sum(times)
                }
                for g, times in groups.items()
            }
        }

    def get_results(self) -> List[BenchmarkResult]:
        """Get current benchmark results."""
        return self._results.copy()

    def print_summary(self) -> None:
        """Print a summary of results to console."""
        summary = self._generate_summary()

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Total Benchmarks: {summary.get('total_benchmarks', 0)}")
        print(f"Passed: {summary.get('passed', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        print(f"Pass Rate: {summary.get('pass_rate', 0):.1%}")
        print()

        print("Results by Group:")
        for group, stats in summary.get('groups', {}).items():
            print(f"  {group}:")
            print(f"    Count: {stats['count']}")
            print(f"    Mean: {stats['mean_ms']:.3f}ms")
            print(f"    Total: {stats['total_ms']:.3f}ms")

        print()
        print("Top 10 Slowest Benchmarks:")
        sorted_results = sorted(
            self._results,
            key=lambda r: r.timing.mean_ms,
            reverse=True
        )[:10]
        for i, r in enumerate(sorted_results, 1):
            status = "OK" if r.passed else "FAIL"
            print(f"  {i}. [{status}] {r.name}: {r.timing.mean_ms:.3f}ms")

        print("=" * 60 + "\n")


def benchmark(
    group: str = "default",
    tags: Optional[List[str]] = None,
    warmup: int = 3
):
    """
    Decorator to mark a function as a benchmark.

    Example:
        >>> @benchmark(group="core", tags=["attention"])
        ... def test_attention_performance():
        ...     return compute_attention(q, k, v)
    """
    def decorator(func: F) -> F:
        func._benchmark_group = group
        func._benchmark_tags = tags or []
        func._benchmark_warmup = warmup
        return func
    return decorator


def run_benchmarks(
    benchmarks: Optional[List[Callable]] = None,
    output_file: Optional[str] = None,
    **kwargs
) -> List[BenchmarkResult]:
    """
    Convenience function to run benchmarks.

    Args:
        benchmarks: List of benchmark functions to run
        output_file: Optional file to save results
        **kwargs: Additional arguments for BenchmarkRunner

    Returns:
        List of BenchmarkResult objects
    """
    runner = BenchmarkRunner(**kwargs)

    if benchmarks:
        for func in benchmarks:
            name = func.__name__
            group = getattr(func, '_benchmark_group', 'default')
            tags = getattr(func, '_benchmark_tags', [])
            runner.add_benchmark(name, func, group, tags)

    results = runner.run()

    if output_file:
        runner.save_results(output_file)

    runner.print_summary()
    return results


@contextmanager
def benchmark_context(name: str):
    """
    Context manager for timing a code block.

    Example:
        >>> with benchmark_context("data_loading") as timer:
        ...     data = load_data()
        >>> print(f"Took {timer.elapsed_ms:.2f}ms")
    """
    class Timer:
        def __init__(self):
            self.start_time = 0.0
            self.end_time = 0.0
            self.elapsed_ms = 0.0

    timer = Timer()
    gc.collect()
    timer.start_time = time.perf_counter()

    try:
        yield timer
    finally:
        timer.end_time = time.perf_counter()
        timer.elapsed_ms = (timer.end_time - timer.start_time) * 1000
        logger.debug(f"Benchmark '{name}': {timer.elapsed_ms:.3f}ms")
