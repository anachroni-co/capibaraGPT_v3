"""
Comprehensive Performance Benchmarking for Meta-Consensus-Comp System

This module provides extensive benchmarking capabilities for the optimized meta-consensus system:
- Performance comparison across optimization levels
- Throughput and latency benchmarking
- Memory efficiency analysis
- Cost-performance trade-off analysis
- Scalability testing with concurrent loads
- Quality preservation validation
- Expert utilization efficiency
- Cache performance analysis
"""

import logging
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

# Import system components
from .optimized_meta_consensus import (
    OptimizedMetaConsensusSystem, OptimizedConsensusConfig, OptimizationLevel
)
from .meta_consensus_system import MetaConsensusSystem, MetaConsensusConfig, ConsensusMode
from .distributed_consensus_cache import DistributedConsensusCache, CacheConfig
from .consensus_memory_manager import ConsensusMemoryManager, MemoryPoolType

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """Types of benchmarks to run."""
    LATENCY = "latency"                   # Response time benchmarks
    THROUGHPUT = "throughput"             # Queries per second
    MEMORY_EFFICIENCY = "memory_efficiency"  # Memory usage optimization
    COST_ANALYSIS = "cost_analysis"       # Cost vs performance
    QUALITY_PRESERVATION = "quality_preservation"  # Quality impact of optimizations
    SCALABILITY = "scalability"           # Concurrent load testing
    CACHE_PERFORMANCE = "cache_performance"  # Cache hit rates and efficiency
    EXPERT_UTILIZATION = "expert_utilization"  # Expert selection efficiency

@dataclass
class BenchmarkConfig:
    """Configurestion for benchmarking."""
    
    # Test parameters
    num_queries: int = 100
    concurrent_users: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    query_complexity_levels: List[str] = field(default_factory=lambda: ["simple", "moderate", "complex"])
    optimization_levels: List[OptimizationLevel] = field(default_factory=lambda: [
        OptimizationLevel.BASIC, OptimizationLevel.STANDARD, 
        OptimizationLevel.ADVANCED, OptimizationLevel.EXTREME
    ])
    
    # Quality parameters
    quality_thresholds: List[float] = field(default_factory=lambda: [7.0, 8.0, 9.0])
    cost_limits: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    
    # Performance parameters
    warmup_queries: int = 10
    measurement_duration_seconds: int = 60
    memory_sampling_interval_seconds: int = 1
    
    # Output configuration
    save_results: bool = True
    output_directory: str = "benchmark_results"
    generate_plots: bool = True
    generate_report: bool = True

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    benchmark_type: BenchmarkType
    optimization_level: OptimizationLevel
    timestamp: datetime
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Throughput metrics
    queries_per_second: float = 0.0
    peak_qps: float = 0.0
    sustained_qps: float = 0.0
    
    # Quality metrics
    avg_quality_score: float = 0.0
    quality_variance: float = 0.0
    quality_degradation: float = 0.0
    
    # Resource metrics
    avg_memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    
    # Cost metrics
    avg_cost_per_query: float = 0.0
    total_cost: float = 0.0
    cost_efficiency: float = 0.0  # Quality per dollar
    
    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_efficiency: float = 0.0
    
    # Expert metrics
    expert_utilization_rate: float = 0.0
    avg_experts_per_query: float = 0.0
    
    # Error metrics
    success_rate: float = 0.0
    error_rate: float = 0.0
    timeout_rate: float = 0.0

@dataclass
class ComparisonReport:
    """Comprehensive comparison report."""
    
    # Performance comparison
    latency_improvement: Dict[str, float] = field(default_factory=dict)
    throughput_improvement: Dict[str, float] = field(default_factory=dict)
    memory_efficiency_gain: Dict[str, float] = field(default_factory=dict)
    
    # Quality analysis
    quality_preservation_rate: float = 0.0
    quality_consistency: float = 0.0
    
    # Cost analysis
    cost_reduction: Dict[str, float] = field(default_factory=dict)
    roi_analysis: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    optimal_configuration: Dict[str, Any] = field(default_factory=dict)
    performance_bottlenecks: List[str] = field(default_factory=list)
    optimization_recommendations: List[str] = field(default_factory=list)

class MetaConsensusCompBenchmark:
    """
    📊 Comprehensive Benchmarking System for Meta-Consensus-Comp
    
    Provides extensive benchmarking and analysis capabilities:
    - Performance comparison across optimization levels
    - Scalability testing with concurrent loads
    - Memory and resource efficiency analysis
    - Quality preservation validation
    - Cost-performance trade-off analysis
    - Cache and expert utilization metrics
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        
        # Create output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test data
        self.test_queries = self._generate_test_queries()
        
        logger.info(f"📊 Meta-Consensus-Comp Benchmark initialized")
    
    def _generate_test_queries(self) -> Dict[str, List[str]]:
        """Generates test queries for different complexity levels."""
        
        return {
            "simple": [
                "What is AI?",
                "Define machine learning",
                "Explain neural networks",
                "What is deep learning?",
                "Describe algorithms",
                "What is programming?",
                "Define data science",
                "Explain statistics",
                "What is Python?",
                "Describe databases"
            ],
            "moderate": [
                "Explain the difference between supervised and unsupervised learning",
                "How do convolutional neural networks work in computer vision?",
                "What are the advantages and disadvantages of different sorting algorithms?",
                "Describe the process of natural language processing in AI systems",
                "Explain the concept of reinforcement learning with examples",
                "How do recommendation systems work in e-commerce platforms?",
                "What is the role of feature engineering in machine learning?",
                "Describe the architecture and benefits of microservices",
                "Explain blockchain technology and its applications",
                "How do search engines rank and retrieve relevant content?"
            ],
            "complex": [
                "Design a distributed machine learning system that can handle petabyte-scale data with fault tolerance, explain the architecture, data flow, and optimization strategies",
                "Develop a comprehensive natural language understanding system that can handle multiple languages, dialects, and cultural contexts while maintaining high accuracy",
                "Create a quantum computing algorithm for optimizing supply chain logistics in real-time across multiple geographical regions",
                "Design an AI system for medical diagnosis that can integrate genomic data, medical imaging, patient history, and real-time monitoring",
                "Develop a financial trading algorithm that uses deep reinforcement learning and can adapt to changing market conditions while managing risk",
                "Create a multi-modal AI system that can understand and generate content across text, images, audio, and video with semantic consistency",
                "Design a autonomous vehicle navigation system that can handle complex urban environments with unpredictable human behavior",
                "Develop a climate modeling system that can predict regional weather patterns using satellite data and ground sensors",
                "Create an educational AI tutor that can adapt to individual learning styles and provide personalized curriculum",
                "Design a cybersecurity system that can detect and respond to novel attack patterns using behavioral analysis"
            ]
        }
    
    async def run_comprehensive_benchmark(self) -> ComparisonReport:
        """Run comprehensive benchmark suite."""
        
        logger.info("🚀 Starting comprehensive benchmark suite")
        
        # Run all benchmark types
        for benchmark_type in BenchmarkType:
            logger.info(f"Running {benchmark_type.value} benchmark...")
            await self._run_benchmark_type(benchmark_type)
        
        # Generate comparison report
        report = self._generate_comparison_report()
        
        # Save results
        if self.config.save_results:
            await self._save_benchmark_results()
        
        # Generate plots
        if self.config.generate_plots:
            await self._generate_benchmark_plots()
        
        # Generate report
        if self.config.generate_report:
            await self._generate_benchmark_report(report)
        
        logger.info("✅ Comprehensive benchmark completed")
        return report
    
    async def _run_benchmark_type(self, benchmark_type: BenchmarkType):
        """Run specific type of benchmark."""
        
        if benchmark_type == BenchmarkType.LATENCY:
            await self._benchmark_latency()
        elif benchmark_type == BenchmarkType.THROUGHPUT:
            await self._benchmark_throughput()
        elif benchmark_type == BenchmarkType.MEMORY_EFFICIENCY:
            await self._benchmark_memory_efficiency()
        elif benchmark_type == BenchmarkType.COST_ANALYSIS:
            await self._benchmark_cost_analysis()
        elif benchmark_type == BenchmarkType.QUALITY_PRESERVATION:
            await self._benchmark_quality_preservation()
        elif benchmark_type == BenchmarkType.SCALABILITY:
            await self._benchmark_scalability()
        elif benchmark_type == BenchmarkType.CACHE_PERFORMANCE:
            await self._benchmark_cache_performance()
        elif benchmark_type == BenchmarkType.EXPERT_UTILIZATION:
            await self._benchmark_expert_utilization()
    
    async def _benchmark_latency(self):
        """Benchmark response latency across optimization levels."""
        
        for opt_level in self.config.optimization_levels:
            logger.info(f"Benchmarking latency for {opt_level.value}")
            
            # Create optimized system
            system = await self._create_optimized_system(opt_level)
            
            latencies = []
            quality_scores = []
            
            # Warmup
            for _ in range(self.config.warmup_queries):
                await self._execute_test_query(system, "warmup query")
            
            # Measure latency
            for complexity in self.config.query_complexity_levels:
                queries = self.test_queries[complexity]
                
                for query in queries[:10]:  # Use subset for latency testing
                    start_time = time.time()
                    
                    result = await self._execute_test_query(system, query)
                    
                    latency = time.time() - start_time
                    latencies.append(latency * 1000)  # Convert to ms
                    
                    if result:
                        quality_scores.append(result.get("quality_score", 8.0))
            
            # Calculate statistics
            if latencies:
                benchmark_result = BenchmarkResult(
                    benchmark_type=BenchmarkType.LATENCY,
                    optimization_level=opt_level,
                    timestamp=datetime.now(),
                    avg_latency_ms=statistics.mean(latencies),
                    min_latency_ms=min(latencies),
                    max_latency_ms=max(latencies),
                    p95_latency_ms=np.percentile(latencies, 95),
                    p99_latency_ms=np.percentile(latencies, 99),
                    avg_quality_score=statistics.mean(quality_scores) if quality_scores else 0.0,
                    quality_variance=statistics.variance(quality_scores) if len(quality_scores) > 1 else 0.0
                )
                
                self.results.append(benchmark_result)
    
    async def _benchmark_throughput(self):
        """Benchmark query throughput across optimization levels."""
        
        for opt_level in self.config.optimization_levels:
            logger.info(f"Benchmarking throughput for {opt_level.value}")
            
            system = await self._create_optimized_system(opt_level)
            
            # Measure throughput over time
            start_time = time.time()
            end_time = start_time + self.config.measurement_duration_seconds
            
            queries_completed = 0
            total_cost = 0.0
            quality_scores = []
            
            query_idx = 0
            all_queries = []
            for complexity in self.config.query_complexity_levels:
                all_queries.extend(self.test_queries[complexity])
            
            while time.time() < end_time:
                query = all_queries[query_idx % len(all_queries)]
                query_idx += 1
                
                result = await self._execute_test_query(system, query)
                
                if result:
                    queries_completed += 1
                    total_cost += result.get("total_cost", 0.0)
                    quality_scores.append(result.get("quality_score", 8.0))
            
            duration = time.time() - start_time
            qps = queries_completed / duration if duration > 0 else 0
            
            benchmark_result = BenchmarkResult(
                benchmark_type=BenchmarkType.THROUGHPUT,
                optimization_level=opt_level,
                timestamp=datetime.now(),
                queries_per_second=qps,
                peak_qps=qps,  # Simplified - would track actual peak
                sustained_qps=qps,
                avg_quality_score=statistics.mean(quality_scores) if quality_scores else 0.0,
                total_cost=total_cost,
                avg_cost_per_query=total_cost / max(queries_completed, 1)
            )
            
            self.results.append(benchmark_result)
    
    async def _benchmark_memory_efficiency(self):
        """Benchmark memory efficiency across optimization levels."""
        
        for opt_level in self.config.optimization_levels:
            logger.info(f"Benchmarking memory efficiency for {opt_level.value}")
            
            # Monitor memory usage
            memory_samples = []
            
            # Get baseline memory
            gc.collect()
            baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            system = await self._create_optimized_system(opt_level)
            
            # Monitor memory during query processing
            async def memory_monitor():
                while True:
                    memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                    memory_samples.append(memory_mb - baseline_memory)
                    await asyncio.sleep(self.config.memory_sampling_interval_seconds)
            
            # Start memory monitoring
            monitor_task = asyncio.create_task(memory_monitor())
            
            try:
                # Process queries
                for complexity in self.config.query_complexity_levels:
                    queries = self.test_queries[complexity][:5]  # Use subset
                    
                    for query in queries:
                        await self._execute_test_query(system, query)
                        await asyncio.sleep(0.1)  # Small delay for monitoring
                
                # Stop monitoring
                monitor_task.cancel()
                
                # Calculate memory metrics
                if memory_samples:
                    benchmark_result = BenchmarkResult(
                        benchmark_type=BenchmarkType.MEMORY_EFFICIENCY,
                        optimization_level=opt_level,
                        timestamp=datetime.now(),
                        avg_memory_usage_mb=statistics.mean(memory_samples),
                        peak_memory_usage_mb=max(memory_samples)
                    )
                    
                    self.results.append(benchmark_result)
                    
            except asyncio.CancelledError:
                pass
            finally:
                if not monitor_task.cancelled():
                    monitor_task.cancel()
    
    async def _benchmark_scalability(self):
        """Benchmark system scalability with concurrent loads."""
        
        for opt_level in self.config.optimization_levels:
            for concurrent_users in self.config.concurrent_users:
                logger.info(f"Benchmarking scalability: {opt_level.value} with {concurrent_users} users")
                
                system = await self._create_optimized_system(opt_level)
                
                # Create concurrent tasks
                async def user_simulation(user_id: int):
                    user_latencies = []
                    user_quality_scores = []
                    user_costs = []
                    
                    for i in range(10):  # 10 queries per user
                        query = self.test_queries["moderate"][i % len(self.test_queries["moderate"])]
                        
                        start_time = time.time()
                        result = await self._execute_test_query(system, query)
                        latency = time.time() - start_time
                        
                        user_latencies.append(latency * 1000)
                        if result:
                            user_quality_scores.append(result.get("quality_score", 8.0))
                            user_costs.append(result.get("total_cost", 0.0))
                    
                    return {
                        "user_id": user_id,
                        "avg_latency_ms": statistics.mean(user_latencies),
                        "avg_quality": statistics.mean(user_quality_scores) if user_quality_scores else 0.0,
                        "total_cost": sum(user_costs)
                    }
                
                # Run concurrent users
                start_time = time.time()
                
                tasks = [user_simulation(i) for i in range(concurrent_users)]
                user_results = await asyncio.gather(*tasks)
                
                duration = time.time() - start_time
                total_queries = concurrent_users * 10
                
                # Aggregate results
                all_latencies = [ur["avg_latency_ms"] for ur in user_results]
                all_quality_scores = [ur["avg_quality"] for ur in user_results if ur["avg_quality"] > 0]
                total_cost = sum(ur["total_cost"] for ur in user_results)
                
                benchmark_result = BenchmarkResult(
                    benchmark_type=BenchmarkType.SCALABILITY,
                    optimization_level=opt_level,
                    timestamp=datetime.now(),
                    avg_latency_ms=statistics.mean(all_latencies),
                    queries_per_second=total_queries / duration,
                    avg_quality_score=statistics.mean(all_quality_scores) if all_quality_scores else 0.0,
                    total_cost=total_cost,
                    avg_cost_per_query=total_cost / total_queries
                )
                
                # Add concurrent user info to result
                benchmark_result.concurrent_users = concurrent_users
                
                self.results.append(benchmark_result)
    
    async def _benchmark_cache_performance(self):
        """Benchmark cache performance and efficiency."""
        
        # Create cache system
        cache_config = CacheConfig(
            l1_max_size_mb=128,
            l2_enabled=False,  # Disable Redis for testing
            l3_enabled=True,
            compression_type=CompressionType.LZ4 if LZ4_AVAILABLE else CompressionType.GZIP
        )
        
        cache = DistributedConsensusCache(cache_config)
        
        # Test cache operations
        cache_operations = 1000
        cache_hits = 0
        cache_misses = 0
        
        start_time = time.time()
        
        # Populate cache
        for i in range(cache_operations // 2):
            key = f"cache_test_{i}"
            value = {"data": f"test data {i}", "timestamp": time.time()}
            await cache.set(key, value)
        
        # Test cache retrieval (mix of hits and misses)
        for i in range(cache_operations):
            if i < cache_operations // 2:
                # Should be cache hit
                key = f"cache_test_{i}"
            else:
                # Should be cache miss
                key = f"cache_miss_{i}"
            
            result = await cache.get(key)
            
            if result is not None:
                cache_hits += 1
            else:
                cache_misses += 1
        
        duration = time.time() - start_time
        
        # Get cache statistics
        cache_stats = cache.get_comprehensive_stats()
        
        benchmark_result = BenchmarkResult(
            benchmark_type=BenchmarkType.CACHE_PERFORMANCE,
            optimization_level=OptimizationLevel.STANDARD,  # Cache is independent of opt level
            timestamp=datetime.now(),
            cache_hit_rate=cache_hits / (cache_hits + cache_misses),
            cache_efficiency=cache_stats["performance_metrics"]["overall_hit_rate"],
            avg_latency_ms=(duration / cache_operations) * 1000
        )
        
        self.results.append(benchmark_result)
    
    async def _create_optimized_system(self, opt_level: OptimizationLevel) -> OptimizedMetaConsensusSystem:
        """Creates optimized system for testing."""
        
        base_config = MetaConsensusConfig(
            system_name=f"Benchmark-{opt_level.value}",
            hf_api_token="test_token",
            enable_serverless=True,
            max_concurrent_experts=10
        )
        
        opt_config = OptimizedConsensusConfig(
            base_config=base_config,
            optimization_level=opt_level,
            enable_jit_compilation=opt_level in [OptimizationLevel.ADVANCED, OptimizationLevel.EXTREME],
            enable_memory_pooling=opt_level != OptimizationLevel.BASIC,
            enable_distributed_caching=opt_level in [OptimizationLevel.STANDARD, OptimizationLevel.ADVANCED, OptimizationLevel.EXTREME],
            enable_gpu_acceleration=opt_level == OptimizationLevel.EXTREME,
            tpu_v6_enabled=opt_level == OptimizationLevel.TPU_V6
        )
        
        system = OptimizedMetaConsensusSystem(opt_config)
        
        # Mock initialization for testing
        await system.initialize()
        if hasattr(system, 'initialize_optimizations'):
            await system.initialize_optimizations()
        
        return system
    
    async def _execute_test_query(self, system: OptimizedMetaConsensusSystem, query: str) -> Optional[Dict[str, Any]]:
        """Execute test query and return result."""
        
        try:
            if hasattr(system, 'process_optimized_query'):
                result = await system.process_optimized_query(query)
            else:
                result = await system.process_query(query)
            
            # Convert result to dictionary for analysis
            return {
                "response": result.response,
                "quality_score": result.quality_score,
                "confidence": result.confidence,
                "total_cost": result.total_cost,
                "response_time_ms": result.response_time_ms,
                "participating_experts": result.participating_experts
            }
            
        except Exception as e:
            logger.warning(f"Test query execution failed: {e}")
            return None
    
    def _generate_comparison_report(self) -> ComparisonReport:
        """Generates comprehensive comparison report."""
        
        report = ComparisonReport()
        
        # Group results by optimization level
        results_by_level = defaultdict(list)
        for result in self.results:
            results_by_level[result.optimization_level].append(result)
        
        # Calculate improvements relative to BASIC level
        basic_results = results_by_level.get(OptimizationLevel.BASIC, [])
        
        if basic_results:
            basic_latency = statistics.mean([r.avg_latency_ms for r in basic_results if r.avg_latency_ms > 0])
            basic_throughput = statistics.mean([r.queries_per_second for r in basic_results if r.queries_per_second > 0])
            basic_memory = statistics.mean([r.avg_memory_usage_mb for r in basic_results if r.avg_memory_usage_mb > 0])
            
            for opt_level, level_results in results_by_level.items():
                if opt_level == OptimizationLevel.BASIC:
                    continue
                
                # Calculate improvements
                level_latencies = [r.avg_latency_ms for r in level_results if r.avg_latency_ms > 0]
                level_throughputs = [r.queries_per_second for r in level_results if r.queries_per_second > 0]
                level_memory = [r.avg_memory_usage_mb for r in level_results if r.avg_memory_usage_mb > 0]
                
                if level_latencies and basic_latency > 0:
                    avg_latency = statistics.mean(level_latencies)
                    report.latency_improvement[opt_level.value] = (basic_latency - avg_latency) / basic_latency
                
                if level_throughputs and basic_throughput > 0:
                    avg_throughput = statistics.mean(level_throughputs)
                    report.throughput_improvement[opt_level.value] = (avg_throughput - basic_throughput) / basic_throughput
                
                if level_memory and basic_memory > 0:
                    avg_memory = statistics.mean(level_memory)
                    report.memory_efficiency_gain[opt_level.value] = (basic_memory - avg_memory) / basic_memory
        
        # Quality preservation analysis
        quality_results = [r for r in self.results if r.benchmark_type == BenchmarkType.QUALITY_PRESERVATION]
        if quality_results:
            quality_scores = [r.avg_quality_score for r in quality_results if r.avg_quality_score > 0]
            if quality_scores:
                report.quality_preservation_rate = statistics.mean(quality_scores) / 10.0
                report.quality_consistency = 1.0 - (statistics.stdev(quality_scores) / statistics.mean(quality_scores))
        
        # Generate recommendations
        report.optimization_recommendations = self._generate_optimization_recommendations(results_by_level)
        
        return report
    
    def _generate_optimization_recommendations(self, results_by_level: Dict[OptimizationLevel, List[BenchmarkResult]]) -> List[str]:
        """Generates optimization recommendations based on benchmark results."""
        
        recommendations = []
        
        # Analyze latency improvements
        latency_improvements = {}
        for opt_level, level_results in results_by_level.items():
            latency_results = [r for r in level_results if r.benchmark_type == BenchmarkType.LATENCY]
            if latency_results:
                avg_latency = statistics.mean([r.avg_latency_ms for r in latency_results])
                latency_improvements[opt_level] = avg_latency
        
        if latency_improvements:
            best_latency_level = min(latency_improvements.keys(), key=lambda k: latency_improvements[k])
            recommendations.append(f"For best latency: Use {best_latency_level.value} optimization level")
        
        # Analyze throughput
        throughput_improvements = {}
        for opt_level, level_results in results_by_level.items():
            throughput_results = [r for r in level_results if r.benchmark_type == BenchmarkType.THROUGHPUT]
            if throughput_results:
                avg_throughput = statistics.mean([r.queries_per_second for r in throughput_results])
                throughput_improvements[opt_level] = avg_throughput
        
        if throughput_improvements:
            best_throughput_level = max(throughput_improvements.keys(), key=lambda k: throughput_improvements[k])
            recommendations.append(f"For best throughput: Use {best_throughput_level.value} optimization level")
        
        # Memory efficiency recommendations
        memory_results = [r for r in self.results if r.benchmark_type == BenchmarkType.MEMORY_EFFICIENCY]
        if memory_results:
            efficient_levels = [r.optimization_level for r in memory_results if r.avg_memory_usage_mb < 500]
            if efficient_levels:
                recommendations.append(f"For memory efficiency: Consider {efficient_levels[0].value} level")
        
        # Cost efficiency
        cost_results = [r for r in self.results if r.avg_cost_per_query > 0 and r.avg_quality_score > 0]
        if cost_results:
            cost_efficiencies = [(r.avg_quality_score / r.avg_cost_per_query, r.optimization_level) for r in cost_results]
            if cost_efficiencies:
                best_cost_efficiency = max(cost_efficiencies, key=lambda x: x[0])
                recommendations.append(f"For cost efficiency: Use {best_cost_efficiency[1].value} optimization level")
        
        return recommendations
    
    async def _save_benchmark_results(self):
        """Save benchmark results to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results as JSON
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        results_data = []
        for result in self.results:
            result_dict = {
                "benchmark_type": result.benchmark_type.value,
                "optimization_level": result.optimization_level.value,
                "timestamp": result.timestamp.isoformat(),
                "metrics": {
                    "avg_latency_ms": result.avg_latency_ms,
                    "queries_per_second": result.queries_per_second,
                    "avg_quality_score": result.avg_quality_score,
                    "avg_memory_usage_mb": result.avg_memory_usage_mb,
                    "avg_cost_per_query": result.avg_cost_per_query,
                    "cache_hit_rate": result.cache_hit_rate,
                    "success_rate": result.success_rate
                }
            }
            results_data.append(result_dict)
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"✅ Benchmark results saved to {results_file}")
    
    async def _generate_benchmark_plots(self):
        """Generates visualization plots for benchmark results."""
        
        try:
            # Create plots directory
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Latency comparison plot
            self._plot_latency_comparison(plots_dir)
            
            # Throughput comparison plot
            self._plot_throughput_comparison(plots_dir)
            
            # Memory efficiency plot
            self._plot_memory_efficiency(plots_dir)
            
            # Quality vs Cost plot
            self._plot_quality_vs_cost(plots_dir)
            
            logger.info(f"✅ Benchmark plots generated in {plots_dir}")
            
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
    
    def _plot_latency_comparison(self, plots_dir: Path):
        """Plot latency comparison across optimization levels."""
        
        latency_results = [r for r in self.results if r.benchmark_type == BenchmarkType.LATENCY]
        
        if latency_results:
            levels = [r.optimization_level.value for r in latency_results]
            latencies = [r.avg_latency_ms for r in latency_results]
            
            plt.figure(figsize=(10, 6))
            plt.bar(levels, latencies, color='skyblue')
            plt.title('Average Latency by Optimization Level')
            plt.xlabel('Optimization Level')
            plt.ylabel('Average Latency (ms)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(plots_dir / "latency_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_throughput_comparison(self, plots_dir: Path):
        """Plot throughput comparison across optimization levels."""
        
        throughput_results = [r for r in self.results if r.benchmark_type == BenchmarkType.THROUGHPUT]
        
        if throughput_results:
            levels = [r.optimization_level.value for r in throughput_results]
            throughputs = [r.queries_per_second for r in throughput_results]
            
            plt.figure(figsize=(10, 6))
            plt.bar(levels, throughputs, color='lightgreen')
            plt.title('Throughput by Optimization Level')
            plt.xlabel('Optimization Level')
            plt.ylabel('Queries per Second')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(plots_dir / "throughput_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_memory_efficiency(self, plots_dir: Path):
        """Plot memory efficiency across optimization levels."""
        
        memory_results = [r for r in self.results if r.benchmark_type == BenchmarkType.MEMORY_EFFICIENCY]
        
        if memory_results:
            levels = [r.optimization_level.value for r in memory_results]
            avg_memory = [r.avg_memory_usage_mb for r in memory_results]
            peak_memory = [r.peak_memory_usage_mb for r in memory_results]
            
            x = np.arange(len(levels))
            width = 0.35
            
            plt.figure(figsize=(12, 6))
            plt.bar(x - width/2, avg_memory, width, label='Average Memory', color='lightcoral')
            plt.bar(x + width/2, peak_memory, width, label='Peak Memory', color='darkred')
            
            plt.title('Memory Usage by Optimization Level')
            plt.xlabel('Optimization Level')
            plt.ylabel('Memory Usage (MB)')
            plt.xticks(x, levels, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(plots_dir / "memory_efficiency.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_quality_vs_cost(self, plots_dir: Path):
        """Plot quality vs cost trade-off."""
        
        cost_results = [r for r in self.results if r.avg_cost_per_query > 0 and r.avg_quality_score > 0]
        
        if cost_results:
            costs = [r.avg_cost_per_query for r in cost_results]
            qualities = [r.avg_quality_score for r in cost_results]
            levels = [r.optimization_level.value for r in cost_results]
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(costs, qualities, c=range(len(levels)), cmap='viridis', s=100, alpha=0.7)
            
            # Add labels for each point
            for i, level in enumerate(levels):
                plt.annotate(level, (costs[i], qualities[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            plt.title('Quality vs Cost Trade-off by Optimization Level')
            plt.xlabel('Average Cost per Query ($)')
            plt.ylabel('Average Quality Score')
            plt.colorbar(scatter, label='Optimization Level')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(plots_dir / "quality_vs_cost.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    async def _generate_benchmark_report(self, report: ComparisonReport):
        """Generates comprehensive benchmark report."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_content = f"""
# Meta-Consensus-Comp Benchmark Report

**Generated**: {timestamp}
**Test Configuration**: {self.config.num_queries} queries, {len(self.config.optimization_levels)} optimization levels

## Executive Summary

### Performance Improvements
"""
        
        # Add latency improvements
        if report.latency_improvement:
            report_content += "\n#### Latency Improvements\n"
            for level, improvement in report.latency_improvement.items():
                report_content += f"- **{level}**: {improvement:.1%} faster\n"
        
        # Add throughput improvements
        if report.throughput_improvement:
            report_content += "\n#### Throughput Improvements\n"
            for level, improvement in report.throughput_improvement.items():
                report_content += f"- **{level}**: {improvement:.1%} higher throughput\n"
        
        # Add memory efficiency
        if report.memory_efficiency_gain:
            report_content += "\n#### Memory Efficiency Gains\n"
            for level, gain in report.memory_efficiency_gain.items():
                report_content += f"- **{level}**: {gain:.1%} memory reduction\n"
        
        # Add quality analysis
        report_content += f"""
### Quality Analysis
- **Quality Preservation Rate**: {report.quality_preservation_rate:.1%}
- **Quality Consistency**: {report.quality_consistency:.1%}

### Recommendations
"""
        
        for recommendation in report.optimization_recommendations:
            report_content += f"- {recommendation}\n"
        
        # Add detailed results
        report_content += "\n## Detailed Results\n\n"
        
        for benchmark_type in BenchmarkType:
            type_results = [r for r in self.results if r.benchmark_type == benchmark_type]
            if type_results:
                report_content += f"### {benchmark_type.value.title().replace('_', ' ')}\n\n"
                
                for result in type_results:
                    report_content += f"**{result.optimization_level.value}**:\n"
                    if result.avg_latency_ms > 0:
                        report_content += f"- Latency: {result.avg_latency_ms:.1f}ms\n"
                    if result.queries_per_second > 0:
                        report_content += f"- Throughput: {result.queries_per_second:.1f} QPS\n"
                    if result.avg_quality_score > 0:
                        report_content += f"- Quality: {result.avg_quality_score:.1f}/10\n"
                    if result.avg_cost_per_query > 0:
                        report_content += f"- Cost: ${result.avg_cost_per_query:.4f}\n"
                    if result.cache_hit_rate > 0:
                        report_content += f"- Cache Hit Rate: {result.cache_hit_rate:.1%}\n"
                    report_content += "\n"
        
        # Save report
        report_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"✅ Benchmark report generated: {report_file}")


# Factory functions
def create_benchmark_suite(config: Optional[BenchmarkConfig] = None) -> MetaConsensusCompBenchmark:
    """Creates benchmark suite with configuration."""
    
    if config is None:
        config = BenchmarkConfig()
    
    return MetaConsensusCompBenchmark(config)

def create_quick_benchmark(num_queries: int = 20) -> MetaConsensusCompBenchmark:
    """Creates quick benchmark for basic testing."""
    
    config = BenchmarkConfig(
        num_queries=num_queries,
        concurrent_users=[1, 5],
        optimization_levels=[OptimizationLevel.BASIC, OptimizationLevel.ADVANCED],
        warmup_queries=5,
        measurement_duration_seconds=30,
        generate_plots=False
    )
    
    return MetaConsensusCompBenchmark(config)


# Export main components
__all__ = [
    'MetaConsensusCompBenchmark',
    'BenchmarkConfig',
    'BenchmarkResult',
    'ComparisonReport',
    'BenchmarkType',
    'create_benchmark_suite',
    'create_quick_benchmark'
]


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create benchmark suite
        benchmark = create_quick_benchmark(num_queries=10)
        
        print("📊 Meta-Consensus-Comp Benchmark Suite")
        print("=" * 50)
        
        # Run comprehensive benchmark
        report = await benchmark.run_comprehensive_benchmark()
        
        print("\n🏆 Benchmark Results Summary:")
        
        # Latency improvements
        if report.latency_improvement:
            print("\n⚡ Latency Improvements:")
            for level, improvement in report.latency_improvement.items():
                print(f"  {level}: {improvement:.1%} faster")
        
        # Throughput improvements
        if report.throughput_improvement:
            print("\n🚀 Throughput Improvements:")
            for level, improvement in report.throughput_improvement.items():
                print(f"  {level}: {improvement:.1%} higher")
        
        # Memory efficiency
        if report.memory_efficiency_gain:
            print("\n🧠 Memory Efficiency Gains:")
            for level, gain in report.memory_efficiency_gain.items():
                print(f"  {level}: {gain:.1%} reduction")
        
        # Quality analysis
        print(f"\n✅ Quality Preservation: {report.quality_preservation_rate:.1%}")
        print(f"✅ Quality Consistency: {report.quality_consistency:.1%}")
        
        # Recommendations
        if report.optimization_recommendations:
            print(f"\n💡 Recommendations:")
            for rec in report.optimization_recommendations:
                print(f"  • {rec}")
        
        print(f"\n📁 Results saved to: {benchmark.output_dir}")
    
    import asyncio
    asyncio.run(main())