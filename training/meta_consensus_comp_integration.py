"""
Meta-Consensus-Comp Integration Layer

This is the main integration layer that brings together all meta-consensus-comp components:
- Optimized Meta-Consensus System with comp optimizations
- Advanced consensus algorithms with JIT compilation
- TPU v6 optimization for maximum performance
- Memory management and distributed caching
- Real-time monitoring and analytics
- Comprehensive benchmarking and testing
"""

import logging
import asyncio
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json

# Import all meta-consensus-comp components
from .optimized_meta_consensus import (
    OptimizedMetaConsensusSystem, OptimizedConsensusConfig, OptimizationLevel
)
from .optimized_consensus_router import (
    OptimizedConsensusRouter, OptimizedRoutingConfig, OptimizedRoutingMode
)
from .tpu_v6_consensus_optimizer import (
    TPUv6ConsensusOptimizer, TPUv6ConsensusConfig, TPUConsensusMode
)
from .consensus_memory_manager import (
    ConsensusMemoryManager, MemoryPoolType, create_consensus_memory_manager
)
from .distributed_consensus_cache import (
    DistributedConsensusCache, CacheConfig, get_consensus_cache
)
from .meta_consensus_comp_benchmark import (
    MetaConsensusCompBenchmark, BenchmarkConfig, create_benchmark_suite
)
from .monitoring_dashboard import (
    MonitoringDashboard, MetaConsensusMonitor, DashboardConfig
)
from .advanced_consensus_algorithms import (
    AdvancedConsensusEngine, ConsensusAlgorithm
)

# Import base meta-consensus components
from .meta_consensus_system import (
    MetaConsensusSystem, MetaConsensusConfig, QueryContext, 
    ConsensusResult, ConsensusMode, create_meta_consensus_system
)

logger = logging.getLogger(__name__)

class SystemProfile(Enum):
    """Predefined system profiles for different use cases."""
    DEVELOPMENT = "development"           # Development and testing
    PRODUCTION_BALANCED = "production_balanced"  # Balanced production setup
    PRODUCTION_PERFORMANCE = "production_performance"  # Maximum performance
    PRODUCTION_COST_OPTIMIZED = "production_cost_optimized"  # Cost optimized
    RESEARCH_EXPERIMENTAL = "research_experimental"  # Research and experimentation
    TPU_V6_MAXIMUM = "tpu_v6_maximum"    # TPU v6 maximum performance

@dataclass
class IntegratedSystemConfig:
    """Configurestion for the integrated meta-consensus-comp system."""
    
    # System profile
    profile: SystemProfile = SystemProfile.PRODUCTION_BALANCED
    
    # Core system configuration
    system_name: str = "MetaConsensusComp"
    version: str = "1.0.0"
    
    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED
    routing_mode: OptimizedRoutingMode = OptimizedRoutingMode.HYBRID_OPTIMIZED
    consensus_mode: ConsensusMode = ConsensusMode.ADAPTIVE
    
    # Resource limits
    max_memory_mb: int = 2048
    max_concurrent_queries: int = 50
    max_cost_per_query: float = 0.02
    
    # Feature flags
    enable_tpu_v6: bool = False
    enable_monitoring: bool = True
    enable_caching: bool = True
    enable_benchmarking: bool = True
    
    # API and networking
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    monitoring_port: int = 8080
    
    # HuggingFace integration
    hf_api_token: str = ""
    enable_serverless: bool = True

@dataclass
class SystemStatus:
    """Current system status and health."""
    
    # System state
    is_initialized: bool = False
    is_healthy: bool = False
    uptime_seconds: float = 0.0
    
    # Performance metrics
    current_qps: float = 0.0
    avg_latency_ms: float = 0.0
    avg_quality_score: float = 0.0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    
    # Cache performance
    cache_hit_rate: float = 0.0
    cache_size_mb: float = 0.0
    
    # Expert utilization
    active_experts: int = 0
    expert_utilization_rate: float = 0.0
    
    # Error tracking
    error_rate: float = 0.0
    last_error: Optional[str] = None

class MetaConsensusCompSystem:
    """
    🎭 Meta-Consensus-Comp Integrated System
    
    The main integration layer that combines all meta-consensus-comp components:
    - Optimized meta-consensus with comp optimizations (5-15x speedup)
    - Advanced consensus algorithms with JIT compilation
    - TPU v6 optimization for maximum performance
    - Memory management and distributed caching
    - Real-time monitoring and analytics
    - Comprehensive benchmarking and testing
    """
    
    def __init__(self, config: IntegratedSystemConfig):
        self.config = config
        self.start_time = datetime.now()
        self.status = SystemStatus()
        
        # Core components
        self.optimized_system: Optional[OptimizedMetaConsensusSystem] = None
        self.consensus_router: Optional[OptimizedConsensusRouter] = None
        self.tpu_optimizer: Optional[TPUv6ConsensusOptimizer] = None
        self.memory_manager: Optional[ConsensusMemoryManager] = None
        self.cache_system: Optional[DistributedConsensusCache] = None
        self.monitoring_system: Optional[MetaConsensusMonitor] = None
        self.benchmark_system: Optional[MetaConsensusCompBenchmark] = None
        
        # Performance tracking
        self.query_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        logger.info(f"🎭 Meta-Consensus-Comp System created with profile: {config.profile.value}")
    
    async def initialize(self) -> bool:
        """Initialize the complete integrated system."""
        
        try:
            logger.info("🚀 Initializing Meta-Consensus-Comp System...")
            
            # Apply profile-specific configurations
            await self._apply_system_profile()
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize optimization components
            await self._initialize_optimization_components()
            
            # Initialize monitoring
            if self.config.enable_monitoring:
                await self._initialize_monitoring()
            
            # Initialize benchmarking
            if self.config.enable_benchmarking:
                await self._initialize_benchmarking()
            
            # System health check
            health_check = await self._perform_health_check()
            
            if health_check:
                self.status.is_initialized = True
                self.status.is_healthy = True
                logger.info("✅ Meta-Consensus-Comp System initialized successfully")
                return True
            else:
                logger.error("❌ System health check failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def _apply_system_profile(self):
        """Apply profile-specific configurations."""
        
        profile = self.config.profile
        
        if profile == SystemProfile.DEVELOPMENT:
            self.config.optimization_level = OptimizationLevel.STANDARD
            self.config.max_memory_mb = 512
            self.config.max_concurrent_queries = 10
            self.config.enable_tpu_v6 = False
            
        elif profile == SystemProfile.PRODUCTION_BALANCED:
            self.config.optimization_level = OptimizationLevel.ADVANCED
            self.config.max_memory_mb = 2048
            self.config.max_concurrent_queries = 50
            self.config.enable_monitoring = True
            
        elif profile == SystemProfile.PRODUCTION_PERFORMANCE:
            self.config.optimization_level = OptimizationLevel.EXTREME
            self.config.routing_mode = OptimizedRoutingMode.HYBRID_OPTIMIZED
            self.config.max_memory_mb = 4096
            self.config.max_concurrent_queries = 100
            
        elif profile == SystemProfile.PRODUCTION_COST_OPTIMIZED:
            self.config.optimization_level = OptimizationLevel.STANDARD
            self.config.routing_mode = OptimizedRoutingMode.CACHE_OPTIMIZED
            self.config.max_cost_per_query = 0.01
            self.config.enable_caching = True
            
        elif profile == SystemProfile.TPU_V6_MAXIMUM:
            self.config.optimization_level = OptimizationLevel.TPU_V6
            self.config.enable_tpu_v6 = True
            self.config.max_memory_mb = 8192
            self.config.max_concurrent_queries = 200
            
        logger.info(f"Applied profile: {profile.value}")
    
    async def _initialize_core_components(self):
        """Initialize core meta-consensus components."""
        
        # Base meta-consensus configuration
        base_config = MetaConsensusConfig(
            system_name=self.config.system_name,
            hf_api_token=self.config.hf_api_token,
            enable_serverless=self.config.enable_serverless,
            max_concurrent_experts=min(self.config.max_concurrent_queries // 5, 20),
            max_cost_per_query=self.config.max_cost_per_query
        )
        
        # Optimized system configuration
        opt_config = OptimizedConsensusConfig(
            base_config=base_config,
            optimization_level=self.config.optimization_level,
            tpu_v6_enabled=self.config.enable_tpu_v6,
            memory_pool_size_mb=self.config.max_memory_mb // 4  # 25% for memory pool
        )
        
        # Create optimized system
        self.optimized_system = OptimizedMetaConsensusSystem(opt_config)
        await self.optimized_system.initialize()
        
        if hasattr(self.optimized_system, 'initialize_optimizations'):
            await self.optimized_system.initialize_optimizations()
        
        logger.info("✅ Core components initialized")
    
    async def _initialize_optimization_components(self):
        """Initialize optimization-specific components."""
        
        # Memory manager
        self.memory_manager = create_consensus_memory_manager(
            total_memory_mb=self.config.max_memory_mb,
            pool_distribution={
                MemoryPoolType.TENSOR_POOL: 0.3,
                MemoryPoolType.EMBEDDING_POOL: 0.25,
                MemoryPoolType.RESPONSE_POOL: 0.25,
                MemoryPoolType.CACHE_POOL: 0.15,
                MemoryPoolType.TEMPORARY_POOL: 0.05
            }
        )
        
        # Cache system
        if self.config.enable_caching:
            cache_config = CacheConfig(
                l1_max_size_mb=self.config.max_memory_mb // 8,  # 12.5% for L1 cache
                l2_enabled=False,  # Disable Redis by default
                l3_enabled=True,
                l3_max_size_mb=self.config.max_memory_mb // 4  # 25% for L3 cache
            )
            self.cache_system = get_consensus_cache(cache_config)
        
        # Optimized router
        router_config = OptimizedRoutingConfig(
            routing_mode=self.config.routing_mode,
            enable_jit_compilation=self.config.optimization_level != OptimizationLevel.BASIC,
            use_memory_pool=True,
            cache_expert_scores=self.config.enable_caching
        )
        
        from .optimized_consensus_router import create_optimized_consensus_router
        self.consensus_router = create_optimized_consensus_router(router_config)
        
        if hasattr(self.consensus_router, 'initialize_optimizations'):
            await self.consensus_router.initialize_optimizations()
        
        # TPU v6 optimizer (if enabled)
        if self.config.enable_tpu_v6:
            from .tpu_v6_consensus_optimizer import create_tpu_v6_consensus_optimizer
            self.tpu_optimizer = create_tpu_v6_consensus_optimizer(
                tpu_cores=64,
                consensus_mode=TPUConsensusMode.MESH_DISTRIBUTED
            )
        
        logger.info("✅ Optimization components initialized")
    
    async def _initialize_monitoring(self):
        """Initialize monitoring and analytics."""
        
        dashboard_config = DashboardConfig(
            host=self.config.api_host,
            port=self.config.monitoring_port,
            dashboard_title=f"{self.config.system_name} Monitor"
        )
        
        self.monitoring_system = MetaConsensusMonitor(dashboard_config)
        
        logger.info(f"✅ Monitoring initialized on port {self.config.monitoring_port}")
    
    async def _initialize_benchmarking(self):
        """Initialize benchmarking system."""
        
        benchmark_config = BenchmarkConfig(
            num_queries=50,
            optimization_levels=[
                OptimizationLevel.BASIC,
                self.config.optimization_level
            ],
            output_directory=str(Path("benchmarks") / self.config.system_name.lower())
        )
        
        self.benchmark_system = create_benchmark_suite(benchmark_config)
        
        logger.info("✅ Benchmarking system initialized")
    
    async def _perform_health_check(self) -> bool:
        """Perform comprehensive system health check."""
        
        try:
            health_checks = {
                "core_system": False,
                "memory_manager": False,
                "cache_system": False,
                "router": False
            }
            
            # Check core system
            if self.optimized_system:
                test_result = await self._test_core_system()
                health_checks["core_system"] = test_result
            
            # Check memory manager
            if self.memory_manager:
                memory_stats = self.memory_manager.get_comprehensive_statistics()
                health_checks["memory_manager"] = memory_stats["global_metrics"]["memory_efficiency"] > 0.5
            
            # Check cache system
            if self.cache_system:
                # Test cache operations
                test_key = "health_check"
                test_value = {"timestamp": time.time()}
                
                set_success = await self.cache_system.set(test_key, test_value)
                get_result = await self.cache_system.get(test_key)
                
                health_checks["cache_system"] = set_success and get_result is not None
            
            # Check router
            if self.consensus_router:
                # Test routing operation
                try:
                    test_query = "health check query"
                    routing_result = await self.consensus_router.route_optimized_query(
                        query=test_query,
                        max_experts=3
                    )
                    health_checks["router"] = len(routing_result.get("selected_models", [])) > 0
                except:
                    health_checks["router"] = False
            
            # Overall health
            overall_health = all(health_checks.values())
            
            logger.info(f"Health check results: {health_checks}")
            return overall_health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def _test_core_system(self) -> bool:
        """Test core system functionality."""
        
        try:
            test_query = "Test query for health check"
            
            if hasattr(self.optimized_system, 'process_optimized_query'):
                result = await self.optimized_system.process_optimized_query(test_query)
            else:
                result = await self.optimized_system.process_query(test_query)
            
            return result.response is not None and len(result.response) > 0
            
        except Exception as e:
            logger.warning(f"Core system test failed: {e}")
            return False
    
    async def process_query(
        self,
        query: str,
        context: Optional[QueryContext] = None,
        consensus_mode: Optional[ConsensusMode] = None,
        optimization_level: Optional[OptimizationLevel] = None
    ) -> ConsensusResult:
        """
        Process query through the integrated meta-consensus-comp system.
        
        Args:
            query: Input query
            context: Query context
            consensus_mode: Consensus mode to use
            optimization_level: Optimization level override
            
        Returns:
            Comprehensive consensus result
        """
        
        start_time = time.time()
        
        try:
            # Use default modes if not specified
            consensus_mode = consensus_mode or self.config.consensus_mode
            optimization_level = optimization_level or self.config.optimization_level
            
            # Route through optimized system
            if self.optimized_system and hasattr(self.optimized_system, 'process_optimized_query'):
                result = await self.optimized_system.process_optimized_query(
                    query=query,
                    context=context,
                    consensus_mode=consensus_mode,
                    optimization_level=optimization_level
                )
            else:
                # Fallback to base system
                result = await self.optimized_system.process_query(query, context, consensus_mode)
            
            # Update monitoring
            if self.monitoring_system:
                await self.monitoring_system.monitor_consensus_query(
                    result.query_id, {
                        "response_time_ms": result.response_time_ms,
                        "quality_score": result.quality_score,
                        "total_cost": result.total_cost,
                        "confidence": result.confidence,
                        "participating_experts": result.participating_experts
                    }
                )
            
            # Update system status
            processing_time = time.time() - start_time
            await self._update_system_status(result, processing_time)
            
            # Track query history
            self.query_history.append({
                "timestamp": datetime.now(),
                "query": query,
                "result": result,
                "processing_time_ms": processing_time * 1000,
                "optimization_level": optimization_level.value,
                "consensus_mode": consensus_mode.value
            })
            
            # Limit history size
            if len(self.query_history) > 1000:
                self.query_history = self.query_history[-1000:]
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # Update error tracking
            self.status.error_rate = min(self.status.error_rate + 0.01, 1.0)
            self.status.last_error = str(e)
            
            # Return error result
            return ConsensusResult(
                query_id=f"error_{int(time.time())}",
                response=f"Error processing query: {str(e)}",
                confidence=0.0,
                quality_score=0.0,
                participating_experts=[],
                expert_responses=[],
                routing_decision={"error": str(e)},
                consensus_method="error_fallback",
                response_time_ms=(time.time() - start_time) * 1000,
                total_cost=0.0,
                tokens_generated=0
            )
    
    async def _update_system_status(self, result: ConsensusResult, processing_time: float):
        """Update system status based on query result."""
        
        # Update performance metrics
        if self.status.avg_latency_ms == 0:
            self.status.avg_latency_ms = processing_time * 1000
        else:
            self.status.avg_latency_ms = self.status.avg_latency_ms * 0.9 + (processing_time * 1000) * 0.1
        
        if self.status.avg_quality_score == 0:
            self.status.avg_quality_score = result.quality_score
        else:
            self.status.avg_quality_score = self.status.avg_quality_score * 0.9 + result.quality_score * 0.1
        
        # Update QPS (simplified calculation)
        self.status.current_qps = 1.0 / processing_time if processing_time > 0 else 0
        
        # Update resource usage
        try:
            process = psutil.Process()
            self.status.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
            self.status.cpu_usage_percent = process.cpu_percent()
        except:
            pass
        
        # Update cache metrics
        if self.cache_system:
            cache_stats = self.cache_system.get_comprehensive_stats()
            self.status.cache_hit_rate = cache_stats["performance_metrics"]["overall_hit_rate"]
            self.status.cache_size_mb = cache_stats["storage_metrics"]["total_size_mb"]
        
        # Update expert metrics
        self.status.active_experts = len(result.participating_experts)
        
        # Update uptime
        self.status.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
    
    async def run_system_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive system benchmark."""
        
        if not self.benchmark_system:
            logger.warning("Benchmarking not enabled")
            return {}
        
        logger.info("📊 Running system benchmark...")
        
        report = await self.benchmark_system.run_comprehensive_benchmark()
        
        benchmark_summary = {
            "timestamp": datetime.now().isoformat(),
            "system_profile": self.config.profile.value,
            "optimization_level": self.config.optimization_level.value,
            "latency_improvements": report.latency_improvement,
            "throughput_improvements": report.throughput_improvement,
            "memory_efficiency_gains": report.memory_efficiency_gain,
            "quality_preservation": report.quality_preservation_rate,
            "recommendations": report.optimization_recommendations
        }
        
        return benchmark_summary
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return {
            "system_info": {
                "name": self.config.system_name,
                "version": self.config.version,
                "profile": self.config.profile.value,
                "uptime_seconds": self.status.uptime_seconds,
                "is_healthy": self.status.is_healthy
            },
            "performance": {
                "current_qps": round(self.status.current_qps, 2),
                "avg_latency_ms": round(self.status.avg_latency_ms, 2),
                "avg_quality_score": round(self.status.avg_quality_score, 2),
                "error_rate": f"{self.status.error_rate:.1%}"
            },
            "resources": {
                "memory_usage_mb": round(self.status.memory_usage_mb, 2),
                "cpu_usage_percent": round(self.status.cpu_usage_percent, 2),
                "gpu_usage_percent": round(self.status.gpu_usage_percent, 2)
            },
            "cache": {
                "hit_rate": f"{self.status.cache_hit_rate:.1%}",
                "size_mb": round(self.status.cache_size_mb, 2)
            },
            "experts": {
                "active_experts": self.status.active_experts,
                "utilization_rate": f"{self.status.expert_utilization_rate:.1%}"
            },
            "configuration": {
                "optimization_level": self.config.optimization_level.value,
                "routing_mode": self.config.routing_mode.value,
                "tpu_v6_enabled": self.config.enable_tpu_v6,
                "monitoring_enabled": self.config.enable_monitoring,
                "caching_enabled": self.config.enable_caching
            },
            "recent_queries": len([q for q in self.query_history 
                                 if (datetime.now() - q["timestamp"]).total_seconds() < 3600])
        }
    
    async def start_monitoring_server(self):
        """Start the monitoring dashboard server."""
        
        if self.monitoring_system:
            logger.info(f"🖥️ Starting monitoring server on port {self.config.monitoring_port}")
            await self.monitoring_system.start_monitoring()
        else:
            logger.warning("Monitoring system not initialized")
    
    def get_monitoring_url(self) -> Optional[str]:
        """Get monitoring dashboard URL."""
        
        if self.monitoring_system:
            return f"http://{self.config.api_host}:{self.config.monitoring_port}"
        return None
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        
        logger.info("🛑 Shutting down Meta-Consensus-Comp System...")
        
        # Save final metrics
        if self.query_history:
            await self._save_session_metrics()
        
        # Cleanup memory pools
        if self.memory_manager:
            self.memory_manager.cleanup_all_pools()
        
        # Cleanup cache
        if self.cache_system:
            await self.cache_system.cleanup_all_levels()
        
        self.status.is_healthy = False
        logger.info("✅ System shutdown completed")
    
    async def _save_session_metrics(self):
        """Save session metrics to file."""
        
        session_data = {
            "session_start": self.start_time.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_queries": len(self.query_history),
            "system_config": {
                "profile": self.config.profile.value,
                "optimization_level": self.config.optimization_level.value,
                "routing_mode": self.config.routing_mode.value
            },
            "final_status": self.get_system_status(),
            "query_summary": {
                "avg_processing_time_ms": np.mean([q["processing_time_ms"] for q in self.query_history]),
                "avg_quality_score": np.mean([q["result"].quality_score for q in self.query_history]),
                "total_cost": sum([q["result"].total_cost for q in self.query_history])
            }
        }
        
        session_file = Path("sessions") / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        session_file.parent.mkdir(exist_ok=True)
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        logger.info(f"Session metrics saved to {session_file}")


# Factory functions
def create_integrated_system(
    profile: SystemProfile = SystemProfile.PRODUCTION_BALANCED,
    hf_api_token: str = "",
    **kwargs
) -> MetaConsensusCompSystem:
    """Creates integrated meta-consensus-comp system with profile."""
    
    config = IntegratedSystemConfig(
        profile=profile,
        hf_api_token=hf_api_token,
        **kwargs
    )
    
    return MetaConsensusCompSystem(config)

def create_development_system(hf_api_token: str = "") -> MetaConsensusCompSystem:
    """Creates development system with optimized settings."""
    
    return create_integrated_system(
        profile=SystemProfile.DEVELOPMENT,
        hf_api_token=hf_api_token,
        max_memory_mb=512,
        max_concurrent_queries=10
    )

def create_production_system(hf_api_token: str = "", performance_mode: bool = False) -> MetaConsensusCompSystem:
    """Creates production system."""
    
    profile = SystemProfile.PRODUCTION_PERFORMANCE if performance_mode else SystemProfile.PRODUCTION_BALANCED
    
    return create_integrated_system(
        profile=profile,
        hf_api_token=hf_api_token,
        max_memory_mb=4096 if performance_mode else 2048,
        max_concurrent_queries=100 if performance_mode else 50
    )

def create_tpu_v6_system(hf_api_token: str = "") -> MetaConsensusCompSystem:
    """Creates TPU v6 optimized system."""
    
    return create_integrated_system(
        profile=SystemProfile.TPU_V6_MAXIMUM,
        hf_api_token=hf_api_token,
        enable_tpu_v6=True,
        max_memory_mb=8192,
        max_concurrent_queries=200
    )


# Export main components
__all__ = [
    'MetaConsensusCompSystem',
    'IntegratedSystemConfig',
    'SystemProfile',
    'SystemStatus',
    'create_integrated_system',
    'create_development_system',
    'create_production_system',
    'create_tpu_v6_system'
]


if __name__ == "__main__":
    # Example usage and demonstration
    async def main():
        print("🎭 Meta-Consensus-Comp Integrated System Demo")
        print("=" * 50)
        
        # Create development system
        system = create_development_system(hf_api_token="demo_token")
        
        # Initialize system
        if await system.initialize():
            print("✅ System initialized successfully")
            
            # Get system status
            status = system.get_system_status()
            print(f"\n📊 System Status:")
            print(f"  Profile: {status['system_info']['profile']}")
            print(f"  Optimization: {status['configuration']['optimization_level']}")
            print(f"  Health: {'✅ Healthy' if status['system_info']['is_healthy'] else '❌ Unhealthy'}")
            
            # Test queries
            test_queries = [
                "What is artificial intelligence?",
                "Explain machine learning algorithms",
                "How does natural language processing work?",
                "Describe quantum computing applications",
                "What are the benefits of distributed systems?"
            ]
            
            print(f"\n🔍 Processing {len(test_queries)} test queries...")
            
            results = []
            for i, query in enumerate(test_queries):
                print(f"\nQuery {i+1}: {query}")
                
                result = await system.process_query(query)
                results.append(result)
                
                print(f"  Response: {result.response[:100]}...")
                print(f"  Quality: {result.quality_score:.1f}/10")
                print(f"  Confidence: {result.confidence:.2f}")
                print(f"  Cost: ${result.total_cost:.4f}")
                print(f"  Time: {result.response_time_ms:.0f}ms")
                print(f"  Experts: {len(result.participating_experts)}")
            
            # Run benchmark
            if system.benchmark_system:
                print(f"\n📊 Running system benchmark...")
                
                benchmark_results = await system.run_system_benchmark()
                
                if benchmark_results:
                    print(f"✅ Benchmark completed:")
                    print(f"  Profile: {benchmark_results['system_profile']}")
                    
                    if benchmark_results.get('latency_improvements'):
                        print(f"  Latency Improvements:")
                        for level, improvement in benchmark_results['latency_improvements'].items():
                            print(f"    {level}: {improvement:.1%}")
                    
                    if benchmark_results.get('recommendations'):
                        print(f"  Recommendations:")
                        for rec in benchmark_results['recommendations'][:3]:
                            print(f"    • {rec}")
            
            # Final system status
            final_status = system.get_system_status()
            print(f"\n📈 Final Performance:")
            print(f"  Queries Processed: {final_status['recent_queries']}")
            print(f"  Avg Latency: {final_status['performance']['avg_latency_ms']:.0f}ms")
            print(f"  Avg Quality: {final_status['performance']['avg_quality_score']:.1f}")
            print(f"  Cache Hit Rate: {final_status['cache']['hit_rate']}")
            print(f"  Memory Usage: {final_status['resources']['memory_usage_mb']:.0f}MB")
            
            # Get monitoring URL
            monitoring_url = system.get_monitoring_url()
            if monitoring_url:
                print(f"\n🖥️ Monitoring Dashboard: {monitoring_url}")
            
            # Shutdown
            await system.shutdown()
            
        else:
            print("❌ System initialization failed")
    
    import asyncio
    asyncio.run(main())