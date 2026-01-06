"""
Intelligent VQ Integration System
Advanced integration layer that unifies all VQ capabilities in CapibaraGPT
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any, Union, Callable
from functools import partial
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct

# Import our enhanced VQ system
from .enhanced_vq_system_v2 import (
    EnhancedVectorQuantizer,
    EnhancedVQConfig,
    create_enhanced_vq_layer,
    benchmark_vq_variants
)

logger = logging.getLogger(__name__)

# ============================================================================
# Intelligent VQ Manager
# ============================================================================

@dataclass
class VQManagerConfig:
    """Configurestion for the VQ Manager."""
    
    # System selection
    auto_select_vq: bool = True
    default_vq_system: str = "enhanced"  # "enhanced", "legacy", "adaptive"
    
    # Performance thresholds for auto-selection
    max_latency_ms: float = 10.0
    min_perplexity: float = 50.0
    min_usage_rate: float = 0.1
    
    # Integration settings
    enable_fallback: bool = True
    enable_monitoring: bool = True
    enable_benchmarking: bool = True
    
    # Caching and optimization
    cache_configurations: bool = True
    enable_jit_compilation: bool = True
    
    # Experiment tracking
    log_performance: bool = True
    experiment_tracking: bool = False

class IntelligentVQManager:
    """
    Intelligent VQ Manager that automatically selects and optimizes VQ configurations
    based on use case, performance requirements, and hardware constraints.
    """
    
    def __init__(self, config: VQManagerConfig):
        self.config = config
        self.performance_cache = {}
        self.configuration_cache = {}
        self.usage_stats = {
            'enhanced': 0,
            'legacy': 0,
            'product': 0,
            'adaptive': 0
        }
        
        # Initialize available VQ systems
        self._initialize_vq_systems()
    
    def _initialize_vq_systems(self):
        """Initialize all available VQ systems."""
        self.vq_systems = {
            'enhanced_standard': {
                'factory': create_enhanced_vq_layer,
                'config': {'use_ema': True, 'use_adaptive_commitment': False},
                'description': 'Enhanced standard VQ with EMA updates',
                'use_cases': ['general', 'fast_inference', 'stable_training']
            },
            'enhanced_adaptive': {
                'factory': create_enhanced_vq_layer,
                'config': {
                    'use_ema': True, 
                    'use_adaptive_commitment': True,
                    'use_entropy_loss': True,
                    'use_diversity_loss': True
                },
                'description': 'Adaptive VQ with entropy and diversity regularization',
                'use_cases': ['research', 'high_quality', 'codebook_optimization']
            },
            'product_quantization': {
                'factory': create_enhanced_vq_layer,
                'config': {
                    'use_product_quantization': True,
                    'num_subspaces': 8,
                    'use_ema': True
                },
                'description': 'Product quantization for memory efficiency',
                'use_cases': ['large_scale', 'memory_constrained', 'compression']
            },
            'gumbel_differentiable': {
                'factory': create_enhanced_vq_layer,
                'config': {
                    'use_gumbel': True,
                    'temperature': 1.0,
                    'use_ema': True
                },
                'description': 'Differentiable VQ with Gumbel softmax',
                'use_cases': ['end_to_end_training', 'gradient_flow', 'research']
            },
            'compression_optimized': {
                'factory': create_enhanced_vq_layer,
                'config': {
                    'use_compression': True,
                    'compression_ratio': 0.5,
                    'use_ema': True
                },
                'description': 'VQ with neural compression layers',
                'use_cases': ['bandwidth_limited', 'storage_optimization']
            }
        }
    
    def auto_select_vq(self, 
                      use_case: str,
                      num_embeddings: int = 8192,
                      embedding_dim: int = 1024,
                      performance_requirements: Optional[Dict[str, float]] = None,
                      **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Automatically select the best VQ configuration for the given use case.
        
        Args:
            use_case: Use case identifier ('general', 'research', 'large_scale', etc.)
            num_embeddings: Number of embeddings in codebook
            embedding_dim: Embedding dimension
            performance_requirements: Dict with 'max_latency_ms', 'min_perplexity', etc.
            **kwargs: Additional configuration parameters
            
        Returns:
            Tuple of (vq_module, selection_info)
        """
        if not self.config.auto_select_vq:
            # Use default system
            return self._create_vq_system(
                self.config.default_vq_system,
                num_embeddings,
                embedding_dim,
                **kwargs
            )
        
        # Merge performance requirements
        perf_reqs = {
            'max_latency_ms': self.config.max_latency_ms,
            'min_perplexity': self.config.min_perplexity,
            'min_usage_rate': self.config.min_usage_rate,
        }
        if performance_requirements:
            perf_reqs.update(performance_requirements)
        
        # Find matching VQ systems
        candidates = self._find_candidate_systems(use_case, perf_reqs)
        
        if not candidates:
            logger.warning(f"No VQ systems found for use case '{use_case}', using default")
            return self._create_vq_system(
                'enhanced_standard',
                num_embeddings,
                embedding_dim,
                **kwargs
            )
        
        # Benchmark candidates if enabled
        if self.config.enable_benchmarking and len(candidates) > 1:
            best_system = self._benchmark_candidates(
                candidates, 
                num_embeddings, 
                embedding_dim, 
                perf_reqs,
                **kwargs
            )
        else:
            best_system = candidates[0]
        
        # Create and return the selected system
        vq_module, creation_info = self._create_vq_system(
            best_system,
            num_embeddings,
            embedding_dim,
            **kwargs
        )
        
        # Track usage
        self.usage_stats[best_system] += 1
        
        selection_info = {
            'selected_system': best_system,
            'candidates': candidates,
            'use_case': use_case,
            'performance_requirements': perf_reqs,
            **creation_info
        }
        
        return vq_module, selection_info
    
    def _find_candidate_systems(self, use_case: str, perf_reqs: Dict[str, float]) -> List[str]:
        """Find VQ systems that match the use case and performance requirements."""
        candidates = []
        
        for system_name, system_info in self.vq_systems.items():
            # Check if use case matches
            if use_case in system_info['use_cases'] or 'general' in system_info['use_cases']:
                candidates.append(system_name)
        
        # Apply performance-based filtering
        if perf_reqs['max_latency_ms'] < 5.0:
            # Remove slow systems for low-latency requirements
            candidates = [c for c in candidates if 'product_quantization' not in c]
        
        if perf_reqs.get('memory_constrained', False):
            # Prioritize memory-efficient systems
            if 'product_quantization' in candidates:
                candidates = ['product_quantization'] + [c for c in candidates if c != 'product_quantization']
        
        return candidates or ['enhanced_standard']  # Always have a fallback
    
    def _benchmark_candidates(self, 
                            candidates: List[str],
                            num_embeddings: int,
                            embedding_dim: int,
                            perf_reqs: Dict[str, float],
                            **kwargs) -> str:
        """Benchmark candidate systems and select the best one."""
        import time
        
        # Check cache first
        cache_key = (tuple(candidates), num_embeddings, embedding_dim, tuple(sorted(kwargs.items())))
        if cache_key in self.performance_cache:
            return self.performance_cache[cache_key]
        
        # Create test data
        key = jax.random.PRNGKey(42)
        test_data = jax.random.normal(key, (2, 16, embedding_dim))
        
        best_system = candidates[0]
        best_score = float('-inf')
        benchmark_results = {}
        
        for system_name in candidates:
            try:
                # Create VQ system
                vq_module, _ = self._create_vq_system(
                    system_name, 
                    num_embeddings, 
                    embedding_dim,
                    **kwargs
                )
                
                # Initialize
                params = vq_module.init(key, test_data)
                
                # Warmup
                _ = vq_module.apply(params, test_data, training=True)
                
                # Benchmark performance
                start_time = time.time()
                quantized, metrics = vq_module.apply(params, test_data, training=True)
                latency_ms = (time.time() - start_time) * 1000
                
                # Compute score based on requirements
                score = self._compute_performance_score(
                    latency_ms, 
                    metrics, 
                    perf_reqs
                )
                
                benchmark_results[system_name] = {
                    'latency_ms': latency_ms,
                    'metrics': metrics,
                    'score': score
                }
                
                if score > best_score:
                    best_score = score
                    best_system = system_name
                
                logger.info(f"Benchmarked {system_name}: latency={latency_ms:.2f}ms, score={score:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to benchmark {system_name}: {e}")
                benchmark_results[system_name] = {'error': str(e)}
        
        # Cache result
        if self.config.cache_configurations:
            self.performance_cache[cache_key] = best_system
        
        return best_system
    
    def _compute_performance_score(self, 
                                 latency_ms: float,
                                 metrics: Dict[str, Any],
                                 perf_reqs: Dict[str, float]) -> float:
        """Compute a performance score for ranking VQ systems."""
        score = 0.0
        
        # Latency component (lower is better)
        if latency_ms <= perf_reqs['max_latency_ms']:
            score += 1.0 - (latency_ms / perf_reqs['max_latency_ms'])
        else:
            score -= (latency_ms - perf_reqs['max_latency_ms']) / perf_reqs['max_latency_ms']
        
        # Perplexity component (higher is better)
        perplexity = metrics.get('perplexity', 0)
        if perplexity >= perf_reqs['min_perplexity']:
            score += min(perplexity / 100.0, 1.0)  # Cap at 1.0
        else:
            score -= (perf_reqs['min_perplexity'] - perplexity) / perf_reqs['min_perplexity']
        
        # Usage rate component (higher is better)
        usage_rate = metrics.get('usage_rate', 0)
        if usage_rate >= perf_reqs['min_usage_rate']:
            score += usage_rate
        else:
            score -= (perf_reqs['min_usage_rate'] - usage_rate) * 2
        
        # MSE component (lower is better)
        mse = metrics.get('mse', 1.0)
        score += max(0, 1.0 - mse * 10)  # Penalize high MSE
        
        return score
    
    def _create_vq_system(self, 
                         system_name: str,
                         num_embeddings: int,
                         embedding_dim: int,
                         **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
        """Creates a VQ system instance."""
        if system_name not in self.vq_systems:
            raise ValueError(f"Unknown VQ system: {system_name}")
        
        system_info = self.vq_systems[system_name]
        
        # Merge configurations
        config = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            **system_info['config'],
            **kwargs
        }
        
        # Add experiment tracking if enabled
        if self.config.experiment_tracking:
            config['experiment_name'] = f"{system_name}_{num_embeddings}_{embedding_dim}"
            config['enable_monitoring'] = True
        
        # Create VQ module
        vq_module = system_info['factory'](**config)
        
        creation_info = {
            'system_name': system_name,
            'description': system_info['description'],
            'config': config,
            'use_cases': system_info['use_cases']
        }
        
        return vq_module, creation_info
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for all VQ systems."""
        total_usage = sum(self.usage_stats.values())
        
        stats = {
            'total_creations': total_usage,
            'system_usage': self.usage_stats.copy(),
            'system_percentages': {
                name: (count / total_usage * 100) if total_usage > 0 else 0
                for name, count in self.usage_stats.items()
            },
            'available_systems': list(self.vq_systems.keys()),
            'cache_size': len(self.performance_cache)
        }
        
        return stats
    
    def get_recommended_config(self, use_case: str) -> Dict[str, Any]:
        """Get recommended configuration for a use case without creating the system."""
        candidates = self._find_candidate_systems(use_case, {
            'max_latency_ms': self.config.max_latency_ms,
            'min_perplexity': self.config.min_perplexity,
            'min_usage_rate': self.config.min_usage_rate
        })
        
        recommended_system = candidates[0] if candidates else 'enhanced_standard'
        
        return {
            'recommended_system': recommended_system,
            'candidates': candidates,
            'system_info': self.vq_systems.get(recommended_system, {}),
            'use_case': use_case
        }
    
    def benchmark_all_systems(self, 
                            num_embeddings: int = 1024,
                            embedding_dim: int = 256) -> Dict[str, Any]:
        """Benchmark all available VQ systems."""
        return benchmark_vq_variants()

# ============================================================================
# High-level API for Easy Usage
# ============================================================================

# Global VQ manager instance
_global_vq_manager = None

def get_vq_manager(config: Optional[VQManagerConfig] = None) -> IntelligentVQManager:
    """Get the global VQ manager instance."""
    global _global_vq_manager
    
    if _global_vq_manager is None or config is not None:
        if config is None:
            config = VQManagerConfig()
        _global_vq_manager = IntelligentVQManager(config)
    
    return _global_vq_manager

def create_optimal_vq(use_case: str = "general",
                     num_embeddings: int = 8192,
                     embedding_dim: int = 1024,
                     performance_requirements: Optional[Dict[str, float]] = None,
                     **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Create an optimal VQ system for the given use case.
    
    Args:
        use_case: One of 'general', 'research', 'large_scale', 'memory_constrained',
                 'fast_inference', 'high_quality', 'compression', 'end_to_end_training'
        num_embeddings: Number of embeddings in codebook
        embedding_dim: Embedding dimension
        performance_requirements: Dict with performance constraints
        **kwargs: Additional VQ configuration parameters
        
    Returns:
        Tuple of (vq_module, selection_info)
        
    Examples:
        # General purpose VQ
        vq, info = create_optimal_vq("general", 4096, 512)
        
        # Research with high quality requirements
        vq, info = create_optimal_vq(
            "research", 
            8192, 
            1024,
            performance_requirements={'min_perplexity': 100.0}
        )
        
        # Memory-constrained deployment
        vq, info = create_optimal_vq("memory_constrained", 2048, 256)
    """
    manager = get_vq_manager()
    return manager.auto_select_vq(
        use_case=use_case,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        performance_requirements=performance_requirements,
        **kwargs
    )

def get_vq_recommendations(use_case: str) -> Dict[str, Any]:
    """Get VQ system recommendations for a use case."""
    manager = get_vq_manager()
    return manager.get_recommended_config(use_case)

def get_vq_usage_stats() -> Dict[str, Any]:
    """Get usage statistics for VQ systems."""
    manager = get_vq_manager()
    return manager.get_usage_statistics()

# ============================================================================
# Integration with Existing CapibaraGPT Systems
# ============================================================================

class CapibaraVQIntegration:
    """Integration layer for CapibaraGPT's existing VQ systems."""
    
    def __init__(self):
        self.manager = get_vq_manager()
        self.legacy_systems = self._load_legacy_systems()
    
    def _load_legacy_systems(self) -> Dict[str, Any]:
        """Load existing VQ systems from CapibaraGPT."""
        legacy = {}
        
        try:
            # Try to import existing VQ components
            from . import ultra_vq_orchestrator
            legacy['ultra_orchestrator'] = ultra_vq_orchestrator
        except ImportError:
            pass
        
        try:
            from . import vqbit_layer
            legacy['vqbit'] = vqbit_layer
        except ImportError:
            pass
        
        try:
            from . import multi_modal_vq_intelligence
            legacy['multimodal'] = multi_modal_vq_intelligence
        except ImportError:
            pass
        
        return legacy
    
    def create_unified_vq(self, 
                         use_case: str = "general",
                         prefer_legacy: bool = False,
                         **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Create a unified VQ system that can fall back to legacy systems.
        
        Args:
            use_case: VQ use case
            prefer_legacy: Whether to prefer legacy systems
            **kwargs: VQ configuration parameters
            
        Returns:
            Tuple of (vq_module, system_info)
        """
        if prefer_legacy and self.legacy_systems:
            # Try to use legacy systems first
            logger.info("Using legacy VQ system as requested")
            # Implementation would depend on legacy system APIs
            pass
        
        # Use enhanced VQ system
        return create_optimal_vq(use_case=use_case, **kwargs)
    
    def migrate_legacy_config(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy VQ configuration to enhanced system."""
        enhanced_config = {}
        
        # Map common parameters
        mapping = {
            'codebook_size': 'num_embeddings',
            'embedding_size': 'embedding_dim',
            'beta': 'commitment_cost',
            'use_exponential_moving_average': 'use_ema',
            'ema_decay_rate': 'ema_decay'
        }
        
        for legacy_key, enhanced_key in mapping.items():
            if legacy_key in legacy_config:
                enhanced_config[enhanced_key] = legacy_config[legacy_key]
        
        return enhanced_config

# ============================================================================
# Usage Examples and Demonstrations
# ============================================================================

def demo_intelligent_integration():
    """Demonstrate the intelligent VQ integration system."""
    print("🧠 Intelligent VQ Integration Demo")
    print("=" * 50)
    
    # 1. Auto-selection for different use cases
    print("\n1️⃣ Auto-selection for different use cases:")
    
    use_cases = [
        ("general", "General purpose AI model"),
        ("research", "Research with high quality requirements"),
        ("memory_constrained", "Mobile/edge deployment"),
        ("large_scale", "Large language model training"),
        ("fast_inference", "Real-time inference"),
    ]
    
    for use_case, description in use_cases:
        print(f"\n📋 {use_case.upper()}: {description}")
        
        try:
            vq, info = create_optimal_vq(
                use_case=use_case,
                num_embeddings=1024,
                embedding_dim=256
            )
            
            print(f"   ✅ Selected: {info['selected_system']}")
            print(f"   📝 Description: {info['description']}")
            print(f"   🎯 Candidates: {', '.join(info['candidates'])}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # 2. Performance requirements
    print(f"\n2️⃣ Performance-constrained selection:")
    
    vq, info = create_optimal_vq(
        use_case="general",
        num_embeddings=2048,
        embedding_dim=512,
        performance_requirements={
            'max_latency_ms': 2.0,  # Very strict latency
            'min_perplexity': 80.0
        }
    )
    
    print(f"   ✅ Low-latency system: {info['selected_system']}")
    
    # 3. Usage statistics
    print(f"\n3️⃣ Usage statistics:")
    stats = get_vq_usage_stats()
    print(f"   📊 Total VQ creations: {stats['total_creations']}")
    print(f"   📈 System usage:")
    for system, percentage in stats['system_percentages'].items():
        print(f"      {system}: {percentage:.1f}%")
    
    # 4. Recommendations
    print(f"\n4️⃣ System recommendations:")
    for use_case, _ in use_cases[:3]:
        rec = get_vq_recommendations(use_case)
        print(f"   {use_case}: {rec['recommended_system']}")
    
    print(f"\n✅ Intelligent VQ Integration Demo completed!")

if __name__ == "__main__":
    demo_intelligent_integration()