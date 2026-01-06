"""
VQBit Module for Capibara-6

Advanced Vector Quantization implementations optimized for different hardware backends:
- TPU v6e-64 optimized VQ with BF16 and mesh sharding
- ARM Axion optimized VQ with NEON vectorization
- Multimodal VQ for cross-modal applications
- Adaptive VQ with dynamic optimization
- Integration with MoE and Adaptive systems
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import TPU v6e VQ system
try:
    from .vq_v33_tpu_v6 import (
        TPUv6eVectorQuantizer,
        TPUv6eVQConfig,
        TPUv6eVQSystem,
        create_tpu_v6e_vq_system,
        create_tpu_v6e_vq_layer,
        integrate_with_moe,
        integrate_with_adaptive
    )
    TPU_V6E_VQ_AVAILABLE = True
except ImportError as e:
    logging.warning(f"TPU v6e VQ not available: {e}")
    TPUv6eVectorQuantizer = None
    TPUv6eVQConfig = None
    TPUv6eVQSystem = None
    create_tpu_v6e_vq_system = None
    create_tpu_v6e_vq_layer = None
    integrate_with_moe = None
    integrate_with_adaptive = None
    TPU_V6E_VQ_AVAILABLE = False

# Import ARM Axion VQ system
try:
    from .vq_arm_axion import (
        ARMAxionConfig,
        ARMAxionVQOptimizer
    )
    ARM_AXION_VQ_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ARM Axion VQ not available: {e}")
    ARMAxionConfig = None
    ARMAxionVQOptimizer = None
    ARM_AXION_VQ_AVAILABLE = False

# Import VQbit Layer
try:
    from .vqbit_layer import (
        VQbitLayer,
        VQbitConfig,
        create_vqbit_layer,
        VQBitLayer  # Alias
    )
    VQBIT_LAYER_AVAILABLE = True
except ImportError:
    VQBIT_LAYER_AVAILABLE = False
    VQbitLayer = None
    VQbitConfig = None
    create_vqbit_layer = None
    VQBitLayer = None

# Import Adaptive VQ components
try:
    from .adaptive_vq import (
        AdaptiveVectorQuantizer,
        AdaptiveVQConfig,
        AdaptationHistory,
        create_adaptive_vq,
        create_performance_adaptive_vq,
        AdaptiveVQ  # Alias
    )
    ADAPTIVE_VQ_AVAILABLE = True
except ImportError:
    ADAPTIVE_VQ_AVAILABLE = False
    AdaptiveVectorQuantizer = None
    AdaptiveVQConfig = None
    AdaptationHistory = None
    create_adaptive_vq = None
    create_performance_adaptive_vq = None
    AdaptiveVQ = None

# Import Multimodal VQbit components
try:
    from .multimodal_vqbit import (
        VQbitModule,
        MultimodalVQConfig,
        create_multimodal_vqbit
    )
    MULTIMODAL_VQ_AVAILABLE = True
except ImportError:
    MULTIMODAL_VQ_AVAILABLE = False
    VQbitModule = None
    MultimodalVQConfig = None
    create_multimodal_vqbit = None

# Import VQ Monitoring components
try:
    from .vq_monitoring import (
        VQMonitor,
        MonitoringConfig,
        MetricsCollector,
        PerformanceAnalyzer,
        create_vq_monitor,
        create_production_monitor,
        get_global_monitor,
        set_global_monitor,
        monitor_vq_operation
    )
    VQ_MONITORING_AVAILABLE = True
except ImportError:
    VQ_MONITORING_AVAILABLE = False
    VQMonitor = None
    MonitoringConfig = None
    MetricsCollector = None
    PerformanceAnalyzer = None
    create_vq_monitor = None
    create_production_monitor = None
    get_global_monitor = None
    set_global_monitor = None
    monitor_vq_operation = None

# Import Wrapper components
try:
    from .wrapper import (
        AdaptiveWrapper,
        AdaptiveWrapperConfig,
        PerformanceTracker,
        SimpleVQFallback,
        create_adaptive_wrapper,
        create_performance_optimized_wrapper
    )
    VQ_WRAPPER_AVAILABLE = True
except ImportError:
    VQ_WRAPPER_AVAILABLE = False
    AdaptiveWrapper = None
    AdaptiveWrapperConfig = None
    PerformanceTracker = None
    SimpleVQFallback = None
    create_adaptive_wrapper = None
    create_performance_optimized_wrapper = None

# Import JAX Native Integration
try:
    from .jax_native_integration import (
        VQbitNativeIntegration,
        NativeIntegrationConfig,
        get_native_integration,
        create_native_vqbit_layer,
        create_native_multimodal_vqbit,
        get_native_backend_info,
        benchmark_native_backends,
        integrate_with_tpu_backend
    )
    JAX_NATIVE_INTEGRATION_AVAILABLE = True
except ImportError:
    JAX_NATIVE_INTEGRATION_AVAILABLE = False
    VQbitNativeIntegration = None
    NativeIntegrationConfig = None
    get_native_integration = None
    create_native_vqbit_layer = None
    create_native_multimodal_vqbit = None
    get_native_backend_info = None
    benchmark_native_backends = None
    integrate_with_tpu_backend = None

logger = logging.getLogger(__name__)

def get_project_root():
    """Get the root path of the project."""
    return Path(__file__).parent.parent.parent.parent

# Version information
__version__ = "3.3.0"  # Updated for v3.3 with TPU v6e support
__author__ = "CapibaraGPT Team"


class VQBitSystem:
    """
    Unified VQBit system that provides access to all VQ implementations
    with automatic backend selection and optimization.
    """
    
    def __init__(self):
        self.available_systems = self._detect_available_systems()
        self.performance_cache = {}
        
        logger.info(f"🔧 VQBit System initialized with {len(self.available_systems)} backends")
        
    def _detect_available_systems(self) -> Dict[str, bool]:
        """Detect available VQ systems."""
        return {
            'tpu_v6e': TPU_V6E_VQ_AVAILABLE,
            'arm_axion': ARM_AXION_VQ_AVAILABLE,
            'vqbit_layer': VQBIT_LAYER_AVAILABLE,
            'adaptive': ADAPTIVE_VQ_AVAILABLE,
            'multimodal': MULTIMODAL_VQ_AVAILABLE,
            'monitoring': VQ_MONITORING_AVAILABLE,
            'wrapper': VQ_WRAPPER_AVAILABLE,
            'jax_native_integration': JAX_NATIVE_INTEGRATION_AVAILABLE
        }
    
    def create_vq_for_backend(self, 
                             backend: str,
                             config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create VQ system for specific backend.
        
        Args:
            backend: Backend type ('tpu_v6e', 'arm_axion', 'adaptive', etc.)
            config: Configuration dictionary
            
        Returns:
            VQ system instance
        """
        
        config = config or {}
        
        if backend == 'tpu_v6e' and TPU_V6E_VQ_AVAILABLE:
            vq_config = TPUv6eVQConfig(**config)
            return create_tpu_v6e_vq_system(vq_config)
            
        elif backend == 'arm_axion' and ARM_AXION_VQ_AVAILABLE:
            arm_config = ARMAxionConfig(**config)
            return ARMAxionVQOptimizer(arm_config)
            
        elif backend == 'native_jax' and JAX_NATIVE_INTEGRATION_AVAILABLE:
            return create_native_vqbit_layer(
                codebook_size=config.get('codebook_size', 64),
                embedding_dim=config.get('embedding_dim', 768),
                **config
            )
            
        elif backend == 'vqbit_layer' and VQBIT_LAYER_AVAILABLE:
            vqbit_config = VQbitConfig(
                codebook_size=config.get('codebook_size', 64),
                embedding_dim=config.get('embedding_dim', 768),
                **config
            )
            return VQbitLayer(vqbit_config)
            
        else:
            raise ValueError(f"Backend '{backend}' not available or not supported")
    
    def get_optimal_backend(self, 
                           use_case: str = "general",
                           constraints: Optional[Dict[str, Any]] = None) -> str:
        """
        Get optimal backend for use case and constraints.
        
        Args:
            use_case: Use case ('research', 'production', 'memory_constrained')
            constraints: Hardware/performance constraints
            
        Returns:
            Optimal backend name
        """
        
        constraints = constraints or {}
        
        # Research use case - prefer TPU v6e, then native JAX
        if use_case == 'research':
            if TPU_V6E_VQ_AVAILABLE:
                return 'tpu_v6e'
            elif JAX_NATIVE_INTEGRATION_AVAILABLE:
                return 'native_jax'
            
        # Production use case - prefer ARM Axion, then native implementations
        elif use_case == 'production':
            if ARM_AXION_VQ_AVAILABLE:
                return 'arm_axion'
            elif VQBIT_LAYER_AVAILABLE:
                return 'vqbit_layer'
            
        # Memory constrained - prefer ARM Axion, then adaptive
        elif use_case == 'memory_constrained':
            if ARM_AXION_VQ_AVAILABLE:
                return 'arm_axion'
            elif ADAPTIVE_VQ_AVAILABLE:
                return 'adaptive'
            
        # General case - prefer native JAX integration
        elif JAX_NATIVE_INTEGRATION_AVAILABLE:
            return 'native_jax'
            
        # Fallback to first available system
        priority_order = ['vqbit_layer', 'adaptive', 'multimodal', 'tpu_v6e', 'arm_axion']
        
        for backend in priority_order:
            if self.available_systems.get(backend):
                return backend
        
        # Final fallback
        for backend, available in self.available_systems.items():
            if available:
                return backend
                
        raise RuntimeError("No VQ backends available")
    
    def benchmark_backends(self, 
                          test_configs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Benchmark available VQ backends.
        
        Args:
            test_configs: List of test configurations
            
        Returns:
            Benchmark results
        """
        
        if test_configs is None:
            test_configs = [
                {'num_embeddings': 4096, 'embedding_dim': 512},
                {'num_embeddings': 8192, 'embedding_dim': 1024},
                {'num_embeddings': 16384, 'embedding_dim': 2048}
            ]
        
        results = {}
        
        for backend, available in self.available_systems.items():
            if not available:
                continue
                
            try:
                vq_system = self.create_vq_for_backend(backend, test_configs[0])
                
                if hasattr(vq_system, 'benchmark_performance'):
                    benchmark_result = vq_system.benchmark_performance()
                    results[backend] = benchmark_result
                else:
                    results[backend] = {'status': 'no_benchmark_available'}
                    
            except Exception as e:
                results[backend] = {'error': str(e)}
                
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return {
            'version': __version__,
            'available_systems': self.available_systems,
            'total_backends': sum(self.available_systems.values()),
            'recommended_backend': self.get_optimal_backend('general'),
            'system_info': {
                'tpu_v6e_features': [
                    'BF16 precision',
                    'Mesh sharding',
                    'Product quantization',
                    'EMA updates',
                    'MoE integration'
                ] if TPU_V6E_VQ_AVAILABLE else None,
                'arm_axion_features': [
                    'NEON vectorization',
                    'Memory optimization',
                    'Quantization support',
                    'Multi-threading'
                ] if ARM_AXION_VQ_AVAILABLE else None
            }
        }


# Global VQBit system instance
_vqbit_system = None

def get_vqbit_system() -> VQBitSystem:
    """Get global VQBit system instance."""
    global _vqbit_system
    if _vqbit_system is None:
        _vqbit_system = VQBitSystem()
    return _vqbit_system


# Convenience functions
def create_optimal_vq(use_case: str = "general", 
                     constraints: Optional[Dict[str, Any]] = None,
                     **kwargs) -> Any:
    """
    Create optimal VQ system for use case.
    
    Args:
        use_case: Use case ('research', 'production', 'memory_constrained')
        constraints: Hardware/performance constraints
        **kwargs: Additional configuration
        
    Returns:
        Optimal VQ system instance
    """
    
    system = get_vqbit_system()
    backend = system.get_optimal_backend(use_case, constraints)
    return system.create_vq_for_backend(backend, kwargs)


def get_vq_recommendations(use_case: str,
                          hardware_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get VQ recommendations for specific use case and hardware.
    
    Args:
        use_case: Use case description
        hardware_info: Available hardware information
        
    Returns:
        VQ recommendations
    """
    
    system = get_vqbit_system()
    
    recommendations = {
        'use_case': use_case,
        'hardware_info': hardware_info,
        'available_backends': system.available_systems,
        'recommendations': {}
    }
    
    # Use case specific recommendations
    if use_case == 'research':
        if TPU_V6E_VQ_AVAILABLE:
            recommendations['recommendations'] = {
                'backend': 'tpu_v6e',
                'config': {
                    'num_embeddings': 16384,
                    'embedding_dim': 2048,
                    'use_product_quantization': True,
                    'num_subspaces': 16,
                    'use_entropy_regularization': True,
                    'use_diversity_loss': True
                },
                'features': ['Maximum performance', 'Advanced features', 'BF16 precision']
            }
        else:
            recommendations['recommendations'] = {
                'backend': 'fallback',
                'message': 'TPU v6e not available, consider alternative backends'
            }
            
    elif use_case == 'production':
        if ARM_AXION_VQ_AVAILABLE:
            recommendations['recommendations'] = {
                'backend': 'arm_axion',
                'config': {
                    'embedding_dim': 1024,
                    'num_coofs': 64,
                    'intoble_sve2': True,
                    'enable_quantization': True
                },
                'features': ['High efficiency', 'Low latency', 'Memory optimized']
            }
        else:
            recommendations['recommendations'] = {
                'backend': 'fallback',
                'message': 'ARM Axion not available, consider alternative backends'
            }
            
    else:
        # General recommendations
        optimal_backend = system.get_optimal_backend(use_case)
        recommendations['recommendations'] = {
            'backend': optimal_backend,
            'config': {'use_defaults': True},
            'features': ['Automatic optimization', 'Balanced performance']
        }
    
    return recommendations


# Module exports
__all__ = [
    # Core classes
    'VQBitSystem',
    'get_vqbit_system',
    
    # VQbit Layer (core implementation)
    'VQbitLayer',
    'VQbitConfig', 
    'create_vqbit_layer',
    'VQBitLayer',  # Alias
    
    # TPU v6e VQ (if available)
    'TPUv6eVectorQuantizer',
    'TPUv6eVQConfig',
    'TPUv6eVQSystem',
    'create_tpu_v6e_vq_system',
    'create_tpu_v6e_vq_layer',
    
    # ARM Axion VQ (if available)
    'ARMAxionConfig',
    'ARMAxionVQOptimizer',
    
    # Adaptive VQ components
    'AdaptiveVectorQuantizer',
    'AdaptiveVQConfig',
    'AdaptationHistory',
    'create_adaptive_vq',
    'create_performance_adaptive_vq',
    'AdaptiveVQ',  # Alias
    
    # Multimodal VQbit components
    'VQbitModule',
    'MultimodalVQConfig',
    'create_multimodal_vqbit',
    
    # VQ Monitoring components
    'VQMonitor',
    'MonitoringConfig',
    'MetricsCollector',
    'PerformanceAnalyzer',
    'create_vq_monitor',
    'create_production_monitor',
    'get_global_monitor',
    'set_global_monitor',
    'monitor_vq_operation',
    
    # Wrapper components
    'AdaptiveWrapper',
    'AdaptiveWrapperConfig',
    'PerformanceTracker',
    'SimpleVQFallback',
    'create_adaptive_wrapper',
    'create_performance_optimized_wrapper',
    
    # JAX Native Integration
    'VQbitNativeIntegration',
    'NativeIntegrationConfig',
    'get_native_integration',
    'create_native_vqbit_layer',
    'create_native_multimodal_vqbit',
    'get_native_backend_info',
    'benchmark_native_backends',
    'integrate_with_tpu_backend',
    
    # Convenience functions
    'create_optimal_vq',
    'get_vq_recommendations',
    
    # Integration functions
    'integrate_with_moe',
    'integrate_with_adaptive',
    
    # Utility functions
    'get_project_root',
    
    # Status flags
    'TPU_V6E_VQ_AVAILABLE',
    'ARM_AXION_VQ_AVAILABLE',
    'VQBIT_LAYER_AVAILABLE',
    'ADAPTIVE_VQ_AVAILABLE',
    'MULTIMODAL_VQ_AVAILABLE',
    'VQ_MONITORING_AVAILABLE',
    'VQ_WRAPPER_AVAILABLE',
    'JAX_NATIVE_INTEGRATION_AVAILABLE'
]

# Log initialization status
logger.info(f"VQBit module initialized (v{__version__})")
logger.info(f"TPU v6e VQ available: {TPU_V6E_VQ_AVAILABLE}")
logger.info(f"ARM Axion VQ available: {ARM_AXION_VQ_AVAILABLE}")
logger.info(f"VQbit Layer available: {VQBIT_LAYER_AVAILABLE}")
logger.info(f"Adaptive VQ available: {ADAPTIVE_VQ_AVAILABLE}")
logger.info(f"Multimodal VQ available: {MULTIMODAL_VQ_AVAILABLE}")
logger.info(f"VQ Monitoring available: {VQ_MONITORING_AVAILABLE}")
logger.info(f"VQ Wrapper available: {VQ_WRAPPER_AVAILABLE}")
logger.info(f"JAX Native Integration available: {JAX_NATIVE_INTEGRATION_AVAILABLE}")

# Determine system readiness
available_components = sum([
    TPU_V6E_VQ_AVAILABLE,
    ARM_AXION_VQ_AVAILABLE, 
    VQBIT_LAYER_AVAILABLE,
    ADAPTIVE_VQ_AVAILABLE,
    MULTIMODAL_VQ_AVAILABLE,
    VQ_MONITORING_AVAILABLE,
    VQ_WRAPPER_AVAILABLE,
    JAX_NATIVE_INTEGRATION_AVAILABLE
])

if available_components >= 6:
    logger.info("🚀 Full VQBit system ready with all components")
elif available_components >= 4:
    logger.info("⚡ Most VQBit components available")
elif available_components >= 2:
    logger.info("🔧 Basic VQBit system available")
else:
    logger.warning("⚠️ Limited VQBit capabilities available")
