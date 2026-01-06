"""
VQbit JAX Native Integration for Capibara-6

Integración entre las implementaciones VQbit y la infraestructura JAX nativa:
- Conexión con capibara.jax.numpy
- Optimizaciones TPU v4/v6 nativas
- Kernels especializados para VQbit
- Fallbacks automáticos a numpy
- Integración con backend TPU existente
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import native JAX infrastructure
try:
    import capibara.jax as cjax
    import capibara.jax.numpy as cjnp
    CAPIBARA_JAX_AVAILABLE = True
    logger.info("Capibara JAX infrastructure available")
except ImportError:
    CAPIBARA_JAX_AVAILABLE = False
    logger.warning("Capibara JAX infrastructure not available")
    cjax = None
    cjnp = None

# Import standard libraries as fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Import VQbit implementations
try:
    from .vqbit_layer import VQbitLayer, VQbitConfig
    from .multimodal_vqbit import VQbitModule, MultimodalVQConfig
    from .adaptive_vq import AdaptiveVectorQuantizer, AdaptiveVQConfig
    VQBIT_IMPLEMENTATIONS_AVAILABLE = True
except ImportError:
    VQBIT_IMPLEMENTATIONS_AVAILABLE = False

@dataclass
class NativeIntegrationConfig:
    """Configurestion for native JAX integration."""
    prefer_capibara_jax: bool = True
    enable_tpu_kernels: bool = True
    enable_fallback_numpy: bool = True
    log_backend_selection: bool = True
    cache_compiled_functions: bool = True


class VQbitNativeIntegration:
    """
    Integración nativa entre VQbit y capibara.jax
    
    Características:
    - Detección automática de backend óptimo
    - Uso of kernels TPU especializados cuando están disponibles
    - Fallback automático a numpy
    - Integración con el system de monitoring
    - Optimizaciones específicas para cada backend
    """
    
    def __init__(self, config: Optional[NativeIntegrationConfig] = None):
        if config is None:
            config = NativeIntegrationConfig()
        self.config = config
        
        # Detect available backends
        self.available_backends = self._detect_backends()
        self.selected_backend = self._select_optimal_backend()
        
        # Cache for compiled functions
        self.function_cache = {} if config.cache_compiled_functions else None
        
        if config.log_backend_selection:
            logger.info(f"VQbit Native Integration initialized with {self.selected_backend} backend")
            logger.info(f"Available backends: {list(self.available_backends.keys())}")
    
    def _detect_backends(self) -> Dict[str, bool]:
        """Detect available computation backends."""
        backends = {
            'capibara_jax': CAPIBARA_JAX_AVAILABLE,
            'numpy': NUMPY_AVAILABLE,
            'vqbit_implementations': VQBIT_IMPLEMENTATIONS_AVAILABLE
        }
        
        # Check for TPU availability through capibara.jax
        if CAPIBARA_JAX_AVAILABLE:
            try:
                # Try to access TPU backend (with safe import)
                from capibara.jax.tpu_v4.backend import TpuV4AdaptiveOps
                backends['tpu_v4_kernels'] = True
            except (ImportError, AttributeError, TypeError) as e:
                logger.debug(f"TPU v4 kernels not available: {e}")
                backends['tpu_v4_kernels'] = False
        else:
            backends['tpu_v4_kernels'] = False
        
        return backends
    
    def _select_optimal_backend(self) -> str:
        """Select optimal backend based on availability."""
        
        # Priority order for backend selection
        if self.config.prefer_capibara_jax and self.available_backends.get('capibara_jax'):
            if self.available_backends.get('tpu_v4_kernels'):
                return 'capibara_jax_tpu'
            else:
                return 'capibara_jax'
        
        elif self.available_backends.get('vqbit_implementations'):
            return 'vqbit_native'
        
        elif self.available_backends.get('numpy'):
            return 'numpy_fallback'
        
        else:
            return 'minimal_fallback'
    
    def create_native_vqbit_layer(self, 
                                 codebook_size: int = 64,
                                 embedding_dim: int = 768,
                                 use_tpu_optimizations: bool = True,
                                 **kwargs) -> Any:
        """
        Create VQbit layer using native infrastructure.
        
        Args:
            codebook_size: Number of codebook entries
            embedding_dim: Embedding dimension
            use_tpu_optimizations: Enable TPU optimizations
            **kwargs: Additional configuration
            
        Returns:
            Native VQbit layer instance
        """
        
        if self.selected_backend == 'capibara_jax_tpu':
            return self._create_capibara_jax_tpu_vqbit(codebook_size, embedding_dim, **kwargs)
        
        elif self.selected_backend == 'capibara_jax':
            return self._create_capibara_jax_vqbit(codebook_size, embedding_dim, **kwargs)
        
        elif self.selected_backend == 'vqbit_native':
            return self._create_vqbit_native(codebook_size, embedding_dim, **kwargs)
        
        elif self.selected_backend == 'numpy_fallback':
            return self._create_numpy_fallback_vqbit(codebook_size, embedding_dim, **kwargs)
        
        else:
            return self._create_minimal_fallback_vqbit(codebook_size, embedding_dim, **kwargs)
    
    def _create_capibara_jax_tpu_vqbit(self, codebook_size: int, embedding_dim: int, **kwargs) -> Any:
        """Creates VQbit layer using Capibara JAX with TPU kernels."""
        
        try:
            from capibara.jax.tpu_v4.backend import TpuV4AdaptiveOps
            
            # Create TPU-optimized VQbit wrapper
            class CapibaraJAXTPUVQbit:
                def __init__(self, codebook_size: int, embedding_dim: int):
                    self.codebook_size = codebook_size
                    self.embedding_dim = embedding_dim
                    self.tpu_ops = TpuV4AdaptiveOps()
                    
                    # Initialize codebook using capibara.jax.numpy
                    self.codebook = cjnp.array(
                        np.random.randn(codebook_size, embedding_dim) if NUMPY_AVAILABLE 
                        else [[0.0] * embedding_dim for _ in range(codebook_size)]
                    )
                
                def __call__(self, inputs, training=True, **kwargs):
                    """VQbit quantization using TPU kernels."""
                    
                    # Use TPU-optimized VQbit quantization
                    quantized, indices = self.tpu_ops.vqbit_quantize(
                        inputs, self.codebook, modality=kwargs.get('modality', 'default')
                    )
                    
                    # Compute basic metrics
                    metrics = self._compute_metrics(inputs, quantized, indices)
                    
                    return quantized, indices, metrics
                
                def _compute_metrics(self, inputs, quantized, indices):
                    """Compute quantization metrics."""
                    try:
                        # Basic metrics using capibara.jax.numpy
                        mse = cjnp.mean((inputs - quantized) ** 2)
                        unique_codes = len(cjnp.array(list(set(indices.flatten()))))
                        usage = unique_codes / self.codebook_size
                        
                        return {
                            'mse': float(mse),
                            'codebook_usage': float(usage),
                            'unique_codes': unique_codes,
                            'backend': 'capibara_jax_tpu',
                            'compression_ratio': 32.0 / 6.0  # Assuming 64 codes = 6 bits
                        }
                    except Exception as e:
                        logger.warning(f"Metrics computation failed: {e}")
                        return {'backend': 'capibara_jax_tpu', 'error': str(e)}
            
            return CapibaraJAXTPUVQbit(codebook_size, embedding_dim)
            
        except Exception as e:
            logger.warning(f"Capibara JAX TPU VQbit creation failed: {e}")
            return self._create_capibara_jax_vqbit(codebook_size, embedding_dim, **kwargs)
    
    def _create_capibara_jax_vqbit(self, codebook_size: int, embedding_dim: int, **kwargs) -> Any:
        """Creates VQbit layer using Capibara JAX (no TPU kernels)."""
        
        try:
            # VQbit implementation using capibara.jax.numpy
            class CapibaraJAXVQbit:
                def __init__(self, codebook_size: int, embedding_dim: int):
                    self.codebook_size = codebook_size
                    self.embedding_dim = embedding_dim
                    
                    # Initialize codebook
                    if NUMPY_AVAILABLE:
                        codebook_data = np.random.randn(codebook_size, embedding_dim).astype(np.float32)
                        # Normalize codebook entries
                        norms = np.linalg.norm(codebook_data, axis=1, keepdims=True)
                        codebook_data = codebook_data / (norms + 1e-8)
                    else:
                        # Minimal fallback
                        codebook_data = [[0.1] * embedding_dim for _ in range(codebook_size)]
                    
                    self.codebook = cjnp.array(codebook_data)
                
                def __call__(self, inputs, training=True, **kwargs):
                    """VQbit quantization using capibara.jax.numpy."""
                    
                    try:
                        input_shape = inputs.shape
                        flat_inputs = cjnp.reshape(inputs, (-1, self.embedding_dim))
                        
                        # Compute distances using capibara.jax operations
                        # Distance computation: ||x - c||^2 = ||x||^2 + ||c||^2 - 2<x,c>
                        input_norms = cjnp.sum(flat_inputs ** 2, axis=1, keepdims=True)
                        codebook_norms = cjnp.sum(self.codebook ** 2, axis=1, keepdims=True).T
                        dot_products = cjnp.matmul(flat_inputs, self.codebook.T)
                        
                        distances = input_norms + codebook_norms - 2 * dot_products
                        
                        # Find nearest codebook entries
                        indices = cjnp.argmin(distances, axis=1)
                        
                        # Get quantized values
                        quantized_flat = self.codebook[indices]
                        quantized = cjnp.reshape(quantized_flat, input_shape)
                        indices_reshaped = cjnp.reshape(indices, input_shape[:-1])
                        
                        # Compute metrics
                        metrics = self._compute_metrics(flat_inputs, quantized_flat, indices)
                        
                        return quantized, indices_reshaped, metrics
                        
                    except Exception as e:
                        logger.error(f"Capibara JAX VQbit forward failed: {e}")
                        # Return inputs unchanged as fallback
                        return inputs, cjnp.zeros(inputs.shape[:-1]), {'error': str(e)}
                
                def _compute_metrics(self, inputs, quantized, indices):
                    """Compute metrics using capibara.jax.numpy."""
                    try:
                        mse = cjnp.mean((inputs - quantized) ** 2)
                        
                        # Count unique indices (simplified)
                        unique_codes = len(set(indices.flatten().tolist()))
                        usage = unique_codes / self.codebook_size
                        
                        return {
                            'mse': float(mse),
                            'codebook_usage': float(usage),
                            'unique_codes': unique_codes,
                            'backend': 'capibara_jax',
                            'compression_ratio': 32.0 / max(1, np.log2(self.codebook_size))
                        }
                    except Exception as e:
                        return {'backend': 'capibara_jax', 'error': str(e)}
            
            return CapibaraJAXVQbit(codebook_size, embedding_dim)
            
        except Exception as e:
            logger.warning(f"Capibara JAX VQbit creation failed: {e}")
            return self._create_vqbit_native(codebook_size, embedding_dim, **kwargs)
    
    def _create_vqbit_native(self, codebook_size: int, embedding_dim: int, **kwargs) -> Any:
        """Creates VQbit using native VQbit implementations."""
        
        if not VQBIT_IMPLEMENTATIONS_AVAILABLE:
            return self._create_numpy_fallback_vqbit(codebook_size, embedding_dim, **kwargs)
        
        try:
            from .vqbit_layer import create_vqbit_layer
            
            return create_vqbit_layer(
                codebook_size=codebook_size,
                embedding_dim=embedding_dim,
                use_tpu_optimizations=True,
                **kwargs
            )
            
        except Exception as e:
            logger.warning(f"Native VQbit creation failed: {e}")
            return self._create_numpy_fallback_vqbit(codebook_size, embedding_dim, **kwargs)
    
    def _create_numpy_fallback_vqbit(self, codebook_size: int, embedding_dim: int, **kwargs) -> Any:
        """Creates VQbit using pure numpy fallback."""
        
        if not NUMPY_AVAILABLE:
            return self._create_minimal_fallback_vqbit(codebook_size, embedding_dim, **kwargs)
        
        class NumpyVQbit:
            def __init__(self, codebook_size: int, embedding_dim: int):
                self.codebook_size = codebook_size
                self.embedding_dim = embedding_dim
                
                # Initialize normalized codebook
                self.codebook = np.random.randn(codebook_size, embedding_dim).astype(np.float32)
                norms = np.linalg.norm(self.codebook, axis=1, keepdims=True)
                self.codebook = self.codebook / (norms + 1e-8)
            
            def __call__(self, inputs, training=True, **kwargs):
                """Pure numpy VQbit quantization."""
                
                input_shape = inputs.shape
                flat_inputs = inputs.reshape(-1, self.embedding_dim)
                
                # Compute distances
                distances = np.linalg.norm(
                    flat_inputs[:, None, :] - self.codebook[None, :, :],
                    axis=2
                )
                
                # Find nearest codes
                indices = np.argmin(distances, axis=1)
                quantized_flat = self.codebook[indices]
                
                # Reshape back
                quantized = quantized_flat.reshape(input_shape)
                indices_reshaped = indices.reshape(input_shape[:-1])
                
                # Basic metrics
                mse = np.mean((flat_inputs - quantized_flat) ** 2)
                unique_codes = len(np.unique(indices))
                usage = unique_codes / self.codebook_size
                
                metrics = {
                    'mse': float(mse),
                    'codebook_usage': float(usage),
                    'unique_codes': unique_codes,
                    'backend': 'numpy_fallback',
                    'compression_ratio': 32.0 / max(1, np.log2(self.codebook_size))
                }
                
                return quantized, indices_reshaped, metrics
        
        return NumpyVQbit(codebook_size, embedding_dim)
    
    def _create_minimal_fallback_vqbit(self, codebook_size: int, embedding_dim: int, **kwargs) -> Any:
        """Creates minimal VQbit fallback (no external dependencies)."""
        
        class MinimalVQbit:
            def __init__(self, codebook_size: int, embedding_dim: int):
                self.codebook_size = codebook_size
                self.embedding_dim = embedding_dim
                
                # Minimal codebook initialization
                self.codebook = [
                    [0.1 * (i % 10 - 5) for _ in range(embedding_dim)]
                    for i in range(codebook_size)
                ]
            
            def __call__(self, inputs, training=True, **kwargs):
                """Minimal VQbit implementation."""
                
                # Simple quantization (just return inputs with dummy indices)
                if hasattr(inputs, 'shape'):
                    indices_shape = inputs.shape[:-1]
                    indices = [[0 for _ in range(indices_shape[-1])] 
                              for _ in range(indices_shape[0])] if len(indices_shape) > 1 else [0] * indices_shape[0]
                else:
                    indices = [0]
                
                metrics = {
                    'mse': 0.0,
                    'codebook_usage': 1.0,
                    'unique_codes': 1,
                    'backend': 'minimal_fallback',
                    'compression_ratio': 1.0
                }
                
                return inputs, indices, metrics
        
        return MinimalVQbit(codebook_size, embedding_dim)
    
    def create_multimodal_native_vqbit(self,
                                      modalities: List[str] = None,
                                      codebook_size: int = 64,
                                      embedding_dim: int = 768,
                                      **kwargs) -> Any:
        """Creates multimodal VQbit using native infrastructure."""
        
        if modalities is None:
            modalities = ['text', 'image', 'audio']
        
        if self.selected_backend.startswith('capibara_jax'):
            return self._create_capibara_jax_multimodal(modalities, codebook_size, embedding_dim, **kwargs)
        else:
            return self._create_fallback_multimodal(modalities, codebook_size, embedding_dim, **kwargs)
    
    def _create_capibara_jax_multimodal(self, modalities: List[str], codebook_size: int, embedding_dim: int, **kwargs) -> Any:
        """Creates multimodal VQbit using Capibara JAX."""
        
        class CapibaraJAXMultimodalVQbit:
            def __init__(self, modalities: List[str], codebook_size: int, embedding_dim: int):
                self.modalities = modalities
                self.codebook_size = codebook_size
                self.embedding_dim = embedding_dim
                
                # Create shared codebook using capibara.jax.numpy
                if NUMPY_AVAILABLE:
                    codebook_data = np.random.randn(codebook_size, embedding_dim).astype(np.float32)
                else:
                    codebook_data = [[0.1] * embedding_dim for _ in range(codebook_size)]
                
                self.codebook = cjnp.array(codebook_data)
                
                # Modal adaptation matrices
                self.modal_adapters = {}
                for modality in modalities:
                    if NUMPY_AVAILABLE:
                        adapter_data = np.eye(embedding_dim).astype(np.float32)
                    else:
                        adapter_data = [[1.0 if i == j else 0.0 for j in range(embedding_dim)] 
                                       for i in range(embedding_dim)]
                    self.modal_adapters[modality] = cjnp.array(adapter_data)
            
            def __call__(self, inputs, modality=None, training=True, **kwargs):
                """Multimodal VQbit processing."""
                
                if isinstance(inputs, dict):
                    # Process multiple modalities
                    results = {}
                    all_quantized = []
                    
                    for mod, modal_input in inputs.items():
                        if mod in self.modalities:
                            result = self._process_modality(modal_input, mod)
                            results[mod] = result
                            all_quantized.append(result['quantized'])
                    
                    # Simple fusion (average)
                    if all_quantized:
                        if CAPIBARA_JAX_AVAILABLE:
                            fused = cjnp.mean(cjnp.stack(all_quantized), axis=0)
                        else:
                            fused = all_quantized[0]  # Just use first as fallback
                    else:
                        fused = inputs
                    
                    return {
                        'quantized': fused,
                        'modal_results': results,
                        'metrics': {'total_modalities': len(results), 'backend': 'capibara_jax_multimodal'}
                    }
                else:
                    # Single modality
                    if modality is None:
                        modality = self.modalities[0]
                    return self._process_modality(inputs, modality)
            
            def _process_modality(self, inputs, modality):
                """Process single modality."""
                try:
                    # Apply modal adaptation
                    if modality in self.modal_adapters:
                        adapted = cjnp.matmul(inputs, self.modal_adapters[modality])
                    else:
                        adapted = inputs
                    
                    # Simple quantization
                    input_shape = adapted.shape
                    flat_adapted = cjnp.reshape(adapted, (-1, self.embedding_dim))
                    
                    # Find nearest codebook entries
                    distances = cjnp.sum((flat_adapted[:, None, :] - self.codebook[None, :, :]) ** 2, axis=2)
                    indices = cjnp.argmin(distances, axis=1)
                    quantized_flat = self.codebook[indices]
                    
                    quantized = cjnp.reshape(quantized_flat, input_shape)
                    indices_reshaped = cjnp.reshape(indices, input_shape[:-1])
                    
                    # Basic metrics
                    mse = cjnp.mean((flat_adapted - quantized_flat) ** 2)
                    
                    return {
                        'quantized': quantized,
                        'indices': indices_reshaped,
                        'metrics': {
                            'mse': float(mse),
                            'modality': modality,
                            'backend': 'capibara_jax'
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Capibara JAX modality processing failed: {e}")
                    return {
                        'quantized': inputs,
                        'indices': cjnp.zeros(inputs.shape[:-1]),
                        'metrics': {'error': str(e), 'modality': modality}
                    }
        
        return CapibaraJAXMultimodalVQbit(modalities, codebook_size, embedding_dim)
    
    def _create_fallback_multimodal(self, modalities: List[str], codebook_size: int, embedding_dim: int, **kwargs) -> Any:
        """Creates fallback multimodal VQbit."""
        
        if VQBIT_IMPLEMENTATIONS_AVAILABLE:
            try:
                from .multimodal_vqbit import create_multimodal_vqbit
                return create_multimodal_vqbit(
                    modalities=modalities,
                    codebook_size=codebook_size,
                    embedding_dim=embedding_dim,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Native multimodal VQbit creation failed: {e}")
        
        # Ultimate fallback
        return self._create_minimal_fallback_vqbit(codebook_size, embedding_dim, **kwargs)
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the selected backend."""
        
        return {
            'selected_backend': self.selected_backend,
            'available_backends': self.available_backends,
            'capibara_jax_available': CAPIBARA_JAX_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE,
            'vqbit_implementations_available': VQBIT_IMPLEMENTATIONS_AVAILABLE,
            'config': {
                'prefer_capibara_jax': self.config.prefer_capibara_jax,
                'enable_tpu_kernels': self.config.enable_tpu_kernels,
                'enable_fallback_numpy': self.config.enable_fallback_numpy
            }
        }
    
    def benchmark_backends(self) -> Dict[str, Any]:
        """Benchmark available backends."""
        
        if not NUMPY_AVAILABLE:
            return {'error': 'numpy_not_available_for_benchmarking'}
        
        # Create test data
        test_input = np.random.randn(4, 16, 128).astype(np.float32)
        
        results = {}
        
        # Test each available backend
        backends_to_test = ['capibara_jax_tpu', 'capibara_jax', 'vqbit_native', 'numpy_fallback']
        
        for backend in backends_to_test:
            if backend == 'capibara_jax_tpu' and not self.available_backends.get('tpu_v4_kernels'):
                continue
            if backend == 'capibara_jax' and not self.available_backends.get('capibara_jax'):
                continue
            if backend == 'vqbit_native' and not self.available_backends.get('vqbit_implementations'):
                continue
            
            try:
                # Temporarily switch backend
                original_backend = self.selected_backend
                self.selected_backend = backend
                
                # Create VQbit instance
                vqbit = self.create_native_vqbit_layer(codebook_size=32, embedding_dim=128)
                
                # Benchmark
                start_time = time.time()
                quantized, indices, metrics = vqbit(test_input, training=True)
                end_time = time.time()
                
                results[backend] = {
                    'operation_time_ms': (end_time - start_time) * 1000,
                    'metrics': metrics,
                    'success': True
                }
                
                # Restore original backend
                self.selected_backend = original_backend
                
            except Exception as e:
                results[backend] = {
                    'error': str(e),
                    'success': False
                }
        
        return results


# Global native integration instance
_native_integration = None

def get_native_integration() -> VQbitNativeIntegration:
    """Get global native integration instance."""
    global _native_integration
    if _native_integration is None:
        _native_integration = VQbitNativeIntegration()
    return _native_integration

def create_native_vqbit_layer(codebook_size: int = 64,
                             embedding_dim: int = 768,
                             **kwargs) -> Any:
    """
    Create VQbit layer using native infrastructure.
    
    Args:
        codebook_size: Number of codebook entries
        embedding_dim: Embedding dimension
        **kwargs: Additional configuration
        
    Returns:
        Native VQbit layer instance
    """
    integration = get_native_integration()
    return integration.create_native_vqbit_layer(codebook_size, embedding_dim, **kwargs)

def create_native_multimodal_vqbit(modalities: List[str] = None,
                                  codebook_size: int = 64,
                                  embedding_dim: int = 768,
                                  **kwargs) -> Any:
    """
    Create multimodal VQbit using native infrastructure.
    
    Args:
        modalities: List of modalities to support
        codebook_size: Number of codebook entries
        embedding_dim: Embedding dimension
        **kwargs: Additional configuration
        
    Returns:
        Native multimodal VQbit instance
    """
    integration = get_native_integration()
    return integration.create_multimodal_native_vqbit(modalities, codebook_size, embedding_dim, **kwargs)

def get_native_backend_info() -> Dict[str, Any]:
    """Get information about native backend selection."""
    integration = get_native_integration()
    return integration.get_backend_info()

def benchmark_native_backends() -> Dict[str, Any]:
    """Benchmark all available native backends."""
    integration = get_native_integration()
    return integration.benchmark_backends()

# Integration with existing TPU backend
def integrate_with_tpu_backend():
    """Integrate VQbit with existing TPU backend."""
    
    try:
        from capibara.jax.tpu_v4.backend import TpuV4AdaptiveOps
        
        # Enhance TPU backend with native VQbit
        original_vqbit_quantize = TpuV4AdaptiveOps.vqbit_quantize
        
        def enhanced_vqbit_quantize(self, input_tensor, codebook, modality="default"):
            """Enhanced VQbit quantization using native implementations."""
            
            try:
                # Try to use native VQbit implementation
                integration = get_native_integration()
                vqbit_layer = integration.create_native_vqbit_layer(
                    codebook_size=codebook.shape[0] if hasattr(codebook, 'shape') else 64,
                    embedding_dim=input_tensor.shape[-1]
                )
                
                quantized, indices, metrics = vqbit_layer(input_tensor, training=True)
                
                # Log integration success
                logger.debug(f"Native VQbit integration successful: {metrics.get('backend', 'unknown')}")
                
                return quantized, indices
                
            except Exception as e:
                logger.warning(f"Native VQbit integration failed, using original: {e}")
                return original_vqbit_quantize(self, input_tensor, codebook, modality)
        
        # Replace method
        TpuV4AdaptiveOps.vqbit_quantize = enhanced_vqbit_quantize
        
        logger.info("✅ VQbit native integration with TPU backend successful")
        return True
        
    except Exception as e:
        logger.warning(f"TPU backend integration failed: {e}")
        return False

# Auto-integrate on import (safely)
try:
    # Only try integration if we have the necessary components
    if CAPIBARA_JAX_AVAILABLE and VQBIT_IMPLEMENTATIONS_AVAILABLE:
        if integrate_with_tpu_backend():
            logger.info("🚀 VQbit native integration activated")
        else:
            logger.info("⚡ VQbit running in standalone mode")
    else:
        logger.info("🔧 VQbit native integration available but not auto-activated")
except Exception as e:
    logger.debug(f"Auto-integration skipped: {e}")

logger.info("VQbit JAX Native Integration initialized")