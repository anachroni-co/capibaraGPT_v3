"""
Quantization Adapter

Provides a unified interface for different quantization methods
(VQbit, BitNet, INT8, etc.) con selección automática basada en requisitos
de rendimiento y precisión.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
import numpy as np

from .base_adapter import BaseAdapter, AdapterConfig
from .adapter_registry import register_adapter_decorator, AdapterType

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Available quantization types."""
    VQBIT = "vqbit"
    BITNET = "bitnet"
    INT8 = "int8"
    INT4 = "int4"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    DYNAMIC = "dynamic"
    STATIC = "static"

class QuantizationQuality(Enum):
    """Quantization quality levels."""
    MAXIMUM_COMPRESSION = "maximum_compression"  # Maximum compression, lower quality
    BALANCED = "balanced"                        # Balance between compression and quality
    HIGH_QUALITY = "high_quality"               # High quality, lower compression
    LOSSLESS = "lossless"                       # No loss of information

@dataclass
class QuantizationConfig:
    """Quantization configuration."""
    quantization_type: QuantizationType
    quality_level: QuantizationQuality = QuantizationQuality.BALANCED
    target_compression_ratio: Optional[float] = None
    preserve_accuracy_threshold: float = 0.95  # Minimum 95% accuracy
    enable_calibration: bool = True
    calibration_samples: int = 1000
    enable_fine_tuning: bool = False
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QuantizationResult:
    """Result of a quantization operation."""
    quantized_data: Any
    compression_ratio: float
    accuracy_retention: float
    quantization_time_ms: float
    memory_savings_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class QuantizationMethod:
    """Base class for quantization methods."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.calibrated = False
        self.calibration_data = None
        
    def calibrate(self, calibration_data: Any) -> bool:
        """Calibrates the quantization method."""
        self.calibration_data = calibration_data
        self.calibrated = True
        return True
    
    def quantize(self, data: Any) -> QuantizationResult:
        """Quantizes the data."""
        raise NotImplementedError
    
    def dequantize(self, quantized_data: Any) -> Any:
        """Dequantizes the data."""
        raise NotImplementedError
    
    def estimate_compression_ratio(self, data_shape: Tuple[int, ...]) -> float:
        """Estimates compression ratio."""
        raise NotImplementedError

class VQbitQuantizationMethod(QuantizationMethod):
    """VQbit quantization method."""
    
    def __init__(self, config: QuantizationConfig):
        super().__init__(config)
        self.codebook_size = config.custom_parameters.get('codebook_size', 64)
        self.num_codebooks = config.custom_parameters.get('num_codebooks', 8)
        self.codebooks = None
        
    def calibrate(self, calibration_data: Any) -> bool:
        """Calibrates VQ codebooks."""
        try:
            # Import VQbit from project
            from capibara.jax.tpu_v4.backend import tpu_v4_ops
            
            # Use calibration data to train codebooks
            if hasattr(calibration_data, 'shape') and len(calibration_data.shape) >= 2:
                # Train codebooks using approximate K-means
                self.codebooks = self._train_codebooks(calibration_data)
                self.calibrated = True
                logger.info(f"VQbit calibration completed with {self.num_codebooks} codebooks")
                return True
            else:
                logger.warning("Invalid calibration data for VQbit")
                return False
                
        except ImportError:
            logger.warning("VQbit modules not available, using fallback calibration")
            self.calibrated = True
            return True
        except Exception as e:
            logger.error(f"VQbit calibration failed: {e}")
            return False
    
    def quantize(self, data: Any) -> QuantizationResult:
        """Quantizes using VQbit."""
        start_time = time.time()
        
        try:
            if not self.calibrated:
                # Automatic calibration with input data
                self.calibrate(data)
            
            # Import and use VQbit
            from capibara.jax.tpu_v4.backend import tpu_v4_ops
            
            original_size = self._calculate_size(data)
            
            # Quantize using VQbit
            quantized_codes, codebooks = tpu_v4_ops.vqbit_quantize(data, self.codebooks)
            
            if quantized_codes is not None:
                quantized_size = self._calculate_size(quantized_codes) + self._calculate_size(codebooks)
                compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
                
                # Estimate accuracy retention
                accuracy_retention = self._estimate_vqbit_accuracy()
                
                quantization_time = (time.time() - start_time) * 1000
                memory_savings = (original_size - quantized_size) / (1024 * 1024)  # MB
                
                return QuantizationResult(
                    quantized_data={'codes': quantized_codes, 'codebooks': codebooks},
                    compression_ratio=compression_ratio,
                    accuracy_retention=accuracy_retention,
                    quantization_time_ms=quantization_time,
                    memory_savings_mb=memory_savings,
                    metadata={
                        'method': 'vqbit',
                        'codebook_size': self.codebook_size,
                        'num_codebooks': self.num_codebooks
                    }
                )
            else:
                # Fallback to simple quantization
                return self._fallback_quantization(data, start_time)
                
        except Exception as e:
            logger.warning(f"VQbit quantization failed: {e}, using fallback")
            return self._fallback_quantization(data, start_time)
    
    def dequantize(self, quantized_data: Any) -> Any:
        """Dequantizes VQbit data."""
        try:
            if isinstance(quantized_data, dict) and 'codes' in quantized_data:
                codes = quantized_data['codes']
                codebooks = quantized_data.get('codebooks', self.codebooks)
                
                # Reconstruir datos originales
                return self._reconstruct_from_codes(codes, codebooks)
            else:
                logger.warning("Invalid VQbit quantized data format")
                return quantized_data
                
        except Exception as e:
            logger.error(f"VQbit dequantization failed: {e}")
            return quantized_data
    
    def estimate_compression_ratio(self, data_shape: Tuple[int, ...]) -> float:
        """Estimates compression ratio for VQbit."""
        if len(data_shape) < 2:
            return 1.0
        
        # Cálculo teórico de compresión VQbit
        total_elements = np.prod(data_shape)
        vector_dim = data_shape[-1]
        
        # Bits por vector cuantizado
        bits_per_code = np.log2(self.codebook_size)
        codes_per_vector = self.num_codebooks
        
        # Tamaño cuantizado vs original
        original_bits = total_elements * 32  # float32
        quantized_bits = (total_elements // vector_dim) * codes_per_vector * bits_per_code
        codebook_bits = self.num_codebooks * self.codebook_size * (vector_dim // self.num_codebooks) * 32
        
        total_quantized_bits = quantized_bits + codebook_bits
        
        return original_bits / total_quantized_bits if total_quantized_bits > 0 else 1.0
    
    def _train_codebooks(self, data: Any) -> Any:
        """Trains codebooks using approximate K-means."""
        # Simplified implementation of entrenamiento de codebooks
        try:
            if hasattr(data, 'shape'):
                batch_size, vector_dim = data.shape[0], data.shape[-1]
                vectors_per_codebook = vector_dim // self.num_codebooks
                
                # Create initial random codebooks
                import numpy as np
                codebooks = []
                
                for i in range(self.num_codebooks):
                    start_idx = i * vectors_per_codebook
                    end_idx = start_idx + vectors_per_codebook
                    
                    # Extract sub-vectors
                    if hasattr(data, 'numpy'):
                        sub_data = data.numpy()[..., start_idx:end_idx]
                    else:
                        sub_data = np.array(data)[..., start_idx:end_idx]
                    
                    # Create random codebook based on data
                    if sub_data.size > 0:
                        mean = np.mean(sub_data, axis=0)
                        std = np.std(sub_data, axis=0)
                        codebook = np.random.normal(mean, std, (self.codebook_size, vectors_per_codebook))
                    else:
                        codebook = np.random.randn(self.codebook_size, vectors_per_codebook)
                    
                    codebooks.append(codebook)
                
                return np.array(codebooks) if codebooks else None
            else:
                return None
                
        except Exception as e:
            logger.error(f"Codebook training failed: {e}")
            return None
    
    def _estimate_vqbit_accuracy(self) -> float:
        """Estimates accuracy retention for VQbit."""
        # Estimation based on quality configuration
        quality_accuracy = {
            QuantizationQuality.MAXIMUM_COMPRESSION: 0.85,
            QuantizationQuality.BALANCED: 0.92,
            QuantizationQuality.HIGH_QUALITY: 0.97,
            QuantizationQuality.LOSSLESS: 0.99
        }
        
        base_accuracy = quality_accuracy.get(self.config.quality_level, 0.90)
        
        # Adjust based on codebook size
        codebook_factor = min(self.codebook_size / 64.0, 1.0)  # Normalize to 64
        
        return min(base_accuracy * (0.9 + 0.1 * codebook_factor), 0.99)
    
    def _fallback_quantization(self, data: Any, start_time: float) -> QuantizationResult:
        """Simple fallback quantization."""
        try:
            import numpy as np
            
            if hasattr(data, 'numpy'):
                np_data = data.numpy()
            else:
                np_data = np.array(data)
            
            # Simple int8 quantization
            data_min, data_max = np_data.min(), np_data.max()
            scale = (data_max - data_min) / 255.0
            quantized = np.round((np_data - data_min) / scale).astype(np.int8)
            
            original_size = np_data.nbytes
            quantized_size = quantized.nbytes + 8  # +8 para min/max
            
            quantization_time = (time.time() - start_time) * 1000
            
            return QuantizationResult(
                quantized_data={'data': quantized, 'scale': scale, 'offset': data_min},
                compression_ratio=original_size / quantized_size,
                accuracy_retention=0.85,  # Conservative estimate
                quantization_time_ms=quantization_time,
                memory_savings_mb=(original_size - quantized_size) / (1024 * 1024),
                metadata={'method': 'fallback_int8'}
            )
            
        except Exception as e:
            logger.error(f"Fallback quantization failed: {e}")
            return QuantizationResult(
                quantized_data=data,
                compression_ratio=1.0,
                accuracy_retention=1.0,
                quantization_time_ms=(time.time() - start_time) * 1000,
                memory_savings_mb=0.0,
                metadata={'method': 'no_quantization', 'error': str(e)}
            )
    
    def _calculate_size(self, data: Any) -> int:
        """Calculates data size in bytes."""
        try:
            if hasattr(data, 'nbytes'):
                return data.nbytes
            elif hasattr(data, 'size'):
                return data.size * 4  # Assume float32
            elif isinstance(data, (list, tuple)):
                return len(data) * 4  # Estimation
            else:
                return 0
        except:
            return 0
    
    def _reconstruct_from_codes(self, codes: Any, codebooks: Any) -> Any:
        """Reconstructs original data from VQ codes."""
        try:
            # Simplified implementation of reconstrucción
            if hasattr(codes, 'shape') and codebooks is not None:
                # Reconstruct using codebook lookup
                reconstructed = []
                
                for i, codebook in enumerate(codebooks):
                    if i < codes.shape[-1]:
                        code_indices = codes[..., i]
                        reconstructed_part = codebook[code_indices]
                        reconstructed.append(reconstructed_part)
                
                if reconstructed:
                    import numpy as np
                    return np.concatenate(reconstructed, axis=-1)
            
            return codes  # Fallback
            
        except Exception as e:
            logger.error(f"VQbit reconstruction failed: {e}")
            return codes

class BitNetQuantizationMethod(QuantizationMethod):
    """BitNet quantization method."""
    
    def quantize(self, data: Any) -> QuantizationResult:
        """Quantizes using BitNet (1-bit)."""
        start_time = time.time()
        
        try:
            import numpy as np
            
            if hasattr(data, 'numpy'):
                np_data = data.numpy()
            else:
                np_data = np.array(data)
            
            # BitNet: cuantización a -1, 0, +1
            threshold = np.std(np_data) * 0.5
            quantized = np.sign(np_data)
            quantized[np.abs(np_data) < threshold] = 0
            
            # Calcular statistics
            original_size = np_data.nbytes
            quantized_size = quantized.nbytes // 4  # Aproximación para 3-level quantization
            
            quantization_time = (time.time() - start_time) * 1000
            
            return QuantizationResult(
                quantized_data={'data': quantized, 'threshold': threshold},
                compression_ratio=original_size / quantized_size,
                accuracy_retention=0.80,  # BitNet típicamente retiene ~80%
                quantization_time_ms=quantization_time,
                memory_savings_mb=(original_size - quantized_size) / (1024 * 1024),
                metadata={'method': 'bitnet', 'threshold': threshold}
            )
            
        except Exception as e:
            logger.error(f"BitNet quantization failed: {e}")
            return self._create_error_result(data, start_time, str(e))
    
    def dequantize(self, quantized_data: Any) -> Any:
        """Dequantizes BitNet data."""
        if isinstance(quantized_data, dict) and 'data' in quantized_data:
            return quantized_data['data']  # BitNet es directo
        return quantized_data
    
    def estimate_compression_ratio(self, data_shape: Tuple[int, ...]) -> float:
        """Estimates compression ratio for BitNet."""
        # BitNet: ~8x compression (32-bit -> ~4-bit effective)
        return 8.0

class INT8QuantizationMethod(QuantizationMethod):
    """INT8 quantization method."""
    
    def quantize(self, data: Any) -> QuantizationResult:
        """Quantizes to INT8."""
        start_time = time.time()
        
        try:
            import numpy as np
            
            if hasattr(data, 'numpy'):
                np_data = data.numpy()
            else:
                np_data = np.array(data)
            
            # Symmetric quantization to INT8
            abs_max = np.max(np.abs(np_data))
            scale = abs_max / 127.0
            quantized = np.round(np_data / scale).astype(np.int8)
            
            original_size = np_data.nbytes
            quantized_size = quantized.nbytes + 4  # +4 para scale
            
            quantization_time = (time.time() - start_time) * 1000
            
            return QuantizationResult(
                quantized_data={'data': quantized, 'scale': scale},
                compression_ratio=original_size / quantized_size,
                accuracy_retention=0.95,  # INT8 retiene ~95% típicamente
                quantization_time_ms=quantization_time,
                memory_savings_mb=(original_size - quantized_size) / (1024 * 1024),
                metadata={'method': 'int8', 'scale': scale}
            )
            
        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
            return self._create_error_result(data, start_time, str(e))
    
    def dequantize(self, quantized_data: Any) -> Any:
        """Dequantizes INT8 data."""
        try:
            if isinstance(quantized_data, dict) and 'data' in quantized_data:
                data = quantized_data['data']
                scale = quantized_data.get('scale', 1.0)
                return data.astype(np.float32) * scale
            return quantized_data
        except Exception as e:
            logger.error(f"INT8 dequantization failed: {e}")
            return quantized_data
    
    def estimate_compression_ratio(self, data_shape: Tuple[int, ...]) -> float:
        """Estimates compression ratio for INT8."""
        # INT8: 4x compression (32-bit -> 8-bit)
        return 4.0

class Float16QuantizationMethod(QuantizationMethod):
    """Float16 quantization method."""
    
    def quantize(self, data: Any) -> QuantizationResult:
        """Quantizes to Float16."""
        start_time = time.time()
        
        try:
            import numpy as np
            
            if hasattr(data, 'numpy'):
                np_data = data.numpy()
            else:
                np_data = np.array(data)
            
            # Convert to float16
            quantized = np_data.astype(np.float16)
            
            original_size = np_data.nbytes
            quantized_size = quantized.nbytes
            
            quantization_time = (time.time() - start_time) * 1000
            
            return QuantizationResult(
                quantized_data=quantized,
                compression_ratio=original_size / quantized_size,
                accuracy_retention=0.98,  # Float16 retiene ~98%
                quantization_time_ms=quantization_time,
                memory_savings_mb=(original_size - quantized_size) / (1024 * 1024),
                metadata={'method': 'float16'}
            )
            
        except Exception as e:
            logger.error(f"Float16 quantization failed: {e}")
            return self._create_error_result(data, start_time, str(e))
    
    def dequantize(self, quantized_data: Any) -> Any:
        """Dequantizes Float16 data."""
        try:
            import numpy as np
            if hasattr(quantized_data, 'astype'):
                return quantized_data.astype(np.float32)
            return quantized_data
        except Exception as e:
            logger.error(f"Float16 dequantization failed: {e}")
            return quantized_data
    
    def estimate_compression_ratio(self, data_shape: Tuple[int, ...]) -> float:
        """Estimates compression ratio for Float16."""
        # Float16: 2x compression (32-bit -> 16-bit)
        return 2.0

    def _create_error_result(self, data: Any, start_time: float, error_msg: str) -> QuantizationResult:
        """Creates an error result."""
        return QuantizationResult(
            quantized_data=data,
            compression_ratio=1.0,
            accuracy_retention=1.0,
            quantization_time_ms=(time.time() - start_time) * 1000,
            memory_savings_mb=0.0,
            metadata={'method': 'error', 'error': error_msg}
        )

@register_adapter_decorator(
    adapter_type=AdapterType.QUANTIZATION,
    priority=80,
    capabilities=["multi_method_quantization", "automatic_selection", "quality_preservation"],
    metadata={"version": "1.0", "supports_all_quantization_types": True}
)
class QuantizationAdapter(BaseAdapter):
    """
    Unified quantization adapter that automatically selects
    the best method according to performance and quality requirements.
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self.quantization_methods: Dict[QuantizationType, QuantizationMethod] = {}
        self.method_performance_cache: Dict[QuantizationType, Dict[str, float]] = {}
        self.auto_selection_enabled = True
        
    def _initialize_impl(self) -> bool:
        """Initializes quantization methods."""
        try:
            # Create default configurations for each method
            default_configs = {
                QuantizationType.VQBIT: QuantizationConfig(
                    QuantizationType.VQBIT,
                    QuantizationQuality.BALANCED,
                    custom_parameters={'codebook_size': 64, 'num_codebooks': 8}
                ),
                QuantizationType.BITNET: QuantizationConfig(
                    QuantizationType.BITNET,
                    QuantizationQuality.MAXIMUM_COMPRESSION
                ),
                QuantizationType.INT8: QuantizationConfig(
                    QuantizationType.INT8,
                    QuantizationQuality.HIGH_QUALITY
                ),
                QuantizationType.FLOAT16: QuantizationConfig(
                    QuantizationType.FLOAT16,
                    QuantizationQuality.HIGH_QUALITY
                )
            }
            
            # Initialize methods
            method_classes = {
                QuantizationType.VQBIT: VQbitQuantizationMethod,
                QuantizationType.BITNET: BitNetQuantizationMethod,
                QuantizationType.INT8: INT8QuantizationMethod,
                QuantizationType.FLOAT16: Float16QuantizationMethod
            }
            
            for qtype, method_class in method_classes.items():
                try:
                    config = default_configs.get(qtype)
                    if config:
                        method = method_class(config)
                        self.quantization_methods[qtype] = method
                        self.method_performance_cache[qtype] = {}
                        self.logger.info(f"Initialized {qtype.value} quantization method")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize {qtype.value} method: {e}")
            
            if not self.quantization_methods:
                self.logger.error("No quantization methods could be initialized")
                return False
            
            self.logger.info(f"Quantization adapter initialized with {len(self.quantization_methods)} methods")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize quantization adapter: {e}")
            return False
    
    def _execute_impl(self, 
                     operation: str = "quantize", 
                     data: Any = None,
                     quantization_config: Optional[QuantizationConfig] = None,
                     **kwargs) -> Any:
        """Executes quantization operation."""
        
        if operation == "quantize":
            return self._execute_quantization(data, quantization_config, **kwargs)
        elif operation == "dequantize":
            return self._execute_dequantization(data, **kwargs)
        elif operation == "select_method":
            return self._select_best_method(data, quantization_config, **kwargs)
        elif operation == "benchmark":
            return self._benchmark_methods(data, **kwargs)
        elif operation == "get_methods":
            return self._get_available_methods()
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    def _execute_quantization(self, 
                            data: Any, 
                            config: Optional[QuantizationConfig] = None,
                            **kwargs) -> QuantizationResult:
        """Executes data quantization."""
        if data is None:
            raise ValueError("Data is required for quantization")
        
        # Select method automatically if no configuration specified
        if config is None or self.auto_selection_enabled:
            selected_method_type = self._select_best_method(data, config, **kwargs)
            if selected_method_type in self.quantization_methods:
                method = self.quantization_methods[selected_method_type]
            else:
                # Fallback to first available method
                method = next(iter(self.quantization_methods.values()))
        else:
            method = self.quantization_methods.get(config.quantization_type)
            if method is None:
                raise ValueError(f"Quantization method {config.quantization_type} not available")
        
        # Calibrate if necessary
        if config and config.enable_calibration and not method.calibrated:
            calibration_data = kwargs.get('calibration_data', data)
            method.calibrate(calibration_data)
        
        # Execute quantization
        start_time = time.time()
        result = method.quantize(data)
        execution_time = time.time() - start_time
        
        # Update performance cache
        method_type = config.quantization_type if config else QuantizationType.VQBIT
        self._update_performance_cache(method_type, result, execution_time)
        
        return result
    
    def _execute_dequantization(self, quantized_data: Any, **kwargs) -> Any:
        """Executes data dequantization."""
        if quantized_data is None:
            raise ValueError("Quantized data is required for dequantization")
        
        # Try to determine method from metadata
        method_name = None
        if isinstance(quantized_data, dict) and 'metadata' in quantized_data:
            method_name = quantized_data['metadata'].get('method')
        
        # Map method name to type
        method_type = self._map_method_name_to_type(method_name)
        
        if method_type and method_type in self.quantization_methods:
            method = self.quantization_methods[method_type]
            return method.dequantize(quantized_data)
        else:
            # Try generic dequantization
            self.logger.warning(f"Unknown quantization method: {method_name}, attempting generic dequantization")
            return self._generic_dequantization(quantized_data)
    
    def _select_best_method(self, 
                          data: Any, 
                          config: Optional[QuantizationConfig] = None,
                          **kwargs) -> QuantizationType:
        """Selects the best quantization method for the given data."""
        
        # Si se especifica un tipo específico, usarlo
        if config and config.quantization_type in self.quantization_methods:
            return config.quantization_type
        
        # Automatic selection criteria
        data_size = self._estimate_data_size(data)
        quality_requirement = config.quality_level if config else QuantizationQuality.BALANCED
        target_compression = config.target_compression_ratio if config else None
        
        # Scoring of available methods
        method_scores = {}
        
        for method_type, method in self.quantization_methods.items():
            score = self._calculate_method_score(
                method_type, method, data, data_size, quality_requirement, target_compression
            )
            method_scores[method_type] = score
        
        # Select method with highest score
        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            self.logger.info(f"Selected quantization method: {best_method.value} (score: {method_scores[best_method]:.2f})")
            return best_method
        else:
            # Default fallback
            return QuantizationType.FLOAT16
    
    def _calculate_method_score(self, 
                              method_type: QuantizationType,
                              method: QuantizationMethod,
                              data: Any,
                              data_size: int,
                              quality_requirement: QuantizationQuality,
                              target_compression: Optional[float]) -> float:
        """Calculates quantization method score."""
        score = 0.0
        
        # Factor de calidad vs compresión
        quality_scores = {
            QuantizationQuality.LOSSLESS: 1.0,
            QuantizationQuality.HIGH_QUALITY: 0.8,
            QuantizationQuality.BALANCED: 0.6,
            QuantizationQuality.MAXIMUM_COMPRESSION: 0.4
        }
        
        method_quality_compatibility = {
            QuantizationType.FLOAT16: {
                QuantizationQuality.LOSSLESS: 0.9,
                QuantizationQuality.HIGH_QUALITY: 1.0,
                QuantizationQuality.BALANCED: 0.8,
                QuantizationQuality.MAXIMUM_COMPRESSION: 0.3
            },
            QuantizationType.INT8: {
                QuantizationQuality.LOSSLESS: 0.7,
                QuantizationQuality.HIGH_QUALITY: 0.9,
                QuantizationQuality.BALANCED: 1.0,
                QuantizationQuality.MAXIMUM_COMPRESSION: 0.8
            },
            QuantizationType.VQBIT: {
                QuantizationQuality.LOSSLESS: 0.5,
                QuantizationQuality.HIGH_QUALITY: 0.7,
                QuantizationQuality.BALANCED: 1.0,
                QuantizationQuality.MAXIMUM_COMPRESSION: 0.9
            },
            QuantizationType.BITNET: {
                QuantizationQuality.LOSSLESS: 0.2,
                QuantizationQuality.HIGH_QUALITY: 0.4,
                QuantizationQuality.BALANCED: 0.7,
                QuantizationQuality.MAXIMUM_COMPRESSION: 1.0
            }
        }
        
        # Score based en compatibilidad de calidad
        quality_compatibility = method_quality_compatibility.get(method_type, {})
        quality_score = quality_compatibility.get(quality_requirement, 0.5)
        score += quality_score * 40  # 40% del score
        
        # Score based en compresión objetivo
        if target_compression:
            try:
                data_shape = data.shape if hasattr(data, 'shape') else (data_size,)
                estimated_compression = method.estimate_compression_ratio(data_shape)
                compression_diff = abs(estimated_compression - target_compression) / target_compression
                compression_score = max(0, 1.0 - compression_diff)
                score += compression_score * 30  # 30% del score
            except:
                score += 15  # Score neutral si no se puede estimar
        else:
            score += 20  # Score por defecto
        
        # Score based on performance histórico
        performance_cache = self.method_performance_cache.get(method_type, {})
        if performance_cache:
            avg_time = performance_cache.get('avg_time_ms', 100)
            avg_compression = performance_cache.get('avg_compression_ratio', 1.0)
            avg_accuracy = performance_cache.get('avg_accuracy_retention', 0.9)
            
            # Normalizar métricas y combinar
            time_score = max(0, 1.0 - (avg_time / 1000))  # Penalizar tiempos > 1s
            compression_score = min(avg_compression / 10.0, 1.0)  # Normalizar a 10x max
            accuracy_score = avg_accuracy
            
            performance_score = (time_score + compression_score + accuracy_score) / 3
            score += performance_score * 20  # 20% del score
        else:
            score += 10  # Score neutral para methods sin historial
        
        # Score based en disponibilidad del method
        if method_type == QuantizationType.VQBIT:
            # VQbit puede no estar siempre disponible
            score += 5
        elif method_type == QuantizationType.FLOAT16:
            # Float16 es muy compatible
            score += 10
        else:
            score += 7
        
        return score
    
    def _benchmark_methods(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Runs benchmark of all available methods."""
        benchmark_results = {}
        
        for method_type, method in self.quantization_methods.items():
            try:
                # Create default configuration para benchmark
                config = QuantizationConfig(
                    method_type,
                    QuantizationQuality.BALANCED
                )
                
                # Execute quantization
                start_time = time.time()
                result = method.quantize(data)
                execution_time = (time.time() - start_time) * 1000
                
                benchmark_results[method_type.value] = {
                    'compression_ratio': result.compression_ratio,
                    'accuracy_retention': result.accuracy_retention,
                    'execution_time_ms': execution_time,
                    'memory_savings_mb': result.memory_savings_mb,
                    'success': True
                }
                
            except Exception as e:
                benchmark_results[method_type.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        return {
            'benchmark_results': benchmark_results,
            'data_info': {
                'size_estimate': self._estimate_data_size(data),
                'shape': getattr(data, 'shape', 'unknown')
            }
        }
    
    def _get_available_methods(self) -> Dict[str, Any]:
        """Gets information about available methods."""
        methods_info = {}
        
        for method_type, method in self.quantization_methods.items():
            performance_cache = self.method_performance_cache.get(method_type, {})
            
            methods_info[method_type.value] = {
                'available': True,
                'calibrated': method.calibrated,
                'estimated_compression_ratios': {
                    'small_data': method.estimate_compression_ratio((1000,)),
                    'medium_data': method.estimate_compression_ratio((10000, 512)),
                    'large_data': method.estimate_compression_ratio((100000, 1024))
                },
                'performance_history': performance_cache,
                'quality_compatibility': self._get_quality_compatibility(method_type)
            }
        
        return {
            'available_methods': methods_info,
            'total_methods': len(self.quantization_methods),
            'auto_selection_enabled': self.auto_selection_enabled
        }
    
    def _update_performance_cache(self, 
                                method_type: QuantizationType, 
                                result: QuantizationResult, 
                                execution_time: float):
        """Updates method performance cache."""
        cache = self.method_performance_cache[method_type]
        
        # Actualizar métricas con promedio móvil
        alpha = 0.1  # Factor de suavizado
        
        if 'avg_time_ms' in cache:
            cache['avg_time_ms'] = cache['avg_time_ms'] * (1 - alpha) + execution_time * 1000 * alpha
        else:
            cache['avg_time_ms'] = execution_time * 1000
        
        if 'avg_compression_ratio' in cache:
            cache['avg_compression_ratio'] = cache['avg_compression_ratio'] * (1 - alpha) + result.compression_ratio * alpha
        else:
            cache['avg_compression_ratio'] = result.compression_ratio
        
        if 'avg_accuracy_retention' in cache:
            cache['avg_accuracy_retention'] = cache['avg_accuracy_retention'] * (1 - alpha) + result.accuracy_retention * alpha
        else:
            cache['avg_accuracy_retention'] = result.accuracy_retention
        
        cache['last_updated'] = time.time()
        cache['usage_count'] = cache.get('usage_count', 0) + 1
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estimates data size."""
        try:
            if hasattr(data, 'nbytes'):
                return data.nbytes
            elif hasattr(data, 'size'):
                return data.size * 4  # Assume float32
            elif isinstance(data, (list, tuple)):
                return len(data) * 4
            else:
                return 1000  # Estimation por defecto
        except:
            return 1000
    
    def _map_method_name_to_type(self, method_name: Optional[str]) -> Optional[QuantizationType]:
        """Maps method name to QuantizationType."""
        if not method_name:
            return None
        
        name_mapping = {
            'vqbit': QuantizationType.VQBIT,
            'bitnet': QuantizationType.BITNET,
            'int8': QuantizationType.INT8,
            'float16': QuantizationType.FLOAT16,
            'fallback_int8': QuantizationType.INT8
        }
        
        return name_mapping.get(method_name.lower())
    
    def _generic_dequantization(self, quantized_data: Any) -> Any:
        """Generic dequantization for unknown data."""
        try:
            if isinstance(quantized_data, dict):
                if 'data' in quantized_data:
                    data = quantized_data['data']
                    scale = quantized_data.get('scale', 1.0)
                    offset = quantized_data.get('offset', 0.0)
                    
                    # Try generic dequantization
                    if hasattr(data, 'astype'):
                        return data.astype(np.float32) * scale + offset
                    else:
                        return data
                else:
                    return quantized_data
            else:
                return quantized_data
        except Exception as e:
            self.logger.error(f"Generic dequantization failed: {e}")
            return quantized_data
    
    def _get_quality_compatibility(self, method_type: QuantizationType) -> Dict[str, float]:
        """Gets quality compatibility for a method."""
        compatibility_map = {
            QuantizationType.FLOAT16: {
                'lossless': 0.9, 'high_quality': 1.0, 'balanced': 0.8, 'maximum_compression': 0.3
            },
            QuantizationType.INT8: {
                'lossless': 0.7, 'high_quality': 0.9, 'balanced': 1.0, 'maximum_compression': 0.8
            },
            QuantizationType.VQBIT: {
                'lossless': 0.5, 'high_quality': 0.7, 'balanced': 1.0, 'maximum_compression': 0.9
            },
            QuantizationType.BITNET: {
                'lossless': 0.2, 'high_quality': 0.4, 'balanced': 0.7, 'maximum_compression': 1.0
            }
        }
        
        return compatibility_map.get(method_type, {})
    
    # Métodos de conveniencia
    
    def quantize(self, 
                data: Any, 
                method: Optional[QuantizationType] = None,
                quality: QuantizationQuality = QuantizationQuality.BALANCED,
                **kwargs) -> QuantizationResult:
        """Convenience method for quantization."""
        config = None
        if method:
            config = QuantizationConfig(method, quality)
        
        return self.execute("quantize", data=data, quantization_config=config, **kwargs)
    
    def dequantize(self, quantized_data: Any) -> Any:
        """Convenience method for dequantization."""
        return self.execute("dequantize", data=quantized_data)
    
    def benchmark(self, data: Any) -> Dict[str, Any]:
        """Convenience method for benchmark."""
        return self.execute("benchmark", data=data)
    
    def set_auto_selection(self, enabled: bool):
        """Enables or disables automatic method selection."""
        self.auto_selection_enabled = enabled
        self.logger.info(f"Auto-selection {'enabled' if enabled else 'disabled'}")


# Global adapter instance
quantization_adapter = QuantizationAdapter()

# Funciones de utilidad globales
def quantize_data(data: Any, 
                 method: Optional[QuantizationType] = None,
                 quality: QuantizationQuality = QuantizationQuality.BALANCED) -> QuantizationResult:
    """Global function to quantize data."""
    return quantization_adapter.quantize(data, method, quality)

def dequantize_data(quantized_data: Any) -> Any:
    """Global function to dequantize data."""
    return quantization_adapter.dequantize(quantized_data)

def get_quantization_info():
    """Gets information about available quantization methods."""
    return {
        'adapter_status': quantization_adapter.get_status().value,
        'available_methods': quantization_adapter.execute("get_methods"),
        'adapter_metrics': quantization_adapter.get_metrics().__dict__
    }