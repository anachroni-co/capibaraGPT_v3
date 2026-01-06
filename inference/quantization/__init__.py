"""
Quantization Module for Capibara-6

Advanced INT8 quantization system with:
- Post-Training Quantization (PTQ) for weights
- KV-cache INT8 quantization
- TPU v6e-64 and ARM Axion VM optimizations
- Integration with Capibara-6 inference system

Components:
- INT8Quantizer: Main quantization engine
- QuantizedLayers: JAX/Flax quantized neural network layers
- KVCacheINT8: Quantized key-value cache management
- CalibrationEngine: Automatic calibration system
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Component availability flags
QUANTIZATION_AVAILABLE = False
QUANTIZED_LAYERS_AVAILABLE = False
KV_CACHE_INT8_AVAILABLE = False
CALIBRATION_AVAILABLE = False

# Try to import quantization components
try:
    from .int8_quantizer import INT8Quantizer, QuantizationConfig
    QUANTIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"INT8 quantizer not available: {e}")

try:
    from .quantized_layers import QuantDense, QuantAttention, QuantEmbedding
    QUANTIZED_LAYERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Quantized layers not available: {e}")

try:
    from .kv_cache_int8 import KVCacheINT8, KVCacheConfig
    KV_CACHE_INT8_AVAILABLE = True
except ImportError as e:
    logger.warning(f"KV-cache INT8 not available: {e}")

try:
    from .calibration import CalibrationEngine, CalibrationConfig
    CALIBRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Calibration engine not available: {e}")


class QuantizationSystem:
    """
    Unified quantization system for Capibara-6.
    
    Orchestrates all quantization components and provides
    a unified interface for model quantization operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Quantization components
        self.quantizer = None
        self.kv_cache = None
        self.calibration_engine = None
        
        # System state
        self.is_initialized = False
        self.available_components = self._detect_available_components()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("⚡ Quantization System initialized")
    
    def _detect_available_components(self) -> Dict[str, bool]:
        """Detect available quantization components."""
        return {
            'quantizer': QUANTIZATION_AVAILABLE,
            'quantized_layers': QUANTIZED_LAYERS_AVAILABLE,
            'kv_cache_int8': KV_CACHE_INT8_AVAILABLE,
            'calibration': CALIBRATION_AVAILABLE
        }
    
    def _initialize_components(self):
        """Initialize available quantization components."""
        
        # Initialize main quantizer
        if QUANTIZATION_AVAILABLE:
            try:
                quant_config = QuantizationConfig(**self.config.get('quantization', {}))
                self.quantizer = INT8Quantizer(quant_config)
                logger.info("✅ INT8 quantizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize quantizer: {e}")
        
        # Initialize KV-cache
        if KV_CACHE_INT8_AVAILABLE:
            try:
                kv_config = KVCacheConfig(**self.config.get('kv_cache', {}))
                self.kv_cache = KVCacheINT8(kv_config)
                logger.info("✅ KV-cache INT8 initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize KV-cache: {e}")
        
        # Initialize calibration engine
        if CALIBRATION_AVAILABLE:
            try:
                calib_config = CalibrationConfig(**self.config.get('calibration', {}))
                self.calibration_engine = CalibrationEngine(calib_config)
                logger.info("✅ Calibration engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize calibration: {e}")
        
        self.is_initialized = True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive quantization system status."""
        
        return {
            'initialized': self.is_initialized,
            'available_components': self.available_components,
            'active_components': {
                'quantizer': self.quantizer is not None,
                'kv_cache': self.kv_cache is not None,
                'calibration_engine': self.calibration_engine is not None
            },
            'total_components': sum(self.available_components.values()),
            'system_health': self._get_system_health()
        }
    
    def _get_system_health(self) -> str:
        """Get overall quantization system health status."""
        
        if not self.is_initialized:
            return 'not_initialized'
        
        available_count = sum(self.available_components.values())
        
        if available_count >= 3:
            return 'excellent'
        elif available_count >= 2:
            return 'good'
        elif available_count >= 1:
            return 'limited'
        else:
            return 'unavailable'
    
    def quantize_model(self, model_params: Dict[str, Any], 
                      calibration_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Quantize a model using the available quantization components.
        
        Args:
            model_params: Model parameters to quantize
            calibration_data: Optional calibration dataset
            
        Returns:
            Quantized model parameters
        """
        if not self.quantizer:
            raise RuntimeError("Quantizer not available")
        
        # Perform calibration if needed
        if calibration_data is not None and self.calibration_engine:
            self.calibration_engine.calibrate(model_params, calibration_data)
        
        # Quantize the model
        quantized_params = self.quantizer.quantize_weights(model_params)
        
        return quantized_params


# Global quantization system instance
_quantization_system: Optional[QuantizationSystem] = None


def get_quantization_system(config: Optional[Dict[str, Any]] = None) -> QuantizationSystem:
    """Get or create the global quantization system instance."""
    global _quantization_system
    
    if _quantization_system is None:
        _quantization_system = QuantizationSystem(config)
    
    return _quantization_system


def create_quantization_system(config: Dict[str, Any]) -> QuantizationSystem:
    """Create a new quantization system instance."""
    return QuantizationSystem(config)


def get_quantization_status() -> Dict[str, Any]:
    """Get quantization system status without creating instance."""
    
    return {
        'components_available': {
            'quantizer': QUANTIZATION_AVAILABLE,
            'quantized_layers': QUANTIZED_LAYERS_AVAILABLE,
            'kv_cache_int8': KV_CACHE_INT8_AVAILABLE,
            'calibration': CALIBRATION_AVAILABLE
        },
        'total_available': sum([
            QUANTIZATION_AVAILABLE,
            QUANTIZED_LAYERS_AVAILABLE,
            KV_CACHE_INT8_AVAILABLE,
            CALIBRATION_AVAILABLE
        ]),
        'system_ready': any([
            QUANTIZATION_AVAILABLE,
            QUANTIZED_LAYERS_AVAILABLE,
            KV_CACHE_INT8_AVAILABLE
        ])
    }


# Module exports
__all__ = [
    # Core quantization system
    'QuantizationSystem',
    'get_quantization_system',
    'create_quantization_system',
    'get_quantization_status',
    
    # Component availability flags
    'QUANTIZATION_AVAILABLE',
    'QUANTIZED_LAYERS_AVAILABLE',
    'KV_CACHE_INT8_AVAILABLE',
    'CALIBRATION_AVAILABLE',
    
    # Individual components (if available)
    'INT8Quantizer',
    'QuantDense',
    'QuantAttention',
    'KVCacheINT8',
    'CalibrationEngine',
    
    # Configuration classes
    'QuantizationConfig',
    'KVCacheConfig',
    'CalibrationConfig'
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "Capibara-6 Team"
__description__ = "INT8 Quantization System for Capibara-6"

# Log initialization
logger.info(f"Quantization module initialized (v{__version__})")
logger.info(f"Available quantization components: {sum([QUANTIZATION_AVAILABLE, QUANTIZED_LAYERS_AVAILABLE, KV_CACHE_INT8_AVAILABLE, CALIBRATION_AVAILABLE])}/4")

if sum([QUANTIZATION_AVAILABLE, QUANTIZED_LAYERS_AVAILABLE, KV_CACHE_INT8_AVAILABLE]) >= 2:
    logger.info("⚡ Advanced quantization capabilities ready")
elif any([QUANTIZATION_AVAILABLE, QUANTIZED_LAYERS_AVAILABLE, KV_CACHE_INT8_AVAILABLE]):
    logger.info("🔧 Basic quantization capabilities available")
else:
    logger.warning("⚠️ Limited quantization capabilities")