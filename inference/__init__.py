"""
Inference module for Capibara-6 - Advanced Inference System.

This module provides specialized inference engines optimized for different architectures:
- ARM Axion VM optimizations for Google Cloud
- INT8 quantization for memory efficiency
- TPU v6e-64 optimizations
- KV-cache quantization for long sequences
- Memory-efficient inference strategies

Components:
- ARMInferenceEngine: ARM-optimized inference with Axion VM support
- QuantizedInferenceEngine: INT8 quantized inference engine
- Quantization System: Complete quantization pipeline
"""

import logging


from .arm_optimized_inference import ARMInferenceEngine
from .quantization import (
    WeightQuantizer,
    KVCacheCalibrator,
    QuantizedInferenceEngine,
    create_weight_quantizer,
    create_kv_calibrator,
    quantize_model_for_deployment,
    calibrate_kv_cache
)


logger = logging.getLogger(__name__)

# Component availability flags
ARM_INFERENCE_AVAILABLE = False
QUANTIZED_INFERENCE_AVAILABLE = False
QUANTIZATION_SYSTEM_AVAILABLE = False

# Try to import inference components
try:
    from .arm_optimized_inference import ARMInferenceEngine
    ARM_INFERENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ARM inference engine not available: {e}")

try:
    from .engines.quantized_engine import QuantizedInferenceEngine, QuantizedEngineConfig
    QUANTIZED_INFERENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Quantized inference engine not available: {e}")

try:
    from .quantization import (
        QuantizationSystem,
        get_quantization_system,
        get_quantization_status,
        INT8Quantizer,
        KVCacheINT8,
        CalibrationEngine
    )
    QUANTIZATION_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Quantization system not available: {e}")


class UnifiedInferenceSystem:
    """
    Unified inference system for Capibara-6.
    
    Orchestrates different inference engines and provides
    automatic engine selection based on model and hardware.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Inference engines
        self.arm_engine = None
        self.quantized_engine = None
        self.quantization_system = None
        
        # System state
        self.available_engines = self._detect_available_engines()
        self.active_engine = None
        
        self._initialize_engines()
        
        logger.info("🚀 Unified Inference System initialized")
    
    def _detect_available_engines(self) -> Dict[str, bool]:
        """Detect available inference engines."""
        return {
            'arm_inference': ARM_INFERENCE_AVAILABLE,
            'quantized_inference': QUANTIZED_INFERENCE_AVAILABLE,
            'quantization_system': QUANTIZATION_SYSTEM_AVAILABLE
        }
    
    def _initialize_engines(self):
        """Initialize available inference engines."""
        
        # Initialize ARM engine
        if ARM_INFERENCE_AVAILABLE:
            try:
                self.arm_engine = ARMInferenceEngine()
                logger.info("✅ ARM inference engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ARM engine: {e}")
        
        # Initialize quantization system
        if QUANTIZATION_SYSTEM_AVAILABLE:
            try:
                quant_config = self.config.get('quantization', {})
                self.quantization_system = get_quantization_system(quant_config)
                logger.info("✅ Quantization system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize quantization system: {e}")
    
    def create_inference_engine(self, model_path: str, 
                              engine_type: str = "auto",
                              **kwargs) -> Any:
        """
        Create an inference engine for the specified model.
        
        Args:
            model_path: Path to the model
            engine_type: Type of engine ('auto', 'arm', 'quantized', 'baseline')
            **kwargs: Additional engine configuration
            
        Returns:
            Configured inference engine
        """
        
        if engine_type == "auto":
            engine_type = self._select_optimal_engine(model_path, **kwargs)
        
        if engine_type == "quantized" and QUANTIZED_INFERENCE_AVAILABLE:
            # Create quantized engine
            from .engines.quantized_engine import QuantizedEngineConfig
            
            engine_config = QuantizedEngineConfig(
                model_path=model_path,
                **kwargs
            )
            
            engine = QuantizedInferenceEngine(engine_config)
            engine.load_model(model_path)
            
            self.active_engine = engine
            logger.info(f"✅ Created quantized inference engine for {model_path}")
            return engine
        
        elif engine_type == "arm" and ARM_INFERENCE_AVAILABLE:
            # Create ARM engine
            engine = ARMInferenceEngine()
            engine.load_model(model_path)
            
            self.active_engine = engine
            logger.info(f"✅ Created ARM inference engine for {model_path}")
            return engine
        
        else:
            # Fallback to available engine
            if QUANTIZED_INFERENCE_AVAILABLE:
                return self.create_inference_engine(model_path, "quantized", **kwargs)
            elif ARM_INFERENCE_AVAILABLE:
                return self.create_inference_engine(model_path, "arm", **kwargs)
            else:
                raise RuntimeError("No inference engines available")
    
    def _select_optimal_engine(self, model_path: str, **kwargs) -> str:
        """Select optimal engine based on model and hardware."""
        
        # Check for quantization preferences
        if kwargs.get('use_quantization', True) and QUANTIZED_INFERENCE_AVAILABLE:
            return "quantized"
        
        # Check for ARM optimization
        if kwargs.get('target_hardware') == 'arm_axion' and ARM_INFERENCE_AVAILABLE:
            return "arm"
        
        # Default preferences
        if QUANTIZED_INFERENCE_AVAILABLE:
            return "quantized"
        elif ARM_INFERENCE_AVAILABLE:
            return "arm"
        else:
            raise RuntimeError("No suitable inference engine available")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive inference system status."""
        
        status = {
            'available_engines': self.available_engines,
            'active_engine_type': type(self.active_engine).__name__ if self.active_engine else None,
            'system_health': self._get_system_health()
        }
        
        # Add quantization system status
        if self.quantization_system:
            status['quantization_status'] = self.quantization_system.get_system_status()
        
        return status
    
    def _get_system_health(self) -> str:
        """Get overall system health."""
        available_count = sum(self.available_engines.values())
        
        if available_count >= 2:
            return 'excellent'
        elif available_count >= 1:
            return 'good'
        else:
            return 'unavailable'


# Global inference system instance
_inference_system: Optional[UnifiedInferenceSystem] = None


def get_inference_system(config: Optional[Dict[str, Any]] = None) -> UnifiedInferenceSystem:
    """Get or create the global inference system instance."""
    global _inference_system
    
    if _inference_system is None:
        _inference_system = UnifiedInferenceSystem(config)
    
    return _inference_system


def create_inference_engine(model_path: str, **kwargs):
    """Convenience function to create an inference engine."""
    system = get_inference_system()
    return system.create_inference_engine(model_path, **kwargs)


def get_inference_status() -> Dict[str, Any]:
    """Get inference system status without creating instance."""
    
    return {
        'engines_available': {
            'arm_inference': ARM_INFERENCE_AVAILABLE,
            'quantized_inference': QUANTIZED_INFERENCE_AVAILABLE,
            'quantization_system': QUANTIZATION_SYSTEM_AVAILABLE
        },
        'total_available': sum([
            ARM_INFERENCE_AVAILABLE,
            QUANTIZED_INFERENCE_AVAILABLE,
            QUANTIZATION_SYSTEM_AVAILABLE
        ]),
        'system_ready': any([
            ARM_INFERENCE_AVAILABLE,
            QUANTIZED_INFERENCE_AVAILABLE
        ])
    }


# Export components based on availability
__all__ = [
    # Core system
    'UnifiedInferenceSystem',
    'get_inference_system',
    'create_inference_engine',
    'get_inference_status',
    
    # Component availability flags
    'ARM_INFERENCE_AVAILABLE',
    'QUANTIZED_INFERENCE_AVAILABLE', 
    'QUANTIZATION_SYSTEM_AVAILABLE'
]

# Conditionally export available components
if ARM_INFERENCE_AVAILABLE:
    __all__.append('ARMInferenceEngine')

if QUANTIZED_INFERENCE_AVAILABLE:
    __all__.extend(['QuantizedInferenceEngine', 'QuantizedEngineConfig'])

if QUANTIZATION_SYSTEM_AVAILABLE:
    __all__.extend([
        'QuantizationSystem',
        'INT8Quantizer', 
        'KVCacheINT8',
        'CalibrationEngine',
        'get_quantization_system',
        'get_quantization_status'
    ])

# Version and metadata
__version__ = "2.0.0"  # Updated for quantization support
__author__ = "Capibara-6 Team"
__description__ = "Advanced Inference System with INT8 Quantization for Capibara-6"

# Log initialization
logger.info(f"Inference module initialized (v{__version__})")
logger.info(f"Available inference engines: {sum([ARM_INFERENCE_AVAILABLE, QUANTIZED_INFERENCE_AVAILABLE, QUANTIZATION_SYSTEM_AVAILABLE])}/3")

if sum([ARM_INFERENCE_AVAILABLE, QUANTIZED_INFERENCE_AVAILABLE]) >= 2:
    logger.info("🚀 Advanced inference capabilities ready")
elif any([ARM_INFERENCE_AVAILABLE, QUANTIZED_INFERENCE_AVAILABLE]):
    logger.info("⚡ Basic inference capabilities available")
else:
    logger.warning("⚠️ Limited inference capabilities")

# Available components for export
__all__ = [
    'ARMInferenceEngine',
    'WeightQuantizer',
    'KVCacheCalibrator', 
    'QuantizedInferenceEngine',
    'create_weight_quantizer',
    'create_kv_calibrator',
    'quantize_model_for_deployment',
    'calibrate_kv_cache',
]
