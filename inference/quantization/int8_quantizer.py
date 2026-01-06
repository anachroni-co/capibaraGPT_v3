"""
INT8 Quantizer for Capibara-6

Post-Training Quantization (PTQ) implementation with:
- Symmetric per-channel weight quantization
- Asymmetric activation quantization
- Calibration-based scale computation
- TPU and ARM optimization support
- JAX/Flax integration

Optimized for Capibara-6 inference on TPU v6e-64 and ARM Axion VMs.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

# JAX imports with fallbacks
try:
    from capibara.jax import jax, numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    logger.warning("JAX not available - using NumPy fallbacks")
    JAX_AVAILABLE = False
    import numpy as jnp

# Optional imports
try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False


@dataclass
class QuantizationConfig:
    """Configuration for INT8 quantization in Capibara-6."""
    
    # Weight quantization
    weight_quantization: bool = True
    weight_quant_method: str = "symmetric_per_channel"  # symmetric_per_channel, asymmetric
    weight_percentile: float = 99.9
    weight_clip_ratio: float = 1.0
    
    # Activation quantization
    activation_quantization: bool = False  # Typically keep in bf16 for TPU
    activation_quant_method: str = "asymmetric"
    activation_percentile: float = 99.5
    
    # KV-cache quantization
    kv_cache_quantization: bool = True
    kv_cache_method: str = "per_head"  # per_head, per_channel
    kv_cache_percentile: float = 99.5
    
    # Calibration
    use_calibration: bool = True
    calibration_samples: int = 512
    calibration_method: str = "percentile"  # percentile, mse, entropy
    
    # Target hardware
    target_hardware: str = "tpu_v6e"  # tpu_v6e, arm_axion, cpu
    memory_efficient: bool = True
    preserve_accuracy: bool = True
    
    # Advanced options
    skip_layers: List[str] = None  # Layers to skip quantization
    force_symmetric: List[str] = None  # Force symmetric quantization
    mixed_precision: bool = True  # Keep some layers in FP16
    
    # Performance tuning
    batch_processing: bool = True
    parallel_quantization: bool = True
    cache_scales: bool = True
    
    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = ['embedding', 'lm_head', 'output_projection']
        if self.force_symmetric is None:
            self.force_symmetric = ['attention.query', 'attention.key', 'attention.value']


class WeightQuantizer:
    """Handles weight quantization operations."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
    
    def quantize_symmetric_per_channel(self, weights: np.ndarray, 
                                     percentile: float = 99.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Symmetric per-channel quantization for weights.
        
        Args:
            weights: Weight tensor [out_features, in_features]
            percentile: Percentile for scale computation
            
        Returns:
            Tuple of (quantized_weights_int8, scales_fp16)
        """
        if weights.ndim != 2:
            raise ValueError(f"Expected 2D weight tensor, got {weights.ndim}D")
        
        # Compute scales per output channel (channel 0)
        abs_weights = np.abs(weights)
        scales = np.percentile(abs_weights, percentile, axis=1, keepdims=True)
        
        # Apply clip ratio
        scales = scales * self.config.weight_clip_ratio
        
        # Avoid division by zero
        scales = np.clip(scales, 1e-8, None)
        
        # Quantize: W_quantized = round(W / scale)
        weights_normalized = weights / scales
        weights_quantized = np.round(weights_normalized)
        
        # Clip to int8 range [-127, 127] (avoid -128 for symmetric)
        weights_quantized = np.clip(weights_quantized, -127, 127).astype(np.int8)
        
        # Scales as fp16 for memory efficiency
        scales_fp16 = scales.squeeze(1).astype(np.float16)
        
        return weights_quantized, scales_fp16
    
    def quantize_asymmetric(self, weights: np.ndarray, 
                          percentile: float = 99.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Asymmetric quantization for weights.
        
        Args:
            weights: Weight tensor
            percentile: Percentile for range computation
            
        Returns:
            Tuple of (quantized_weights_int8, scales_fp16, zero_points_int8)
        """
        # Compute min/max per channel
        w_min = np.percentile(weights, 100 - percentile, axis=1, keepdims=True)
        w_max = np.percentile(weights, percentile, axis=1, keepdims=True)
        
        # Compute scale and zero point
        scales = (w_max - w_min) / 255.0  # 256 levels for int8
        zero_points = -w_min / scales
        zero_points = np.round(zero_points).astype(np.int8)
        
        # Quantize
        weights_quantized = weights / scales + zero_points
        weights_quantized = np.round(weights_quantized)
        weights_quantized = np.clip(weights_quantized, -128, 127).astype(np.int8)
        
        return weights_quantized, scales.squeeze(1).astype(np.float16), zero_points.squeeze(1)
    
    def dequantize_weights(self, weights_int8: np.ndarray, 
                          scales: np.ndarray, 
                          zero_points: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Dequantize weights back to floating point.
        
        Args:
            weights_int8: Quantized weights
            scales: Quantization scales
            zero_points: Zero points (for asymmetric quantization)
            
        Returns:
            Dequantized weights in fp32
        """
        # Convert to float for computation
        weights_fp = weights_int8.astype(np.float32)
        
        # Expand scales to match weight dimensions
        if scales.ndim == 1:
            scales_expanded = scales[:, np.newaxis]
        else:
            scales_expanded = scales
        
        # Dequantize
        if zero_points is not None:
            zero_points_expanded = zero_points[:, np.newaxis] if zero_points.ndim == 1 else zero_points
            weights_dequant = (weights_fp - zero_points_expanded) * scales_expanded
        else:
            weights_dequant = weights_fp * scales_expanded
        
        return weights_dequant


class ActivationQuantizer:
    """Handles activation quantization (typically disabled for TPU)."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.activation_scales = {}
        self.activation_zero_points = {}
    
    def calibrate_activations(self, layer_name: str, activations: np.ndarray):
        """Calibrate activation quantization parameters."""
        if not self.config.activation_quantization:
            return
        
        # Compute activation statistics
        if self.config.activation_quant_method == "asymmetric":
            a_min = np.percentile(activations, 100 - self.config.activation_percentile)
            a_max = np.percentile(activations, self.config.activation_percentile)
            
            scale = (a_max - a_min) / 255.0
            zero_point = -a_min / scale
            
            self.activation_scales[layer_name] = scale
            self.activation_zero_points[layer_name] = int(round(zero_point))
        
        logger.debug(f"Calibrated activations for {layer_name}")
    
    def quantize_activations(self, layer_name: str, activations: np.ndarray) -> np.ndarray:
        """Quantize activations to int8."""
        if not self.config.activation_quantization or layer_name not in self.activation_scales:
            return activations  # Return unchanged
        
        scale = self.activation_scales[layer_name]
        zero_point = self.activation_zero_points[layer_name]
        
        quantized = activations / scale + zero_point
        quantized = np.round(quantized)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        
        return quantized


class INT8Quantizer:
    """
    Main INT8 quantization engine for Capibara-6.
    
    Handles Post-Training Quantization (PTQ) with support for:
    - Weight quantization (symmetric per-channel)
    - Optional activation quantization
    - Calibration-based scale computation
    - TPU and ARM optimizations
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.weight_quantizer = WeightQuantizer(config)
        self.activation_quantizer = ActivationQuantizer(config)
        
        # Quantization state
        self.quantized_params = {}
        self.quantization_metadata = {}
        self.calibration_complete = False
        
        # Performance tracking
        self.stats = {
            'layers_quantized': 0,
            'layers_skipped': 0,
            'original_size_mb': 0.0,
            'quantized_size_mb': 0.0,
            'compression_ratio': 0.0,
            'quantization_time': 0.0
        }
        
        logger.info(f"⚡ INT8Quantizer initialized for {self.config.target_hardware}")
    
    def should_quantize_layer(self, layer_name: str, layer_params: Dict[str, Any]) -> bool:
        """Determine if a layer should be quantized."""
        
        # Skip layers in the skip list
        for skip_pattern in self.config.skip_layers:
            if skip_pattern in layer_name.lower():
                return False
        
        # Check if layer has weights to quantize
        has_weights = any(
            param_name in ['kernel', 'weight', 'w'] 
            for param_name in layer_params.keys()
        )
        
        if not has_weights:
            return False
        
        # Skip very small layers (< 1KB)
        total_params = sum(
            np.prod(param.shape) if hasattr(param, 'shape') else 1
            for param in layer_params.values()
        )
        
        if total_params < 256:  # Skip tiny layers
            return False
        
        return True
    
    def quantize_layer_weights(self, layer_name: str, 
                             layer_params: Dict[str, Any]) -> Dict[str, Any]:
        """Quantize weights for a single layer."""
        quantized_layer = {}
        layer_metadata = {}
        
        for param_name, param_value in layer_params.items():
            if param_name in ['kernel', 'weight', 'w']:
                # Convert to numpy if needed
                if JAX_AVAILABLE and hasattr(param_value, 'shape'):
                    weights = np.array(param_value)
                else:
                    weights = param_value
                
                if weights.ndim != 2:
                    logger.warning(f"Skipping {layer_name}.{param_name}: not 2D tensor")
                    quantized_layer[param_name] = param_value
                    continue
                
                # Choose quantization method
                force_symmetric = any(
                    pattern in layer_name for pattern in self.config.force_symmetric
                )
                
                if self.config.weight_quant_method == "symmetric_per_channel" or force_symmetric:
                    weights_q, scales = self.weight_quantizer.quantize_symmetric_per_channel(
                        weights, self.config.weight_percentile
                    )
                    
                    quantized_layer[f"{param_name}_q"] = weights_q
                    quantized_layer[f"{param_name}_scales"] = scales
                    
                    layer_metadata[param_name] = {
                        'method': 'symmetric_per_channel',
                        'original_shape': weights.shape,
                        'scale_shape': scales.shape,
                        'quantized_dtype': 'int8',
                        'scale_dtype': 'float16'
                    }
                
                else:  # Asymmetric quantization
                    weights_q, scales, zero_points = self.weight_quantizer.quantize_asymmetric(
                        weights, self.config.weight_percentile
                    )
                    
                    quantized_layer[f"{param_name}_q"] = weights_q
                    quantized_layer[f"{param_name}_scales"] = scales
                    quantized_layer[f"{param_name}_zero_points"] = zero_points
                    
                    layer_metadata[param_name] = {
                        'method': 'asymmetric',
                        'original_shape': weights.shape,
                        'quantized_dtype': 'int8',
                        'scale_dtype': 'float16'
                    }
                
                # Calculate compression
                original_size = weights.nbytes
                quantized_size = (weights_q.nbytes + scales.nbytes + 
                                getattr(zero_points, 'nbytes', 0) if 'zero_points' in locals() else 0)
                
                layer_metadata[param_name]['compression_ratio'] = original_size / quantized_size
                
            else:
                # Keep other parameters unchanged (bias, etc.)
                quantized_layer[param_name] = param_value
        
        self.quantization_metadata[layer_name] = layer_metadata
        return quantized_layer
    
    def quantize_weights(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quantize all weights in a model.
        
        Args:
            model_params: Dictionary of model parameters
            
        Returns:
            Dictionary of quantized parameters
        """
        import time
        start_time = time.time()
        
        logger.info(f"🚀 Starting model quantization with {len(model_params)} layers")
        
        quantized_model = {}
        original_size = 0
        quantized_size = 0
        
        for layer_name, layer_params in model_params.items():
            if self.should_quantize_layer(layer_name, layer_params):
                # Quantize the layer
                quantized_layer = self.quantize_layer_weights(layer_name, layer_params)
                quantized_model[layer_name] = quantized_layer
                self.stats['layers_quantized'] += 1
                
                logger.debug(f"✅ Quantized layer: {layer_name}")
            else:
                # Keep layer unchanged
                quantized_model[layer_name] = layer_params
                self.stats['layers_skipped'] += 1
                
                logger.debug(f"⏭️ Skipped layer: {layer_name}")
            
            # Update size statistics
            layer_size = self._calculate_layer_size(layer_params)
            original_size += layer_size
            quantized_size += self._calculate_layer_size(quantized_model[layer_name])
        
        # Update statistics
        self.stats['quantization_time'] = time.time() - start_time
        self.stats['original_size_mb'] = original_size / (1024 * 1024)
        self.stats['quantized_size_mb'] = quantized_size / (1024 * 1024)
        self.stats['compression_ratio'] = original_size / quantized_size if quantized_size > 0 else 1.0
        
        logger.info(f"✅ Quantization completed in {self.stats['quantization_time']:.2f}s")
        logger.info(f"📊 Compression: {self.stats['original_size_mb']:.1f}MB → {self.stats['quantized_size_mb']:.1f}MB")
        logger.info(f"📈 Ratio: {self.stats['compression_ratio']:.2f}x compression")
        logger.info(f"🔢 Layers: {self.stats['layers_quantized']} quantized, {self.stats['layers_skipped']} skipped")
        
        self.quantized_params = quantized_model
        return quantized_model
    
    def dequantize_layer(self, layer_name: str, quantized_layer: Dict[str, Any]) -> Dict[str, Any]:
        """Dequantize a layer back to floating point."""
        dequantized_layer = {}
        
        layer_metadata = self.quantization_metadata.get(layer_name, {})
        
        for param_name, param_value in quantized_layer.items():
            if param_name.endswith('_q'):
                # This is a quantized weight
                base_name = param_name[:-2]  # Remove '_q' suffix
                scales_key = f"{base_name}_scales"
                zero_points_key = f"{base_name}_zero_points"
                
                if scales_key in quantized_layer:
                    scales = quantized_layer[scales_key]
                    zero_points = quantized_layer.get(zero_points_key)
                    
                    # Dequantize
                    dequantized_weights = self.weight_quantizer.dequantize_weights(
                        param_value, scales, zero_points
                    )
                    
                    dequantized_layer[base_name] = dequantized_weights
            
            elif not param_name.endswith('_scales') and not param_name.endswith('_zero_points'):
                # Regular parameter (bias, etc.)
                dequantized_layer[param_name] = param_value
        
        return dequantized_layer
    
    def _calculate_layer_size(self, layer_params: Dict[str, Any]) -> int:
        """Calculate total size of layer parameters in bytes."""
        total_size = 0
        
        for param in layer_params.values():
            if hasattr(param, 'nbytes'):
                total_size += param.nbytes
            elif hasattr(param, 'shape'):
                # Estimate size
                dtype_size = 4  # Assume float32
                if hasattr(param, 'dtype'):
                    if 'int8' in str(param.dtype):
                        dtype_size = 1
                    elif 'float16' in str(param.dtype):
                        dtype_size = 2
                
                total_size += np.prod(param.shape) * dtype_size
        
        return total_size
    
    def save_quantized_model(self, filepath: str):
        """Save quantized model and metadata."""
        if not self.quantized_params:
            raise ValueError("No quantized model to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save quantized parameters
        if PICKLE_AVAILABLE:
            with open(filepath, 'wb') as f:
                pickle.dump(self.quantized_params, f)
        else:
            # Fallback to numpy
            np.save(filepath.with_suffix('.npy'), self.quantized_params)
        
        # Save metadata
        metadata = {
            'config': self.config.__dict__,
            'quantization_metadata': self.quantization_metadata,
            'stats': self.stats
        }
        
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"💾 Quantized model saved to {filepath}")
    
    def load_quantized_model(self, filepath: str) -> Dict[str, Any]:
        """Load quantized model and metadata."""
        filepath = Path(filepath)
        
        # Load parameters
        if filepath.suffix == '.pkl' and PICKLE_AVAILABLE:
            with open(filepath, 'rb') as f:
                self.quantized_params = pickle.load(f)
        else:
            self.quantized_params = np.load(filepath.with_suffix('.npy'), allow_pickle=True).item()
        
        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.quantization_metadata = metadata.get('quantization_metadata', {})
            self.stats = metadata.get('stats', {})
        
        logger.info(f"📁 Quantized model loaded from {filepath}")
        return self.quantized_params
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quantization statistics."""
        return {
            **self.stats,
            'config': self.config.__dict__,
            'metadata_available': bool(self.quantization_metadata),
            'quantized_layers': list(self.quantization_metadata.keys())
        }
    
    def validate_quantization(self, original_params: Dict[str, Any], 
                            tolerance: float = 1e-2) -> Dict[str, Any]:
        """Validate quantization quality by comparing with original."""
        if not self.quantized_params:
            raise ValueError("No quantized model available")
        
        validation_results = {}
        
        for layer_name, original_layer in original_params.items():
            if layer_name not in self.quantized_params:
                continue
            
            quantized_layer = self.quantized_params[layer_name]
            dequantized_layer = self.dequantize_layer(layer_name, quantized_layer)
            
            layer_errors = {}
            
            for param_name, original_param in original_layer.items():
                if param_name in dequantized_layer:
                    dequant_param = dequantized_layer[param_name]
                    
                    # Calculate error metrics
                    mse = np.mean((original_param - dequant_param) ** 2)
                    max_error = np.max(np.abs(original_param - dequant_param))
                    rel_error = mse / (np.mean(original_param ** 2) + 1e-8)
                    
                    layer_errors[param_name] = {
                        'mse': float(mse),
                        'max_error': float(max_error),
                        'relative_error': float(rel_error),
                        'within_tolerance': rel_error < tolerance
                    }
            
            validation_results[layer_name] = layer_errors
        
        return validation_results


# Utility functions
def create_default_quantization_config(target_hardware: str = "tpu_v6e") -> QuantizationConfig:
    """Create default quantization configuration for specific hardware."""
    
    if target_hardware == "tpu_v6e":
        return QuantizationConfig(
            weight_quantization=True,
            weight_quant_method="symmetric_per_channel",
            activation_quantization=False,  # Keep activations in bf16 for TPU
            kv_cache_quantization=True,
            target_hardware="tpu_v6e",
            memory_efficient=True,
            preserve_accuracy=True
        )
    
    elif target_hardware == "arm_axion":
        return QuantizationConfig(
            weight_quantization=True,
            weight_quant_method="symmetric_per_channel", 
            activation_quantization=True,  # Can use int8 activations on ARM
            kv_cache_quantization=True,
            target_hardware="arm_axion",
            memory_efficient=True
        )
    
    else:  # Generic CPU
        return QuantizationConfig(
            weight_quantization=True,
            activation_quantization=False,
            kv_cache_quantization=False,
            target_hardware="cpu"
        )


def quick_quantize_model(model_params: Dict[str, Any], 
                        target_hardware: str = "tpu_v6e") -> Tuple[Dict[str, Any], INT8Quantizer]:
    """Quick model quantization for simple use cases."""
    config = create_default_quantization_config(target_hardware)
    quantizer = INT8Quantizer(config)
    quantized_model = quantizer.quantize_weights(model_params)
    
    return quantized_model, quantizer