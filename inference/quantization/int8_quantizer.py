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

"""
INT8 Quantizer - Core Implementation
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False


@dataclass
class QuantizationConfig:
    """Configuration for INT8 quantization."""
    weight_quant_method: str = "symmetric_per_channel"
    weight_percentile: float = 99.9
    weight_clip_ratio: float = 1.0
    skip_layers: List[str] = None
    force_symmetric: List[str] = None
    
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
        Symmetric per-channel quantization.
        
        Args:
            weights: [out_features, in_features]
            percentile: Percentile for scale computation
            
        Returns:
            (quantized_weights_int8, scales_fp16)
        """
        if weights.ndim != 2:
            raise ValueError(f"Expected 2D weight tensor, got {weights.ndim}D")
        
        abs_weights = np.abs(weights)
        scales = np.percentile(abs_weights, percentile, axis=1, keepdims=True)
        scales = scales * self.config.weight_clip_ratio
        scales = np.clip(scales, 1e-8, None)
        
        weights_normalized = weights / scales
        weights_quantized = np.round(weights_normalized)
        weights_quantized = np.clip(weights_quantized, -127, 127).astype(np.int8)
        
        scales_fp16 = scales.squeeze(1).astype(np.float16)
        
        return weights_quantized, scales_fp16
    
    def quantize_asymmetric(self, weights: np.ndarray, 
                            percentile: float = 99.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Asymmetric quantization.
        
        Returns:
            (quantized_weights_int8, scales_fp16, zero_points_int8)
        """
        w_min = np.percentile(weights, 100 - percentile, axis=1, keepdims=True)
        w_max = np.percentile(weights, percentile, axis=1, keepdims=True)
        
        scales = (w_max - w_min) / 255.0
        scales = np.clip(scales, 1e-8, None)
        zero_points = -w_min / scales
        zero_points = np.round(zero_points).astype(np.int8)
        
        weights_quantized = weights / scales + zero_points
        weights_quantized = np.round(weights_quantized)
        weights_quantized = np.clip(weights_quantized, -128, 127).astype(np.int8)
        
        return weights_quantized, scales.squeeze(1).astype(np.float16), zero_points.squeeze(1)
    
    def dequantize_weights(self, weights_int8: np.ndarray, 
                           scales: np.ndarray, 
                           zero_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Dequantize weights back to fp32."""
        weights_fp = weights_int8.astype(np.float32)
        scales_expanded = scales[:, np.newaxis] if scales.ndim == 1 else scales
        
        if zero_points is not None:
            zp_expanded = zero_points[:, np.newaxis] if zero_points.ndim == 1 else zero_points
            return (weights_fp - zp_expanded) * scales_expanded
        
        return weights_fp * scales_expanded


class INT8Quantizer:
    """Main INT8 quantization engine."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.weight_quantizer = WeightQuantizer(config)
        self.quantized_params = {}
        self.quantization_metadata = {}
    
    def should_quantize_layer(self, layer_name: str, layer_params: Dict[str, Any]) -> bool:
        """Determine if a layer should be quantized."""
        for skip_pattern in self.config.skip_layers:
            if skip_pattern in layer_name.lower():
                return False
        
        has_weights = any(
            param_name in ['kernel', 'weight', 'w'] 
            for param_name in layer_params.keys()
        )
        
        if not has_weights:
            return False
        
        total_params = sum(
            np.prod(param.shape) if hasattr(param, 'shape') else 1
            for param in layer_params.values()
        )
        
        return total_params >= 256
    
    def quantize_layer_weights(self, layer_name: str, 
                               layer_params: Dict[str, Any]) -> Dict[str, Any]:
        """Quantize weights for a single layer."""
        quantized_layer = {}
        layer_metadata = {}
        
        for param_name, param_value in layer_params.items():
            if param_name not in ['kernel', 'weight', 'w']:
                quantized_layer[param_name] = param_value
                continue
            
            weights = np.array(param_value) if hasattr(param_value, 'shape') else param_value
            
            if weights.ndim != 2:
                quantized_layer[param_name] = param_value
                continue
            
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
                    'original_shape': weights.shape
                }
            else:
                weights_q, scales, zero_points = self.weight_quantizer.quantize_asymmetric(
                    weights, self.config.weight_percentile
                )
                quantized_layer[f"{param_name}_q"] = weights_q
                quantized_layer[f"{param_name}_scales"] = scales
                quantized_layer[f"{param_name}_zero_points"] = zero_points
                
                layer_metadata[param_name] = {
                    'method': 'asymmetric',
                    'original_shape': weights.shape
                }
        
        self.quantization_metadata[layer_name] = layer_metadata
        return quantized_layer
    
    def quantize_weights(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Quantize all weights in a model."""
        quantized_model = {}
        
        for layer_name, layer_params in model_params.items():
            if self.should_quantize_layer(layer_name, layer_params):
                quantized_model[layer_name] = self.quantize_layer_weights(layer_name, layer_params)
            else:
                quantized_model[layer_name] = layer_params
        
        self.quantized_params = quantized_model
        return quantized_model
    
    def dequantize_layer(self, layer_name: str, quantized_layer: Dict[str, Any]) -> Dict[str, Any]:
        """Dequantize a layer back to floating point."""
        dequantized_layer = {}
        
        for param_name, param_value in quantized_layer.items():
            if param_name.endswith('_q'):
                base_name = param_name[:-2]
                scales_key = f"{base_name}_scales"
                zero_points_key = f"{base_name}_zero_points"
                
                if scales_key in quantized_layer:
                    scales = quantized_layer[scales_key]
                    zero_points = quantized_layer.get(zero_points_key)
                    dequantized_layer[base_name] = self.weight_quantizer.dequantize_weights(
                        param_value, scales, zero_points
                    )
            elif not param_name.endswith('_scales') and not param_name.endswith('_zero_points'):
                dequantized_layer[param_name] = param_value
        
        return dequantized_layer
    
    def save(self, filepath: str):
        """Save quantized model and metadata."""
        if not self.quantized_params:
            raise ValueError("No quantized model to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if PICKLE_AVAILABLE:
            with open(filepath, 'wb') as f:
                pickle.dump(self.quantized_params, f)
        else:
            np.save(filepath.with_suffix('.npy'), self.quantized_params)
        
        metadata = {
            'config': self.config.__dict__,
            'quantization_metadata': self.quantization_metadata
        }
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def load(self, filepath: str) -> Dict[str, Any]:
        """Load quantized model."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.pkl' and PICKLE_AVAILABLE:
            with open(filepath, 'rb') as f:
                self.quantized_params = pickle.load(f)  # nosec B301
        else:
            self.quantized_params = np.load(
                filepath.with_suffix('.npy'), allow_pickle=True
            ).item()
        
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.quantization_metadata = metadata.get('quantization_metadata', {})
        
        return self.quantized_params


def quick_quantize(model_params: Dict[str, Any]) -> Tuple[Dict[str, Any], INT8Quantizer]:
    """Quick model quantization."""
    quantizer = INT8Quantizer(QuantizationConfig())
    return quantizer.quantize_weights(model_params), quantizer
