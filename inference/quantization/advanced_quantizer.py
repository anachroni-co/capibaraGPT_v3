"""
Advanced INT8/INT4 Quantizer for Meta-Consensus System
Enhanced quantization with support for INT4, mixed precision, and dynamic quantization
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Union, List, Callable
from pathlib import Path
import json
import time
from enum import Enum

logger = logging.getLogger(__name__)

# JAX imports with fallbacks
try:
    from capibara.jax import jax, numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    logger.warning("JAX not available - using NumPy fallbacks")
    JAX_AVAILABLE = False
    import numpy as jnp


class QuantizationMode(Enum):
    """Quantization modes for different use cases."""
    INT8_ONLY = "int8_only"
    INT4_ONLY = "int4_only"
    MIXED_PRECISION = "mixed_precision"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"


class QuantizationStrategy(Enum):
    """Quantization strategies."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    PER_CHANNEL = "per_channel"
    PER_TOKEN = "per_token"
    BLOCK_WISE = "block_wise"


@dataclass
class AdvancedQuantizationConfig:
    """Advanced configuration for INT8/INT4 quantization."""
    
    # Core quantization settings
    mode: QuantizationMode = QuantizationMode.MIXED_PRECISION
    strategy: QuantizationStrategy = QuantizationStrategy.PER_CHANNEL
    
    # INT8 settings
    int8_weight_quantization: bool = True
    int8_activation_quantization: bool = True
    int8_percentile: float = 99.9
    int8_clip_ratio: float = 1.0
    
    # INT4 settings
    int4_weight_quantization: bool = True
    int4_activation_quantization: bool = False  # Usually too aggressive
    int4_percentile: float = 99.5
    int4_block_size: int = 128  # Block size for INT4 quantization
    int4_grouping: bool = True  # Group quantization for better accuracy
    
    # Mixed precision settings
    sensitive_layers: List[str] = field(default_factory=lambda: [
        'attention', 'layer_norm', 'embedding', 'output'
    ])
    int4_layers: List[str] = field(default_factory=lambda: [
        'mlp', 'dense', 'linear'
    ])
    
    # Dynamic quantization
    dynamic_threshold: float = 0.1  # Threshold for switching quantization levels
    calibration_samples: int = 512
    calibration_method: str = "entropy"  # entropy, percentile, mse
    
    # Memory optimization
    enable_weight_sharing: bool = True
    enable_scale_fusion: bool = True
    enable_zero_point_folding: bool = True
    
    # Hardware-specific optimizations
    target_hardware: str = "tpu_v6"  # tpu_v6, arm_axion, cuda
    enable_kernel_fusion: bool = True
    vectorization_width: int = 16
    
    # Quality preservation
    accuracy_threshold: float = 0.95  # Minimum accuracy retention
    outlier_detection: bool = True
    outlier_threshold: float = 6.0  # Standard deviations
    
    # Caching and persistence
    cache_calibration_data: bool = True
    cache_quantization_params: bool = True
    quantization_cache_dir: str = "cache/quantization"


class AdvancedQuantizer:
    """Advanced quantizer supporting INT8/INT4 with mixed precision."""
    
    def __init__(self, config: AdvancedQuantizationConfig):
        self.config = config
        self.calibration_data = {}
        self.quantization_params = {}
        self.layer_sensitivities = {}
        self.performance_stats = {
            'memory_saved': 0.0,
            'speed_improvement': 0.0,
            'accuracy_retention': 1.0,
            'quantization_time': 0.0
        }
        
        # Initialize cache directory
        Path(config.quantization_cache_dir).mkdir(parents=True, exist_ok=True)
    
    def quantize_model(self, model_params: Dict[str, Any], 
                      calibration_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Quantize model with advanced INT8/INT4 techniques."""
        start_time = time.time()
        
        logger.info(f"🔧 Starting advanced quantization (mode: {self.config.mode.value})")
        
        # Step 1: Analyze model sensitivity
        if calibration_data is not None:
            self.analyze_layer_sensitivity(model_params, calibration_data)
        
        # Step 2: Determine quantization strategy per layer
        layer_strategies = self.determine_layer_strategies(model_params)
        
        # Step 3: Quantize parameters
        quantized_params = {}
        original_size = 0
        quantized_size = 0
        
        for layer_name, params in model_params.items():
            strategy = layer_strategies.get(layer_name, self.config.strategy)
            
            if 'weight' in params:
                original_size += params['weight'].nbytes
                quantized_weight, quant_params = self.quantize_weights(
                    params['weight'], layer_name, strategy
                )
                quantized_params[layer_name] = {'weight': quantized_weight}
                quantized_size += quantized_weight.nbytes
                
                # Store quantization parameters
                self.quantization_params[f"{layer_name}_weight"] = quant_params
            
            # Handle biases (usually keep in higher precision)
            if 'bias' in params:
                original_size += params['bias'].nbytes
                if self.should_quantize_bias(layer_name):
                    quantized_bias, bias_params = self.quantize_bias(params['bias'])
                    quantized_params[layer_name]['bias'] = quantized_bias
                    quantized_size += quantized_bias.nbytes
                    self.quantization_params[f"{layer_name}_bias"] = bias_params
                else:
                    quantized_params[layer_name]['bias'] = params['bias']
                    quantized_size += params['bias'].nbytes
        
        # Calculate performance metrics
        self.performance_stats['memory_saved'] = 1.0 - (quantized_size / original_size)
        self.performance_stats['quantization_time'] = time.time() - start_time
        
        logger.info(f"✅ Quantization complete - Memory saved: {self.performance_stats['memory_saved']:.2%}")
        
        return quantized_params
    
    def analyze_layer_sensitivity(self, model_params: Dict[str, Any], 
                                calibration_data: np.ndarray) -> Dict[str, float]:
        """Analyze sensitivity of each layer to quantization."""
        logger.info("🔍 Analyzing layer sensitivity...")
        
        sensitivities = {}
        
        for layer_name, params in model_params.items():
            if 'weight' in params:
                weight = params['weight']
                
                # Calculate weight distribution statistics
                weight_std = np.std(weight)
                weight_range = np.max(weight) - np.min(weight)
                weight_outliers = np.sum(np.abs(weight) > weight_std * self.config.outlier_threshold)
                
                # Calculate sensitivity score
                sensitivity = (weight_range / weight_std) * (1 + weight_outliers / weight.size)
                sensitivities[layer_name] = sensitivity
                
                logger.debug(f"Layer {layer_name}: sensitivity = {sensitivity:.4f}")
        
        self.layer_sensitivities = sensitivities
        return sensitivities
    
    def determine_layer_strategies(self, model_params: Dict[str, Any]) -> Dict[str, QuantizationStrategy]:
        """Determine quantization strategy for each layer."""
        strategies = {}
        
        for layer_name in model_params.keys():
            # Check if layer is in sensitive layers list
            is_sensitive = any(sensitive in layer_name.lower() 
                             for sensitive in self.config.sensitive_layers)
            
            # Check if layer is suitable for INT4
            is_int4_suitable = any(int4_layer in layer_name.lower() 
                                 for int4_layer in self.config.int4_layers)
            
            # Get sensitivity score
            sensitivity = self.layer_sensitivities.get(layer_name, 1.0)
            
            if self.config.mode == QuantizationMode.MIXED_PRECISION:
                if is_sensitive or sensitivity > 2.0:
                    strategies[layer_name] = QuantizationStrategy.ASYMMETRIC  # More careful
                elif is_int4_suitable and sensitivity < 0.5:
                    strategies[layer_name] = QuantizationStrategy.BLOCK_WISE  # INT4 friendly
                else:
                    strategies[layer_name] = self.config.strategy
            else:
                strategies[layer_name] = self.config.strategy
        
        return strategies
    
    def quantize_weights(self, weights: np.ndarray, layer_name: str, 
                        strategy: QuantizationStrategy) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize weights with specified strategy."""
        
        # Determine quantization precision
        use_int4 = self.should_use_int4(layer_name, weights)
        precision = 4 if use_int4 else 8
        
        if strategy == QuantizationStrategy.PER_CHANNEL:
            return self.quantize_per_channel(weights, precision)
        elif strategy == QuantizationStrategy.BLOCK_WISE:
            return self.quantize_block_wise(weights, precision)
        elif strategy == QuantizationStrategy.ASYMMETRIC:
            return self.quantize_asymmetric(weights, precision)
        else:
            return self.quantize_symmetric(weights, precision)
    
    def should_use_int4(self, layer_name: str, weights: np.ndarray) -> bool:
        """Determine if layer should use INT4 quantization."""
        if not self.config.int4_weight_quantization:
            return False
        
        # Check if layer is in INT4 suitable list
        is_int4_suitable = any(int4_layer in layer_name.lower() 
                             for int4_layer in self.config.int4_layers)
        
        # Check weight distribution
        weight_std = np.std(weights)
        weight_range = np.max(weights) - np.min(weights)
        distribution_score = weight_range / (weight_std + 1e-8)
        
        # Use INT4 if layer is suitable and distribution is not too complex
        return is_int4_suitable and distribution_score < 10.0
    
    def quantize_per_channel(self, weights: np.ndarray, precision: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Per-channel quantization with INT4/INT8 support."""
        if len(weights.shape) < 2:
            return self.quantize_symmetric(weights, precision)
        
        # Quantize along output channels (axis=0)
        output_channels = weights.shape[0]
        quantized_weights = np.zeros_like(weights, dtype=np.int8 if precision == 8 else np.int8)
        scales = np.zeros(output_channels, dtype=np.float32)
        zero_points = np.zeros(output_channels, dtype=np.int8 if precision == 8 else np.int8)
        
        qmin = -(2**(precision-1))
        qmax = 2**(precision-1) - 1
        
        for ch in range(output_channels):
            channel_weights = weights[ch]
            
            # Calculate scale and zero point
            w_min = np.min(channel_weights)
            w_max = np.max(channel_weights)
            
            # Apply percentile clipping for INT4
            if precision == 4:
                percentile = self.config.int4_percentile
                w_min = np.percentile(channel_weights, (100 - percentile) / 2)
                w_max = np.percentile(channel_weights, (100 + percentile) / 2)
            
            scale = (w_max - w_min) / (qmax - qmin)
            zero_point = qmin - w_min / scale
            zero_point = np.clip(np.round(zero_point), qmin, qmax)
            
            # Quantize
            quantized_channel = np.clip(
                np.round(channel_weights / scale + zero_point), 
                qmin, qmax
            ).astype(np.int8)
            
            quantized_weights[ch] = quantized_channel
            scales[ch] = scale
            zero_points[ch] = zero_point
        
        params = {
            'scales': scales,
            'zero_points': zero_points,
            'precision': precision,
            'strategy': 'per_channel'
        }
        
        return quantized_weights, params
    
    def quantize_block_wise(self, weights: np.ndarray, precision: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Block-wise quantization for INT4 optimization."""
        block_size = self.config.int4_block_size
        
        # Reshape weights for block processing
        original_shape = weights.shape
        weights_flat = weights.flatten()
        
        # Pad to block size
        padding = (block_size - len(weights_flat) % block_size) % block_size
        if padding > 0:
            weights_flat = np.pad(weights_flat, (0, padding), mode='constant')
        
        num_blocks = len(weights_flat) // block_size
        weights_blocked = weights_flat.reshape(num_blocks, block_size)
        
        quantized_blocks = np.zeros_like(weights_blocked, dtype=np.int8)
        scales = np.zeros(num_blocks, dtype=np.float32)
        zero_points = np.zeros(num_blocks, dtype=np.int8)
        
        qmin = -(2**(precision-1))
        qmax = 2**(precision-1) - 1
        
        for block_idx in range(num_blocks):
            block = weights_blocked[block_idx]
            
            # Calculate block statistics
            b_min = np.min(block)
            b_max = np.max(block)
            
            scale = (b_max - b_min) / (qmax - qmin) if b_max != b_min else 1.0
            zero_point = qmin - b_min / scale if scale != 0 else 0
            zero_point = np.clip(np.round(zero_point), qmin, qmax)
            
            # Quantize block
            quantized_block = np.clip(
                np.round(block / scale + zero_point),
                qmin, qmax
            ).astype(np.int8)
            
            quantized_blocks[block_idx] = quantized_block
            scales[block_idx] = scale
            zero_points[block_idx] = zero_point
        
        # Reshape back to original
        quantized_flat = quantized_blocks.flatten()
        if padding > 0:
            quantized_flat = quantized_flat[:-padding]
        
        quantized_weights = quantized_flat.reshape(original_shape)
        
        params = {
            'scales': scales,
            'zero_points': zero_points,
            'block_size': block_size,
            'original_shape': original_shape,
            'precision': precision,
            'strategy': 'block_wise'
        }
        
        return quantized_weights, params
    
    def quantize_symmetric(self, weights: np.ndarray, precision: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Symmetric quantization (zero point = 0)."""
        qmax = 2**(precision-1) - 1
        
        # Calculate scale
        w_max = np.max(np.abs(weights))
        if precision == 4:
            w_max = np.percentile(np.abs(weights), self.config.int4_percentile)
        
        scale = w_max / qmax if w_max != 0 else 1.0
        
        # Quantize
        quantized_weights = np.clip(
            np.round(weights / scale),
            -qmax, qmax
        ).astype(np.int8)
        
        params = {
            'scale': scale,
            'zero_point': 0,
            'precision': precision,
            'strategy': 'symmetric'
        }
        
        return quantized_weights, params
    
    def quantize_asymmetric(self, weights: np.ndarray, precision: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Asymmetric quantization with zero point."""
        qmin = -(2**(precision-1))
        qmax = 2**(precision-1) - 1
        
        w_min = np.min(weights)
        w_max = np.max(weights)
        
        # Apply percentile clipping for better range utilization
        if precision == 4:
            percentile = self.config.int4_percentile
            w_min = np.percentile(weights, (100 - percentile) / 2)
            w_max = np.percentile(weights, (100 + percentile) / 2)
        
        scale = (w_max - w_min) / (qmax - qmin) if w_max != w_min else 1.0
        zero_point = qmin - w_min / scale if scale != 0 else 0
        zero_point = np.clip(np.round(zero_point), qmin, qmax)
        
        # Quantize
        quantized_weights = np.clip(
            np.round(weights / scale + zero_point),
            qmin, qmax
        ).astype(np.int8)
        
        params = {
            'scale': scale,
            'zero_point': zero_point,
            'precision': precision,
            'strategy': 'asymmetric'
        }
        
        return quantized_weights, params
    
    def should_quantize_bias(self, layer_name: str) -> bool:
        """Determine if bias should be quantized (usually not recommended)."""
        # Generally, biases should remain in higher precision
        return False
    
    def quantize_bias(self, bias: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize bias (if needed)."""
        # Use INT32 for bias to maintain precision
        return bias.astype(np.int32), {'precision': 32, 'strategy': 'identity'}
    
    def dequantize_weights(self, quantized_weights: np.ndarray, 
                          params: Dict[str, Any]) -> np.ndarray:
        """Dequantize weights back to float32."""
        strategy = params['strategy']
        
        if strategy == 'per_channel':
            scales = params['scales']
            zero_points = params['zero_points']
            
            dequantized = np.zeros_like(quantized_weights, dtype=np.float32)
            for ch in range(quantized_weights.shape[0]):
                dequantized[ch] = (quantized_weights[ch] - zero_points[ch]) * scales[ch]
            
            return dequantized
        
        elif strategy == 'block_wise':
            scales = params['scales']
            zero_points = params['zero_points']
            block_size = params['block_size']
            original_shape = params['original_shape']
            
            weights_flat = quantized_weights.flatten()
            
            # Pad if necessary
            padding = (block_size - len(weights_flat) % block_size) % block_size
            if padding > 0:
                weights_flat = np.pad(weights_flat, (0, padding), mode='constant')
            
            num_blocks = len(weights_flat) // block_size
            weights_blocked = weights_flat.reshape(num_blocks, block_size)
            
            dequantized_blocks = np.zeros_like(weights_blocked, dtype=np.float32)
            for block_idx in range(num_blocks):
                dequantized_blocks[block_idx] = ((weights_blocked[block_idx] - zero_points[block_idx]) * 
                                               scales[block_idx])
            
            dequantized_flat = dequantized_blocks.flatten()
            if padding > 0:
                dequantized_flat = dequantized_flat[:-padding]
            
            return dequantized_flat.reshape(original_shape)
        
        else:  # symmetric or asymmetric
            scale = params['scale']
            zero_point = params['zero_point']
            
            return (quantized_weights - zero_point) * scale
    
    def get_memory_footprint(self, quantized_params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate memory footprint of quantized model."""
        footprint = {}
        total_size = 0
        
        for layer_name, params in quantized_params.items():
            layer_size = 0
            if 'weight' in params:
                layer_size += params['weight'].nbytes
            if 'bias' in params:
                layer_size += params['bias'].nbytes
            
            footprint[layer_name] = layer_size
            total_size += layer_size
        
        footprint['total'] = total_size
        return footprint
    
    def benchmark_quantization(self, original_params: Dict[str, Any], 
                             quantized_params: Dict[str, Any]) -> Dict[str, float]:
        """Benchmark quantization performance."""
        # Calculate compression ratio
        original_size = sum(
            sum(param.nbytes for param in layer_params.values())
            for layer_params in original_params.values()
        )
        
        quantized_size = sum(
            sum(param.nbytes for param in layer_params.values())
            for layer_params in quantized_params.values()
        )
        
        compression_ratio = original_size / quantized_size
        memory_saved = 1.0 - (quantized_size / original_size)
        
        return {
            'compression_ratio': compression_ratio,
            'memory_saved': memory_saved,
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'size_reduction_mb': (original_size - quantized_size) / (1024 * 1024)
        }
    
    def save_quantization_config(self, filepath: str):
        """Save quantization configuration and parameters."""
        config_data = {
            'config': {
                'mode': self.config.mode.value,
                'strategy': self.config.strategy.value,
                'int8_weight_quantization': self.config.int8_weight_quantization,
                'int4_weight_quantization': self.config.int4_weight_quantization,
                'int4_block_size': self.config.int4_block_size,
                'sensitive_layers': self.config.sensitive_layers,
                'int4_layers': self.config.int4_layers
            },
            'quantization_params': self.quantization_params,
            'layer_sensitivities': self.layer_sensitivities,
            'performance_stats': self.performance_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        logger.info(f"✅ Quantization config saved to {filepath}")
    
    def load_quantization_config(self, filepath: str):
        """Load quantization configuration and parameters."""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        self.quantization_params = config_data.get('quantization_params', {})
        self.layer_sensitivities = config_data.get('layer_sensitivities', {})
        self.performance_stats = config_data.get('performance_stats', {})
        
        logger.info(f"✅ Quantization config loaded from {filepath}")


def create_advanced_quantizer(mode: QuantizationMode = QuantizationMode.MIXED_PRECISION,
                            target_hardware: str = "tpu_v6") -> AdvancedQuantizer:
    """Create advanced quantizer with optimal settings."""
    config = AdvancedQuantizationConfig(
        mode=mode,
        target_hardware=target_hardware,
        int8_weight_quantization=True,
        int4_weight_quantization=True,
        enable_weight_sharing=True,
        enable_scale_fusion=True,
        outlier_detection=True
    )
    
    return AdvancedQuantizer(config)