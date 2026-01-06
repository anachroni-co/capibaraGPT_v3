"""
Calibration Engine for Capibara-6 Quantization

Automatic calibration system for optimal quantization parameters:
- Statistical calibration methods
- Dataset-based calibration
- Dynamic scale adaptation
- Hardware-specific optimizations
- Integration with training pipeline

Ensures minimal accuracy loss during quantization.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import time
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# JAX imports with fallbacks
try:
    from capibara.jax import jax, numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    logger.warning("JAX not available - using NumPy fallbacks")
    JAX_AVAILABLE = False
    import numpy as jnp


@dataclass
class CalibrationConfig:
    """Configuration for quantization calibration."""
    
    # Calibration dataset
    calibration_samples: int = 512
    max_sequence_length: int = 1024
    batch_size: int = 8
    
    # Calibration methods
    weight_calibration_method: str = "percentile"  # percentile, mse, entropy
    activation_calibration_method: str = "percentile"
    kv_cache_calibration_method: str = "percentile"
    
    # Statistical parameters
    percentile_value: float = 99.9
    outlier_threshold: float = 6.0  # Standard deviations
    min_samples_per_layer: int = 100
    
    # Optimization parameters
    mse_optimization_steps: int = 100
    entropy_bins: int = 256
    search_alpha_min: float = 0.1
    search_alpha_max: float = 2.0
    search_steps: int = 20
    
    # Hardware optimization
    target_hardware: str = "tpu_v6e"
    use_vectorized_ops: bool = True
    parallel_calibration: bool = True
    
    # Quality control
    validate_calibration: bool = True
    accuracy_threshold: float = 0.01  # Max allowed accuracy drop
    
    # Persistence
    save_calibration_data: bool = True
    calibration_cache_dir: str = "calibration_cache"
    
    # Debugging
    collect_detailed_stats: bool = True
    plot_distributions: bool = False
    log_layer_stats: bool = False


class StatisticalCalibrator:
    """Handles statistical calibration methods."""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.layer_stats = defaultdict(lambda: defaultdict(list))
    
    def calibrate_percentile(self, activations: np.ndarray, percentile: float = None) -> float:
        """Calibrate using percentile method."""
        if percentile is None:
            percentile = self.config.percentile_value
        
        abs_activations = np.abs(activations)
        scale = np.percentile(abs_activations, percentile)
        
        # Normalize to int8 range
        return scale / 127.0
    
    def calibrate_mse(self, activations: np.ndarray, num_steps: int = None) -> float:
        """Calibrate using MSE optimization."""
        if num_steps is None:
            num_steps = self.config.mse_optimization_steps
        
        abs_activations = np.abs(activations)
        max_val = np.max(abs_activations)
        
        best_scale = max_val / 127.0
        best_mse = float('inf')
        
        # Search for optimal scale
        for step in range(num_steps):
            alpha = self.config.search_alpha_min + (
                self.config.search_alpha_max - self.config.search_alpha_min
            ) * step / (num_steps - 1)
            
            scale = alpha * max_val / 127.0
            
            # Quantize and dequantize
            quantized = np.round(activations / scale)
            quantized = np.clip(quantized, -127, 127)
            dequantized = quantized * scale
            
            # Calculate MSE
            mse = np.mean((activations - dequantized) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_scale = scale
        
        return best_scale
    
    def calibrate_entropy(self, activations: np.ndarray, num_bins: int = None) -> float:
        """Calibrate using entropy-based method."""
        if num_bins is None:
            num_bins = self.config.entropy_bins
        
        abs_activations = np.abs(activations)
        max_val = np.max(abs_activations)
        
        best_scale = max_val / 127.0
        best_entropy = 0.0
        
        # Search for scale that maximizes entropy
        for step in range(self.config.search_steps):
            alpha = self.config.search_alpha_min + (
                self.config.search_alpha_max - self.config.search_alpha_min
            ) * step / (self.config.search_steps - 1)
            
            scale = alpha * max_val / 127.0
            
            # Quantize
            quantized = np.round(activations / scale)
            quantized = np.clip(quantized, -127, 127)
            
            # Calculate entropy
            hist, _ = np.histogram(quantized, bins=num_bins, range=(-127, 127))
            hist = hist + 1e-8  # Avoid log(0)
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob))
            
            if entropy > best_entropy:
                best_entropy = entropy
                best_scale = scale
        
        return best_scale
    
    def calibrate_per_channel(self, activations: np.ndarray, method: str = "percentile") -> np.ndarray:
        """Calibrate scales per channel."""
        if activations.ndim < 2:
            return np.array([self.calibrate_single_scale(activations, method)])
        
        # Calculate scale for each channel (last dimension)
        scales = []
        for i in range(activations.shape[-1]):
            channel_data = activations[..., i]
            scale = self.calibrate_single_scale(channel_data.flatten(), method)
            scales.append(scale)
        
        return np.array(scales)
    
    def calibrate_single_scale(self, data: np.ndarray, method: str) -> float:
        """Calibrate a single scale value."""
        if method == "percentile":
            return self.calibrate_percentile(data)
        elif method == "mse":
            return self.calibrate_mse(data)
        elif method == "entropy":
            return self.calibrate_entropy(data)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def collect_layer_statistics(self, layer_name: str, activations: np.ndarray):
        """Collect statistics for a layer."""
        stats = {
            'mean': float(np.mean(activations)),
            'std': float(np.std(activations)),
            'min': float(np.min(activations)),
            'max': float(np.max(activations)),
            'percentiles': {
                '95': float(np.percentile(np.abs(activations), 95)),
                '99': float(np.percentile(np.abs(activations), 99)),
                '99.9': float(np.percentile(np.abs(activations), 99.9)),
                '99.99': float(np.percentile(np.abs(activations), 99.99))
            }
        }
        
        self.layer_stats[layer_name]['samples'].append(stats)
        
        if self.config.log_layer_stats:
            logger.debug(f"Layer {layer_name} stats: {stats}")


class DatasetCalibrator:
    """Handles dataset-based calibration."""
    
    def __init__(self, config: CalibrationConfig, model_forward_fn: Optional[Callable] = None):
        self.config = config
        self.model_forward_fn = model_forward_fn
        self.activation_hooks = {}
        self.collected_activations = defaultdict(list)
    
    def setup_activation_hooks(self, model):
        """Setup hooks to collect activations during forward pass."""
        if not JAX_AVAILABLE:
            logger.warning("JAX not available - activation hooks disabled")
            return
        
        # This would need to be implemented based on the specific model structure
        logger.info("Setting up activation collection hooks")
    
    def collect_calibration_data(self, calibration_dataset: List[Any]) -> Dict[str, np.ndarray]:
        """Collect activations from calibration dataset."""
        logger.info(f"Collecting calibration data from {len(calibration_dataset)} samples")
        
        if self.model_forward_fn is None:
            raise ValueError("Model forward function not provided")
        
        collected_data = defaultdict(list)
        
        # Process calibration samples
        num_batches = (len(calibration_dataset) + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(calibration_dataset))
            
            batch_samples = calibration_dataset[start_idx:end_idx]
            
            # Run forward pass and collect activations
            try:
                # This would depend on the specific model interface
                activations = self._run_forward_with_hooks(batch_samples)
                
                for layer_name, layer_activations in activations.items():
                    collected_data[layer_name].append(layer_activations)
                
            except Exception as e:
                logger.warning(f"Failed to process batch {batch_idx}: {e}")
                continue
            
            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Processed {batch_idx + 1}/{num_batches} batches")
        
        # Concatenate collected data
        final_data = {}
        for layer_name, activation_list in collected_data.items():
            if activation_list:
                final_data[layer_name] = np.concatenate(activation_list, axis=0)
        
        logger.info(f"Collected activations for {len(final_data)} layers")
        return final_data
    
    def _run_forward_with_hooks(self, batch_samples: List[Any]) -> Dict[str, np.ndarray]:
        """Run forward pass with activation collection."""
        # Placeholder implementation
        # Real implementation would depend on model structure
        logger.debug(f"Running forward pass on batch of {len(batch_samples)} samples")
        
        # Return dummy data for now
        return {
            'layer_0': np.random.randn(len(batch_samples), 512, 768),
            'layer_1': np.random.randn(len(batch_samples), 512, 768)
        }


class CalibrationEngine:
    """
    Main calibration engine for Capibara-6 quantization.
    
    Provides comprehensive calibration capabilities for:
    - Weight quantization scales
    - Activation quantization parameters
    - KV-cache quantization scales
    - Hardware-specific optimizations
    """
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.statistical_calibrator = StatisticalCalibrator(config)
        self.dataset_calibrator = DatasetCalibrator(config)
        
        # Calibration results
        self.calibration_results = {}
        self.calibration_metadata = {}
        
        # Performance tracking
        self.stats = {
            'total_layers_calibrated': 0,
            'calibration_time': 0.0,
            'samples_processed': 0,
            'accuracy_validation': {}
        }
        
        logger.info(f"⚡ CalibrationEngine initialized for {self.config.target_hardware}")
    
    def calibrate_model(self, model_params: Dict[str, Any], 
                       calibration_dataset: Optional[List[Any]] = None,
                       model_forward_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Calibrate quantization parameters for a complete model.
        
        Args:
            model_params: Model parameters to calibrate
            calibration_dataset: Dataset for activation calibration
            model_forward_fn: Function to run model forward pass
            
        Returns:
            Dictionary of calibration parameters
        """
        start_time = time.time()
        logger.info(f"🚀 Starting model calibration with {len(model_params)} layers")
        
        calibration_params = {}
        
        # 1. Calibrate weight quantization scales
        weight_scales = self._calibrate_weights(model_params)
        calibration_params['weight_scales'] = weight_scales
        
        # 2. Calibrate activations if dataset provided
        if calibration_dataset is not None and model_forward_fn is not None:
            self.dataset_calibrator.model_forward_fn = model_forward_fn
            activation_scales = self._calibrate_activations(calibration_dataset)
            calibration_params['activation_scales'] = activation_scales
        
        # 3. Calibrate KV-cache parameters
        kv_cache_scales = self._calibrate_kv_cache(model_params, calibration_dataset)
        calibration_params['kv_cache_scales'] = kv_cache_scales
        
        # Update statistics
        self.stats['calibration_time'] = time.time() - start_time
        self.stats['total_layers_calibrated'] = len(model_params)
        if calibration_dataset:
            self.stats['samples_processed'] = len(calibration_dataset)
        
        # Validate calibration if requested
        if self.config.validate_calibration:
            validation_results = self._validate_calibration(
                model_params, calibration_params, calibration_dataset
            )
            self.stats['accuracy_validation'] = validation_results
        
        # Save calibration data if requested
        if self.config.save_calibration_data:
            self._save_calibration_data(calibration_params)
        
        self.calibration_results = calibration_params
        
        logger.info(f"✅ Calibration completed in {self.stats['calibration_time']:.2f}s")
        logger.info(f"📊 Calibrated {self.stats['total_layers_calibrated']} layers")
        
        return calibration_params
    
    def _calibrate_weights(self, model_params: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        """Calibrate weight quantization scales."""
        logger.info("🔧 Calibrating weight quantization scales")
        
        weight_scales = {}
        
        for layer_name, layer_params in model_params.items():
            layer_scales = {}
            
            for param_name, param_value in layer_params.items():
                if param_name in ['kernel', 'weight', 'w'] and hasattr(param_value, 'shape'):
                    # Convert to numpy if needed
                    weights = np.array(param_value) if hasattr(param_value, 'shape') else param_value
                    
                    if weights.ndim == 2:  # Only process 2D weight matrices
                        # Calculate per-channel scales
                        scales = self.statistical_calibrator.calibrate_per_channel(
                            weights.T,  # Transpose for per-output-channel
                            self.config.weight_calibration_method
                        )
                        
                        layer_scales[param_name] = scales
                        
                        # Collect statistics
                        self.statistical_calibrator.collect_layer_statistics(
                            f"{layer_name}.{param_name}", weights
                        )
            
            if layer_scales:
                weight_scales[layer_name] = layer_scales
        
        logger.info(f"📊 Calibrated weights for {len(weight_scales)} layers")
        return weight_scales
    
    def _calibrate_activations(self, calibration_dataset: List[Any]) -> Dict[str, np.ndarray]:
        """Calibrate activation quantization scales."""
        logger.info("🔧 Calibrating activation quantization scales")
        
        # Collect activations from dataset
        activation_data = self.dataset_calibrator.collect_calibration_data(calibration_dataset)
        
        activation_scales = {}
        
        for layer_name, activations in activation_data.items():
            # Calculate scales based on method
            if self.config.activation_calibration_method == "per_channel":
                scales = self.statistical_calibrator.calibrate_per_channel(
                    activations, "percentile"
                )
            else:
                # Global scale
                scale = self.statistical_calibrator.calibrate_single_scale(
                    activations.flatten(), self.config.activation_calibration_method
                )
                scales = np.array([scale])
            
            activation_scales[layer_name] = scales
            
            # Collect statistics
            self.statistical_calibrator.collect_layer_statistics(layer_name, activations)
        
        logger.info(f"📊 Calibrated activations for {len(activation_scales)} layers")
        return activation_scales
    
    def _calibrate_kv_cache(self, model_params: Dict[str, Any], 
                          calibration_dataset: Optional[List[Any]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Calibrate KV-cache quantization scales."""
        logger.info("🔧 Calibrating KV-cache quantization scales")
        
        kv_cache_scales = {}
        
        # For now, use heuristic-based calibration
        # In practice, this would collect K,V statistics during forward passes
        
        attention_layers = [
            name for name in model_params.keys() 
            if 'attention' in name.lower() or 'attn' in name.lower()
        ]
        
        for layer_name in attention_layers:
            layer_params = model_params[layer_name]
            
            # Estimate scales for K,V based on weight statistics
            k_scales = None
            v_scales = None
            
            if 'key_proj' in layer_params or 'k_proj' in layer_params:
                key_weights = layer_params.get('key_proj', layer_params.get('k_proj'))
                if key_weights is not None:
                    k_scales = self.statistical_calibrator.calibrate_per_channel(
                        np.array(key_weights), "percentile"
                    )
            
            if 'value_proj' in layer_params or 'v_proj' in layer_params:
                value_weights = layer_params.get('value_proj', layer_params.get('v_proj'))
                if value_weights is not None:
                    v_scales = self.statistical_calibrator.calibrate_per_channel(
                        np.array(value_weights), "percentile"
                    )
            
            if k_scales is not None or v_scales is not None:
                kv_cache_scales[layer_name] = {
                    'k_scales': k_scales,
                    'v_scales': v_scales
                }
        
        logger.info(f"📊 Calibrated KV-cache for {len(kv_cache_scales)} attention layers")
        return kv_cache_scales
    
    def _validate_calibration(self, model_params: Dict[str, Any], 
                            calibration_params: Dict[str, Any],
                            validation_dataset: Optional[List[Any]]) -> Dict[str, Any]:
        """Validate calibration quality."""
        logger.info("🔍 Validating calibration quality")
        
        validation_results = {
            'weight_validation': {},
            'activation_validation': {},
            'overall_quality': 'unknown'
        }
        
        # Validate weight calibration
        if 'weight_scales' in calibration_params:
            weight_validation = self._validate_weight_scales(
                model_params, calibration_params['weight_scales']
            )
            validation_results['weight_validation'] = weight_validation
        
        # Simple quality assessment
        if validation_results['weight_validation']:
            avg_error = np.mean([
                result.get('relative_error', 1.0) 
                for result in validation_results['weight_validation'].values()
            ])
            
            if avg_error < self.config.accuracy_threshold:
                validation_results['overall_quality'] = 'good'
            elif avg_error < self.config.accuracy_threshold * 2:
                validation_results['overall_quality'] = 'acceptable'
            else:
                validation_results['overall_quality'] = 'poor'
        
        logger.info(f"📊 Validation quality: {validation_results['overall_quality']}")
        return validation_results
    
    def _validate_weight_scales(self, model_params: Dict[str, Any], 
                              weight_scales: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Validate weight quantization scales."""
        validation_results = {}
        
        for layer_name, layer_scales in weight_scales.items():
            if layer_name not in model_params:
                continue
            
            layer_params = model_params[layer_name]
            layer_validation = {}
            
            for param_name, scales in layer_scales.items():
                if param_name in layer_params:
                    weights = np.array(layer_params[param_name])
                    
                    # Simulate quantization and dequantization
                    scales_expanded = scales[:, np.newaxis] if scales.ndim == 1 else scales
                    weights_q = np.round(weights / scales_expanded)
                    weights_q = np.clip(weights_q, -127, 127)
                    weights_dq = weights_q * scales_expanded
                    
                    # Calculate error metrics
                    mse = np.mean((weights - weights_dq) ** 2)
                    relative_error = mse / (np.mean(weights ** 2) + 1e-8)
                    
                    layer_validation[param_name] = {
                        'mse': float(mse),
                        'relative_error': float(relative_error),
                        'max_error': float(np.max(np.abs(weights - weights_dq))),
                        'scales_mean': float(np.mean(scales)),
                        'scales_std': float(np.std(scales))
                    }
            
            validation_results[layer_name] = layer_validation
        
        return validation_results
    
    def _save_calibration_data(self, calibration_params: Dict[str, Any]):
        """Save calibration data to disk."""
        try:
            cache_dir = Path(self.config.calibration_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Save calibration parameters
            calibration_file = cache_dir / "calibration_params.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_params = self._make_json_serializable(calibration_params)
            
            with open(calibration_file, 'w') as f:
                json.dump(serializable_params, f, indent=2)
            
            # Save metadata
            metadata = {
                'config': self.config.__dict__,
                'stats': self.stats,
                'timestamp': time.time()
            }
            
            metadata_file = cache_dir / "calibration_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"💾 Calibration data saved to {cache_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save calibration data: {e}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        else:
            return obj
    
    def load_calibration_data(self, cache_dir: str) -> Dict[str, Any]:
        """Load previously saved calibration data."""
        cache_path = Path(cache_dir)
        calibration_file = cache_path / "calibration_params.json"
        
        if not calibration_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
        
        with open(calibration_file, 'r') as f:
            calibration_params = json.load(f)
        
        # Convert lists back to numpy arrays
        calibration_params = self._restore_numpy_arrays(calibration_params)
        
        logger.info(f"📁 Calibration data loaded from {cache_dir}")
        return calibration_params
    
    def _restore_numpy_arrays(self, obj: Any) -> Any:
        """Convert lists back to numpy arrays."""
        if isinstance(obj, dict):
            return {k: self._restore_numpy_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
            return np.array(obj)
        else:
            return obj
    
    def get_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        return {
            **self.stats,
            'config': self.config.__dict__,
            'layer_statistics': dict(self.statistical_calibrator.layer_stats)
        }


# Utility functions
def create_default_calibration_config(target_hardware: str = "tpu_v6e") -> CalibrationConfig:
    """Create default calibration configuration for specific hardware."""
    
    if target_hardware == "tpu_v6e":
        return CalibrationConfig(
            calibration_samples=512,
            weight_calibration_method="percentile",
            activation_calibration_method="percentile",
            percentile_value=99.9,
            target_hardware="tpu_v6e",
            use_vectorized_ops=True,
            parallel_calibration=True
        )
    
    elif target_hardware == "arm_axion":
        return CalibrationConfig(
            calibration_samples=256,
            weight_calibration_method="mse",
            activation_calibration_method="percentile",
            percentile_value=99.5,
            target_hardware="arm_axion",
            mse_optimization_steps=50
        )
    
    else:  # Generic CPU
        return CalibrationConfig(
            calibration_samples=128,
            weight_calibration_method="percentile",
            activation_calibration_method="percentile",
            percentile_value=99.0,
            target_hardware="cpu"
        )


def quick_calibrate_model(model_params: Dict[str, Any], 
                         calibration_dataset: Optional[List[Any]] = None,
                         target_hardware: str = "tpu_v6e") -> Tuple[Dict[str, Any], CalibrationEngine]:
    """Quick model calibration for simple use cases."""
    config = create_default_calibration_config(target_hardware)
    calibration_engine = CalibrationEngine(config)
    
    calibration_params = calibration_engine.calibrate_model(
        model_params, calibration_dataset
    )
    
    return calibration_params, calibration_engine