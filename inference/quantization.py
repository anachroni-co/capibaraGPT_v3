#!/usr/bin/env python3
"""
Quantization Tools for CapibaraGPT Inference

This module provides INT8 quantization capabilities for model weights and KV-cache
calibration, optimized for ARM Axion VM and TPU inference scenarios.
"""

import os
import sys
import glob
import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    weight_percentile: float = 99.9
    kv_percentile: float = 99.5
    target_dtype: str = "int8"
    preserve_float16: bool = True
    
class WeightQuantizer:
    """
    Per-channel symmetric INT8 quantization for model weights.
    
    This class provides efficient weight quantization with minimal accuracy loss,
    suitable for deployment on ARM Axion VMs and other inference environments.
    """
    
    def __init__(self, config: QuantizationConfig):
        """Initialize the weight quantizer."""
        self.config = config
    
    def quantize_per_channel_symmetric(
        self, 
        weights: np.ndarray, 
        percentile: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize weights using per-channel symmetric quantization.
        
        Args:
            weights: Input weight tensor (2D)
            percentile: Percentile for scale computation
            
        Returns:
            Tuple of (quantized_weights, scales)
        """
        percentile = percentile or self.config.weight_percentile
        
        if weights.ndim != 2:
            raise ValueError(f"Expected 2D weight tensor, got shape {weights.shape}")
        
        # Compute per-channel scales using percentile of absolute values
        scales = np.percentile(
            np.abs(weights), 
            percentile, 
            axis=1, 
            keepdims=True
        ) / 127.0
        
        # Avoid division by zero
        scales = np.clip(scales, 1e-8, None)
        
        # Quantize weights
        quantized_weights = np.round(weights / scales).astype(np.int8)
        
        # Return quantized weights and squeezed scales
        return quantized_weights, scales.squeeze(1).astype(np.float16)
    
    def quantize_model_checkpoint(
        self, 
        checkpoint_path: str, 
        output_path: str
    ) -> Dict[str, int]:
        """
        Quantize all eligible layers in a model checkpoint.
        
        Args:
            checkpoint_path: Path to input checkpoint (.npz)
            output_path: Path for quantized checkpoint
            
        Returns:
            Dictionary with quantization statistics
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = self._load_checkpoint(checkpoint_path)
        
        # Quantize eligible weights
        quantized_checkpoint = {}
        stats = {"total_layers": 0, "quantized_layers": 0, "skipped_layers": 0}
        
        for name, array in checkpoint.items():
            stats["total_layers"] += 1
            
            if self._is_quantizable_weight(array):
                # Quantize this layer
                logger.debug(f"Quantizing layer: {name}")
                
                quantized_weights, scales = self.quantize_per_channel_symmetric(
                    array.astype(np.float32)
                )
                
                # Store quantized weights and scales
                quantized_checkpoint[f"{name}.W_q"] = quantized_weights
                quantized_checkpoint[f"{name}.S"] = scales
                stats["quantized_layers"] += 1
                
            else:
                # Keep original array, optionally convert to float16
                if self.config.preserve_float16 and array.dtype not in [np.int8, np.int32, np.int64]:
                    quantized_checkpoint[name] = array.astype(np.float16)
                else:
                    quantized_checkpoint[name] = array
                stats["skipped_layers"] += 1
        
        # Save quantized checkpoint
        self._save_checkpoint(quantized_checkpoint, output_path)
        
        logger.info(f"Quantization complete: {stats['quantized_layers']}/{stats['total_layers']} layers quantized")
        return stats
    
    def _is_quantizable_weight(self, array: np.ndarray) -> bool:
        """Check if an array is suitable for weight quantization."""
        return (
            isinstance(array, np.ndarray) and 
            array.ndim == 2 and 
            array.dtype in [np.float32, np.float16]
        )
    
    def _load_checkpoint(self, path: str) -> Dict[str, np.ndarray]:
        """Load checkpoint from file."""
        try:
            with np.load(path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        except Exception as e:
            logger.error(f"Failed to load checkpoint {path}: {e}")
            raise
    
    def _save_checkpoint(self, checkpoint: Dict[str, np.ndarray], path: str):
        """Save checkpoint to file."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, **checkpoint)
            logger.info(f"Saved quantized checkpoint to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint {path}: {e}")
            raise

class KVCacheCalibrator:
    """
    KV-cache calibration for INT8 quantization during inference.
    
    This class analyzes KV-cache statistics to determine optimal quantization
    scales for key and value tensors in attention mechanisms.
    """
    
    def __init__(self, config: QuantizationConfig):
        """Initialize the KV-cache calibrator."""
        self.config = config
    
    def calibrate_kv_scales(
        self, 
        calibration_files: List[str], 
        output_path: str
    ) -> Dict[str, Dict]:
        """
        Calibrate KV-cache quantization scales from collected statistics.
        
        Args:
            calibration_files: List of KV statistics files (.npz)
            output_path: Output path for calibration results (.json)
            
        Returns:
            Dictionary with calibration results per layer
        """
        logger.info(f"Calibrating KV scales from {len(calibration_files)} files")
        
        layer_scales = {}
        
        for file_path in sorted(calibration_files):
            logger.debug(f"Processing calibration file: {file_path}")
            
            try:
                with np.load(file_path, allow_pickle=False) as data:
                    # Extract layer number
                    layer_id = self._extract_layer_id(data, file_path)
                    
                    # Load K and V tensors
                    K = data["K"]
                    V = data["V"]
                    
                    # Process multi-batch data if needed
                    K_processed, V_processed = self._process_kv_tensors(K, V)
                    
                    # Compute per-head scales (vectorized, no Python loop)
                    # K_processed, V_processed: [heads, seq, dim]
                    # Compute percentile over seq and dim axes for each head
                    k_scales = self._robust_percentile_vectorized(
                        K_processed, self.config.kv_percentile
                    )
                    v_scales = self._robust_percentile_vectorized(
                        V_processed, self.config.kv_percentile
                    )

                    # Store scales for this layer
                    layer_scales[layer_id] = {
                        "sK": k_scales.astype(np.float16),
                        "sV": v_scales.astype(np.float16)
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue
        
        # Convert to JSON-serializable format
        json_scales = {}
        for layer_id, scales in layer_scales.items():
            json_scales[int(layer_id)] = {
                "sK": scales["sK"].astype(float).tolist(),
                "sV": scales["sV"].astype(float).tolist()
            }
        
        # Save calibration results
        self._save_calibration_results(json_scales, output_path)
        
        logger.info(f"KV calibration complete for {len(layer_scales)} layers")
        return json_scales
    
    def _extract_layer_id(self, data: Dict, file_path: str) -> int:
        """Extract layer ID from data or filename."""
        if "layer" in data:
            return int(data["layer"])
        
        # Try to extract from filename
        import re
        match = re.search(r'layer(\d+)', os.path.basename(file_path))
        if match:
            return int(match.group(1))
        
        # Default to 0 if no layer info found
        logger.warning(f"Could not extract layer ID from {file_path}, using 0")
        return 0
    
    def _process_kv_tensors(
        self,
        K: np.ndarray,
        V: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process K and V tensors to handle batch dimensions."""
        if K.ndim == 4:  # [batch, heads, seq, dim]
            # Reshape to concatenate batches along sequence (vectorized, no Python loop)
            # [batch, heads, seq, dim] -> [heads, batch*seq, dim]
            batch, heads, seq, dim = K.shape
            K_processed = K.transpose(1, 0, 2, 3).reshape(heads, batch * seq, dim)
            V_processed = V.transpose(1, 0, 2, 3).reshape(heads, batch * seq, dim)
        else:  # Already in [heads, seq, dim] format
            K_processed = K
            V_processed = V

        return K_processed, V_processed

    def _robust_percentile_vectorized(self, tensor: np.ndarray, percentile: float) -> np.ndarray:
        """Compute robust percentile per head (vectorized over heads)."""
        # tensor: [heads, seq, dim]
        # Flatten seq and dim, compute percentile per head
        num_heads = tensor.shape[0]
        flat = np.abs(tensor).reshape(num_heads, -1)  # [heads, seq*dim]
        return np.percentile(flat, percentile, axis=1)  # [heads]

    def _robust_percentile(self, tensor: np.ndarray, percentile: float) -> float:
        """Compute robust percentile of absolute values."""
        abs_values = np.abs(tensor).reshape(-1)
        return float(np.percentile(abs_values, percentile))
    
    def _save_calibration_results(self, scales: Dict, output_path: str):
        """Save calibration results to JSON file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(scales, f, indent=2)
            logger.info(f"Saved KV calibration results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save calibration results: {e}")
            raise

class QuantizedInferenceEngine:
    """
    Inference engine optimized for quantized models.
    
    This class provides utilities for loading and using quantized models
    with proper dequantization during inference.
    """
    
    def __init__(self, model_path: str, kv_scales_path: Optional[str] = None):
        """
        Initialize the quantized inference engine.
        
        Args:
            model_path: Path to quantized model checkpoint
            kv_scales_path: Optional path to KV-cache scales
        """
        self.model_path = model_path
        self.kv_scales_path = kv_scales_path
        self.model_weights = None
        self.kv_scales = None
        
        self._load_model()
        if kv_scales_path:
            self._load_kv_scales()
    
    def _load_model(self):
        """Load quantized model weights."""
        logger.info(f"Loading quantized model from {self.model_path}")
        try:
            with np.load(self.model_path, allow_pickle=False) as data:
                self.model_weights = {key: data[key] for key in data.files}
            logger.info("Quantized model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            raise
    
    def _load_kv_scales(self):
        """Load KV-cache quantization scales."""
        logger.info(f"Loading KV scales from {self.kv_scales_path}")
        try:
            with open(self.kv_scales_path, "r", encoding="utf-8") as f:
                self.kv_scales = json.load(f)
            logger.info("KV scales loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load KV scales: {e}")
            raise
    
    def dequantize_weights(self, layer_name: str) -> np.ndarray:
        """
        Dequantize weights for a specific layer.
        
        Args:
            layer_name: Name of the layer (without .W_q/.S suffix)
            
        Returns:
            Dequantized weight tensor
        """
        w_q_key = f"{layer_name}.W_q"
        s_key = f"{layer_name}.S"
        
        if w_q_key not in self.model_weights or s_key not in self.model_weights:
            raise KeyError(f"Quantized weights not found for layer {layer_name}")
        
        # Load quantized weights and scales
        W_q = self.model_weights[w_q_key]
        S = self.model_weights[s_key]
        
        # Dequantize: W = W_q * S
        W = W_q.astype(np.float32) * S[:, np.newaxis]
        
        return W
    
    def get_kv_scales(self, layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get KV-cache scales for a specific layer.
        
        Args:
            layer_id: Layer identifier
            
        Returns:
            Tuple of (K_scales, V_scales)
        """
        if self.kv_scales is None:
            raise ValueError("KV scales not loaded")
        
        layer_scales = self.kv_scales.get(str(layer_id))
        if layer_scales is None:
            raise KeyError(f"KV scales not found for layer {layer_id}")
        
        sK = np.array(layer_scales["sK"], dtype=np.float16)
        sV = np.array(layer_scales["sV"], dtype=np.float16)
        
        return sK, sV

# ---------------------------------------------------------------------------
# Slim-specific integration
# ---------------------------------------------------------------------------

QUANTIZATION_AVAILABLE = True  # pure-numpy, always available


@dataclass
class SlimQuantizationStats:
    total_tensors: int
    quantized_tensors: int
    skipped_tensors: int

    def __str__(self) -> str:
        return (
            f"quantized={self.quantized_tensors}/{self.total_tensors}  "
            f"skipped={self.skipped_tensors}"
        )


class SlimQuantizer:
    """INT8 per-channel quantization for Slim model state dicts.

    Works on plain dicts of numpy arrays or torch tensors (if torch is available).
    Returns a *new* state dict with 2-D float tensors replaced by
    ``<name>.W_q`` (int8) + ``<name>.S`` (float16 scales) pairs.
    """

    def __init__(self, config: Optional[QuantizationConfig] = None) -> None:
        self.config = config or QuantizationConfig()
        self._quantizer = WeightQuantizer(self.config)

    def quantize_state_dict(
        self, state_dict: Dict[str, Any]
    ) -> Tuple["SlimQuantizationStats", Dict[str, np.ndarray]]:
        """Quantize a state dict. Returns (stats, quantized_dict)."""
        result: Dict[str, np.ndarray] = {}
        total = quantized = skipped = 0

        for name, tensor in state_dict.items():
            total += 1
            # Accept torch tensors or numpy arrays
            if hasattr(tensor, "detach"):
                arr = tensor.detach().cpu().float().numpy()
            else:
                arr = np.asarray(tensor, dtype=np.float32)

            if self._quantizer._is_quantizable_weight(arr):
                q, scales = self._quantizer.quantize_per_channel_symmetric(arr)
                result[f"{name}.W_q"] = q
                result[f"{name}.S"] = scales
                quantized += 1
            else:
                result[name] = arr
                skipped += 1

        return SlimQuantizationStats(total, quantized, skipped), result

    def apply_to_model(self, model: Any) -> "SlimQuantizationStats":
        """Quantize a torch nn.Module in-place (loads int8 weights back).

        Only replaces weight tensors of ``nn.Linear`` layers.
        Requires torch.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("apply_to_model requires PyTorch") from exc

        total = quantized = skipped = 0
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            w = module.weight.detach().cpu().float().numpy()
            total += 1
            if self._quantizer._is_quantizable_weight(w):
                q, scales = self._quantizer.quantize_per_channel_symmetric(w)
                # Dequantize back into float32 for transparent drop-in use
                w_dq = q.astype(np.float32) * scales[:, np.newaxis]
                with torch.no_grad():
                    module.weight.copy_(torch.from_numpy(w_dq))
                quantized += 1
            else:
                skipped += 1

        return SlimQuantizationStats(total, quantized, skipped)


# Factory functions for easy integration
def create_weight_quantizer(config_dict: Optional[Dict] = None) -> WeightQuantizer:
    """Create a weight quantizer with optional configuration."""
    config_dict = config_dict or {}
    config = QuantizationConfig(**config_dict)
    return WeightQuantizer(config)

def create_kv_calibrator(config_dict: Optional[Dict] = None) -> KVCacheCalibrator:
    """Create a KV-cache calibrator with optional configuration."""
    config_dict = config_dict or {}
    config = QuantizationConfig(**config_dict)
    return KVCacheCalibrator(config)

def quantize_model_for_deployment(
    model_path: str,
    output_path: str,
    config: Optional[Dict] = None
) -> Dict[str, int]:
    """
    Convenience function to quantize a model for deployment.
    
    Args:
        model_path: Path to original model checkpoint
        output_path: Path for quantized model
        config: Optional quantization configuration
        
    Returns:
        Quantization statistics
    """
    logger.info("Quantizing model for deployment")
    
    quantizer = create_weight_quantizer(config)
    stats = quantizer.quantize_model_checkpoint(model_path, output_path)
    
    logger.info("Model quantization for deployment completed")
    return stats

def calibrate_kv_cache(
    calibration_glob: str,
    output_path: str,
    config: Optional[Dict] = None
) -> Dict[str, Dict]:
    """
    Convenience function to calibrate KV-cache quantization.
    
    Args:
        calibration_glob: Glob pattern for calibration files
        output_path: Output path for calibration results
        config: Optional calibration configuration
        
    Returns:
        Calibration results
    """
    logger.info("Calibrating KV-cache quantization")
    
    # Find calibration files
    calibration_files = sorted(glob.glob(calibration_glob))
    if not calibration_files:
        raise ValueError(f"No calibration files found matching pattern: {calibration_glob}")
    
    calibrator = create_kv_calibrator(config)
    results = calibrator.calibrate_kv_scales(calibration_files, output_path)
    
    logger.info("KV-cache calibration completed")
    return results