"""
KV-Cache INT8 Quantization for Capibara-6

Implements INT8 quantization for Key-Value cache in transformer attention:
- Per-head quantization scales
- Dynamic cache management
- Memory-efficient storage
- TPU v6e-64 and ARM Axion VM optimizations
- Integration with attention mechanisms

Optimized for long sequence generation with minimal accuracy loss.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union, List
import time
from collections import defaultdict

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
class KVCacheConfig:
    """Configuration for KV-cache INT8 quantization."""
    
    # Quantization settings
    quantization_method: str = "per_head"  # per_head, per_channel, global
    percentile_calibration: float = 99.5
    use_ema_scales: bool = True
    ema_decay: float = 0.99
    
    # Cache management
    max_sequence_length: int = 8192
    cache_block_size: int = 256  # For paged cache
    prefill_quantization: bool = True
    generation_quantization: bool = True
    
    # Memory optimization
    memory_efficient: bool = True
    lazy_dequantization: bool = True
    cache_compression: bool = True
    
    # Hardware optimization
    target_hardware: str = "tpu_v6e"
    vectorized_ops: bool = True
    parallel_heads: bool = True
    
    # Quality control
    min_scale_value: float = 1e-6
    max_scale_value: float = 1e3
    outlier_threshold: float = 6.0  # Standard deviations
    
    # Debugging and monitoring
    collect_stats: bool = True
    validate_dequantization: bool = False
    log_scale_updates: bool = False


class CacheBlock:
    """Represents a block of quantized KV cache."""
    
    def __init__(self, block_id: int, block_size: int, num_heads: int, head_dim: int):
        self.block_id = block_id
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Storage for quantized cache
        self.k_cache_int8 = np.zeros((block_size, num_heads, head_dim), dtype=np.int8)
        self.v_cache_int8 = np.zeros((block_size, num_heads, head_dim), dtype=np.int8)
        
        # Quantization scales per head
        self.k_scales = np.ones(num_heads, dtype=np.float16)
        self.v_scales = np.ones(num_heads, dtype=np.float16)
        
        # Metadata
        self.sequence_length = 0
        self.is_full = False
        self.last_updated = time.time()
    
    def add_kv(self, k_new: np.ndarray, v_new: np.ndarray, 
               k_scales: np.ndarray, v_scales: np.ndarray, start_pos: int = 0):
        """Add new K,V pairs to the cache block."""
        seq_len = k_new.shape[0]
        end_pos = start_pos + seq_len
        
        if end_pos > self.block_size:
            raise ValueError(f"Cannot fit {seq_len} tokens starting at {start_pos} in block of size {self.block_size}")
        
        # Store quantized K,V
        self.k_cache_int8[start_pos:end_pos] = k_new
        self.v_cache_int8[start_pos:end_pos] = v_new
        
        # Update scales
        self.k_scales = k_scales
        self.v_scales = v_scales
        
        # Update metadata
        self.sequence_length = max(self.sequence_length, end_pos)
        self.is_full = self.sequence_length >= self.block_size
        self.last_updated = time.time()
    
    def get_kv(self, start_pos: int = 0, end_pos: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve and dequantize K,V from cache block."""
        if end_pos is None:
            end_pos = self.sequence_length
        
        # Get quantized data
        k_int8 = self.k_cache_int8[start_pos:end_pos]
        v_int8 = self.v_cache_int8[start_pos:end_pos]
        
        # Dequantize
        k_dequant = k_int8.astype(np.float32) * self.k_scales[np.newaxis, :, np.newaxis]
        v_dequant = v_int8.astype(np.float32) * self.v_scales[np.newaxis, :, np.newaxis]
        
        return k_dequant, v_dequant
    
    def memory_usage_mb(self) -> float:
        """Calculate memory usage in MB."""
        int8_size = self.k_cache_int8.nbytes + self.v_cache_int8.nbytes
        scale_size = self.k_scales.nbytes + self.v_scales.nbytes
        return (int8_size + scale_size) / (1024 * 1024)


class KVQuantizer:
    """Handles quantization operations for K,V tensors."""
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        
        # EMA scales for dynamic adjustment
        self.ema_k_scales = {}  # layer_id -> scales
        self.ema_v_scales = {}
        
        # Statistics collection
        self.stats = defaultdict(lambda: defaultdict(int))
    
    def calibrate_scales(self, k_tensor: np.ndarray, v_tensor: np.ndarray, 
                        layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate quantization scales for K,V tensors.
        
        Args:
            k_tensor: Key tensor [seq_len, num_heads, head_dim]
            v_tensor: Value tensor [seq_len, num_heads, head_dim]
            layer_id: Layer identifier
            
        Returns:
            Tuple of (k_scales, v_scales) per head
        """
        if self.config.quantization_method == "per_head":
            # Calculate scales per head
            k_scales = self._calculate_per_head_scales(k_tensor)
            v_scales = self._calculate_per_head_scales(v_tensor)
        
        elif self.config.quantization_method == "per_channel":
            # Calculate scales per channel (head_dim)
            k_scales = self._calculate_per_channel_scales(k_tensor)
            v_scales = self._calculate_per_channel_scales(v_tensor)
        
        else:  # global
            # Single global scale
            k_scale = self._calculate_global_scale(k_tensor)
            v_scale = self._calculate_global_scale(v_tensor)
            k_scales = np.full(k_tensor.shape[1], k_scale, dtype=np.float16)
            v_scales = np.full(v_tensor.shape[1], v_scale, dtype=np.float16)
        
        # Apply EMA if enabled
        if self.config.use_ema_scales:
            k_scales = self._update_ema_scales(k_scales, layer_id, 'k')
            v_scales = self._update_ema_scales(v_scales, layer_id, 'v')
        
        # Clamp scales to valid range
        k_scales = np.clip(k_scales, self.config.min_scale_value, self.config.max_scale_value)
        v_scales = np.clip(v_scales, self.config.min_scale_value, self.config.max_scale_value)
        
        return k_scales.astype(np.float16), v_scales.astype(np.float16)
    
    def _calculate_per_head_scales(self, tensor: np.ndarray) -> np.ndarray:
        """Calculate quantization scales per attention head."""
        # tensor shape: [seq_len, num_heads, head_dim]
        abs_tensor = np.abs(tensor)
        
        # Calculate percentile per head across seq_len and head_dim
        scales = np.percentile(
            abs_tensor.reshape(tensor.shape[0], tensor.shape[1], -1),
            self.config.percentile_calibration,
            axis=(0, 2)  # Across seq_len and head_dim
        )
        
        # Normalize to int8 range [-127, 127]
        scales = scales / 127.0
        
        return scales
    
    def _calculate_per_channel_scales(self, tensor: np.ndarray) -> np.ndarray:
        """Calculate quantization scales per channel (dimension)."""
        # This is more complex and might not be needed for KV cache
        # For now, fall back to per-head
        return self._calculate_per_head_scales(tensor)
    
    def _calculate_global_scale(self, tensor: np.ndarray) -> float:
        """Calculate single global scale for entire tensor."""
        abs_max = np.percentile(np.abs(tensor), self.config.percentile_calibration)
        return abs_max / 127.0
    
    def _update_ema_scales(self, new_scales: np.ndarray, layer_id: int, kv_type: str) -> np.ndarray:
        """Update scales using exponential moving average."""
        ema_dict = self.ema_k_scales if kv_type == 'k' else self.ema_v_scales
        
        if layer_id not in ema_dict:
            ema_dict[layer_id] = new_scales.copy()
            return new_scales
        
        # EMA update: scale_new = decay * scale_old + (1 - decay) * scale_current
        old_scales = ema_dict[layer_id]
        updated_scales = (self.config.ema_decay * old_scales + 
                         (1 - self.config.ema_decay) * new_scales)
        
        ema_dict[layer_id] = updated_scales
        
        if self.config.log_scale_updates:
            scale_change = np.mean(np.abs(updated_scales - old_scales) / old_scales)
            logger.debug(f"Layer {layer_id} {kv_type.upper()} scales changed by {scale_change:.3%}")
        
        return updated_scales
    
    def quantize_kv(self, k_tensor: np.ndarray, v_tensor: np.ndarray,
                   k_scales: np.ndarray, v_scales: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize K,V tensors to INT8.
        
        Args:
            k_tensor: Key tensor [seq_len, num_heads, head_dim]
            v_tensor: Value tensor [seq_len, num_heads, head_dim]
            k_scales: Per-head scales for K
            v_scales: Per-head scales for V
            
        Returns:
            Tuple of (k_quantized_int8, v_quantized_int8)
        """
        # Expand scales for broadcasting
        k_scales_expanded = k_scales[np.newaxis, :, np.newaxis]  # [1, num_heads, 1]
        v_scales_expanded = v_scales[np.newaxis, :, np.newaxis]
        
        # Quantize: q = round(x / scale)
        k_quantized = np.round(k_tensor / k_scales_expanded)
        v_quantized = np.round(v_tensor / v_scales_expanded)
        
        # Handle outliers before clipping
        if self.config.outlier_threshold > 0:
            k_quantized = self._handle_outliers(k_quantized, self.config.outlier_threshold)
            v_quantized = self._handle_outliers(v_quantized, self.config.outlier_threshold)
        
        # Clip to int8 range
        k_quantized = np.clip(k_quantized, -127, 127).astype(np.int8)
        v_quantized = np.clip(v_quantized, -127, 127).astype(np.int8)
        
        return k_quantized, v_quantized
    
    def _handle_outliers(self, quantized_tensor: np.ndarray, threshold: float) -> np.ndarray:
        """Handle outliers in quantized tensor."""
        # Calculate statistics
        mean = np.mean(quantized_tensor)
        std = np.std(quantized_tensor)
        
        # Clip outliers
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        clipped = np.clip(quantized_tensor, lower_bound, upper_bound)
        
        # Count outliers for statistics
        outliers = np.sum((quantized_tensor < lower_bound) | (quantized_tensor > upper_bound))
        if outliers > 0:
            self.stats['outliers_clipped'] += outliers
        
        return clipped
    
    def dequantize_kv(self, k_int8: np.ndarray, v_int8: np.ndarray,
                     k_scales: np.ndarray, v_scales: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dequantize K,V tensors back to floating point.
        
        Args:
            k_int8: Quantized key tensor
            v_int8: Quantized value tensor
            k_scales: Per-head scales for K
            v_scales: Per-head scales for V
            
        Returns:
            Tuple of (k_dequantized, v_dequantized)
        """
        # Convert to float32 for computation
        k_float = k_int8.astype(np.float32)
        v_float = v_int8.astype(np.float32)
        
        # Expand scales for broadcasting
        k_scales_expanded = k_scales[np.newaxis, :, np.newaxis]
        v_scales_expanded = v_scales[np.newaxis, :, np.newaxis]
        
        # Dequantize: x = q * scale
        k_dequantized = k_float * k_scales_expanded
        v_dequantized = v_float * v_scales_expanded
        
        return k_dequantized, v_dequantized


class KVCacheINT8:
    """
    Main KV-cache INT8 quantization system for Capibara-6.
    
    Manages quantized key-value cache for transformer attention with:
    - Paged cache blocks for memory efficiency
    - Per-head quantization scales
    - Dynamic scale adaptation
    - Hardware-optimized operations
    """
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.quantizer = KVQuantizer(config)
        
        # Cache storage
        self.cache_blocks = {}  # (layer_id, block_id) -> CacheBlock
        self.active_blocks = defaultdict(list)  # layer_id -> [block_ids]
        
        # Cache metadata
        self.layer_configs = {}  # layer_id -> (num_heads, head_dim)
        self.sequence_lengths = defaultdict(int)  # layer_id -> current_seq_len
        
        # Performance tracking
        self.stats = {
            'total_tokens_cached': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'quantization_time': 0.0,
            'dequantization_time': 0.0,
            'memory_usage_mb': 0.0,
            'compression_ratio': 0.0
        }
        
        logger.info(f"💾 KVCacheINT8 initialized for {self.config.target_hardware}")
    
    def register_layer(self, layer_id: int, num_heads: int, head_dim: int):
        """Register a layer configuration for cache management."""
        self.layer_configs[layer_id] = (num_heads, head_dim)
        self.active_blocks[layer_id] = []
        
        logger.debug(f"Registered layer {layer_id}: {num_heads} heads, {head_dim} head_dim")
    
    def _get_or_create_block(self, layer_id: int, block_id: int) -> CacheBlock:
        """Get existing cache block or create new one."""
        cache_key = (layer_id, block_id)
        
        if cache_key not in self.cache_blocks:
            if layer_id not in self.layer_configs:
                raise ValueError(f"Layer {layer_id} not registered")
            
            num_heads, head_dim = self.layer_configs[layer_id]
            self.cache_blocks[cache_key] = CacheBlock(
                block_id, self.config.cache_block_size, num_heads, head_dim
            )
            
            if block_id not in self.active_blocks[layer_id]:
                self.active_blocks[layer_id].append(block_id)
        
        return self.cache_blocks[cache_key]
    
    def add_kv_to_cache(self, layer_id: int, k_new: np.ndarray, v_new: np.ndarray,
                       start_position: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add new K,V pairs to cache and return quantized versions.
        
        Args:
            layer_id: Layer identifier
            k_new: New key tensor [seq_len, num_heads, head_dim]
            v_new: New value tensor [seq_len, num_heads, head_dim]
            start_position: Position to start insertion (for prefill)
            
        Returns:
            Tuple of (k_quantized_int8, v_quantized_int8)
        """
        start_time = time.time()
        
        seq_len = k_new.shape[0]
        if start_position is None:
            start_position = self.sequence_lengths[layer_id]
        
        # Calibrate scales
        k_scales, v_scales = self.quantizer.calibrate_scales(k_new, v_new, layer_id)
        
        # Quantize K,V
        k_quantized, v_quantized = self.quantizer.quantize_kv(k_new, v_new, k_scales, v_scales)
        
        # Determine which blocks to use
        end_position = start_position + seq_len
        start_block_id = start_position // self.config.cache_block_size
        end_block_id = (end_position - 1) // self.config.cache_block_size
        
        # Store in cache blocks
        current_pos = start_position
        for block_id in range(start_block_id, end_block_id + 1):
            block = self._get_or_create_block(layer_id, block_id)
            
            # Calculate positions within this block
            block_start = block_id * self.config.cache_block_size
            block_end = min((block_id + 1) * self.config.cache_block_size, end_position)
            
            in_block_start = current_pos - block_start
            in_block_end = block_end - block_start
            
            # Extract data for this block
            data_start = current_pos - start_position
            data_end = data_start + (in_block_end - in_block_start)
            
            k_block = k_quantized[data_start:data_end]
            v_block = v_quantized[data_start:data_end]
            
            # Add to block
            block.add_kv(k_block, v_block, k_scales, v_scales, in_block_start)
            
            current_pos = block_end
        
        # Update sequence length
        self.sequence_lengths[layer_id] = max(self.sequence_lengths[layer_id], end_position)
        
        # Update statistics
        self.stats['total_tokens_cached'] += seq_len
        self.stats['quantization_time'] += time.time() - start_time
        
        return k_quantized, v_quantized
    
    def get_kv_from_cache(self, layer_id: int, start_pos: int = 0, 
                         end_pos: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve and dequantize K,V from cache.
        
        Args:
            layer_id: Layer identifier
            start_pos: Start position in sequence
            end_pos: End position in sequence (None for all)
            
        Returns:
            Tuple of (k_dequantized, v_dequantized)
        """
        start_time = time.time()
        
        if end_pos is None:
            end_pos = self.sequence_lengths[layer_id]
        
        if end_pos <= start_pos:
            # Return empty tensors
            num_heads, head_dim = self.layer_configs[layer_id]
            k_empty = np.zeros((0, num_heads, head_dim), dtype=np.float32)
            v_empty = np.zeros((0, num_heads, head_dim), dtype=np.float32)
            return k_empty, v_empty
        
        # Determine which blocks to read
        start_block_id = start_pos // self.config.cache_block_size
        end_block_id = (end_pos - 1) // self.config.cache_block_size
        
        k_segments = []
        v_segments = []
        
        for block_id in range(start_block_id, end_block_id + 1):
            cache_key = (layer_id, block_id)
            if cache_key not in self.cache_blocks:
                self.stats['cache_misses'] += 1
                continue
            
            block = self.cache_blocks[cache_key]
            self.stats['cache_hits'] += 1
            
            # Calculate positions within this block
            block_start = block_id * self.config.cache_block_size
            block_end = min((block_id + 1) * self.config.cache_block_size, end_pos)
            
            in_block_start = max(0, start_pos - block_start)
            in_block_end = min(self.config.cache_block_size, block_end - block_start)
            
            # Get data from block
            k_block, v_block = block.get_kv(in_block_start, in_block_end)
            
            k_segments.append(k_block)
            v_segments.append(v_block)
        
        # Concatenate segments
        if k_segments:
            k_result = np.concatenate(k_segments, axis=0)
            v_result = np.concatenate(v_segments, axis=0)
        else:
            # No data found
            num_heads, head_dim = self.layer_configs[layer_id]
            k_result = np.zeros((0, num_heads, head_dim), dtype=np.float32)
            v_result = np.zeros((0, num_heads, head_dim), dtype=np.float32)
        
        # Update statistics
        self.stats['dequantization_time'] += time.time() - start_time
        
        return k_result, v_result
    
    def clear_cache(self, layer_id: Optional[int] = None):
        """Clear cache for specific layer or all layers."""
        if layer_id is not None:
            # Clear specific layer
            keys_to_remove = [key for key in self.cache_blocks.keys() if key[0] == layer_id]
            for key in keys_to_remove:
                del self.cache_blocks[key]
            self.active_blocks[layer_id] = []
            self.sequence_lengths[layer_id] = 0
        else:
            # Clear all cache
            self.cache_blocks.clear()
            self.active_blocks.clear()
            self.sequence_lengths.clear()
        
        logger.debug(f"Cache cleared for layer {layer_id if layer_id is not None else 'all'}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage statistics."""
        total_memory = 0.0
        layer_memory = defaultdict(float)
        
        for (layer_id, block_id), block in self.cache_blocks.items():
            block_memory = block.memory_usage_mb()
            total_memory += block_memory
            layer_memory[layer_id] += block_memory
        
        return {
            'total_mb': total_memory,
            'layers': dict(layer_memory),
            'average_block_mb': total_memory / max(len(self.cache_blocks), 1)
        }
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio vs FP16 storage."""
        if not self.cache_blocks:
            return 1.0
        
        # Estimate FP16 size
        fp16_size = 0
        int8_size = 0
        
        for (layer_id, _), block in self.cache_blocks.items():
            # FP16: 2 bytes per element
            seq_len = block.sequence_length
            num_heads, head_dim = self.layer_configs[layer_id]
            
            fp16_block_size = seq_len * num_heads * head_dim * 2 * 2  # K + V, 2 bytes each
            int8_block_size = block.memory_usage_mb() * 1024 * 1024  # Convert to bytes
            
            fp16_size += fp16_block_size
            int8_size += int8_block_size
        
        return fp16_size / max(int8_size, 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.get_memory_usage()
        compression_ratio = self.get_compression_ratio()
        
        return {
            **self.stats,
            'memory_usage_mb': memory_stats['total_mb'],
            'compression_ratio': compression_ratio,
            'cache_efficiency': self.stats['cache_hits'] / max(
                self.stats['cache_hits'] + self.stats['cache_misses'], 1
            ),
            'active_layers': len(self.layer_configs),
            'total_blocks': len(self.cache_blocks),
            'quantizer_stats': dict(self.quantizer.stats)
        }
    
    def validate_cache_consistency(self, layer_id: int) -> Dict[str, Any]:
        """Validate cache consistency for debugging."""
        validation_results = {
            'layer_id': layer_id,
            'registered': layer_id in self.layer_configs,
            'sequence_length': self.sequence_lengths[layer_id],
            'active_blocks': len(self.active_blocks[layer_id]),
            'issues': []
        }
        
        if layer_id not in self.layer_configs:
            validation_results['issues'].append('Layer not registered')
            return validation_results
        
        # Check block consistency
        for block_id in self.active_blocks[layer_id]:
            cache_key = (layer_id, block_id)
            if cache_key not in self.cache_blocks:
                validation_results['issues'].append(f'Block {block_id} missing from cache')
            else:
                block = self.cache_blocks[cache_key]
                if block.sequence_length > self.config.cache_block_size:
                    validation_results['issues'].append(f'Block {block_id} overflow')
        
        return validation_results


# Utility functions
def create_default_kv_cache_config(target_hardware: str = "tpu_v6e") -> KVCacheConfig:
    """Create default KV cache configuration for specific hardware."""
    
    if target_hardware == "tpu_v6e":
        return KVCacheConfig(
            quantization_method="per_head",
            max_sequence_length=8192,
            cache_block_size=256,
            memory_efficient=True,
            target_hardware="tpu_v6e",
            vectorized_ops=True,
            parallel_heads=True
        )
    
    elif target_hardware == "arm_axion":
        return KVCacheConfig(
            quantization_method="per_head",
            max_sequence_length=4096,
            cache_block_size=128,
            memory_efficient=True,
            target_hardware="arm_axion",
            cache_compression=True
        )
    
    else:  # Generic CPU
        return KVCacheConfig(
            quantization_method="global",
            max_sequence_length=2048,
            cache_block_size=64,
            target_hardware="cpu"
        )


def quick_setup_kv_cache(layer_configs: List[Tuple[int, int, int]], 
                        target_hardware: str = "tpu_v6e") -> KVCacheINT8:
    """
    Quick setup of KV cache for simple use cases.
    
    Args:
        layer_configs: List of (layer_id, num_heads, head_dim) tuples
        target_hardware: Target hardware platform
        
    Returns:
        Configured KV cache instance
    """
    config = create_default_kv_cache_config(target_hardware)
    kv_cache = KVCacheINT8(config)
    
    for layer_id, num_heads, head_dim in layer_configs:
        kv_cache.register_layer(layer_id, num_heads, head_dim)
    
    return kv_cache