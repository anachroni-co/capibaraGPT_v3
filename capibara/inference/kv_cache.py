"""
KV-Cache Compression for CapibaraGPT.

Implements techniques to reduce memory usage of key-value cache during inference:
1. Sliding Window: Keep only recent K tokens
2. H2O (Heavy Hitter Oracle): Keep important tokens based on attention
3. Quantization: INT8/INT4 quantization of cache
4. StreamingLLM: Attention sinks + sliding window
5. Dynamic Compression: Progressively compress older tokens

Enables 128K+ context on consumer GPUs (RTX 4090, 24GB).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any, Dict, Callable
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """KV-cache compression strategies."""
    NONE = "none"                      # No compression
    SLIDING_WINDOW = "sliding"         # Fixed window, discard old
    H2O = "h2o"                        # Heavy Hitter Oracle
    STREAMING_LLM = "streaming"        # Attention sinks + window
    QUANTIZATION = "quantization"      # INT8/INT4 quantization
    PYRAMID = "pyramid"                # Progressive compression
    SCISSORHANDS = "scissorhands"      # Importance-based eviction
    DYNAMIC = "dynamic"                # Adaptive hybrid


class QuantizationType(Enum):
    """Quantization types for cache."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"


@dataclass
class KVCacheConfig:
    """Configuration for KV-cache compression."""

    # Strategy
    strategy: CompressionStrategy = CompressionStrategy.STREAMING_LLM

    # Capacity limits
    max_cache_tokens: int = 32768       # Maximum tokens in cache
    target_memory_mb: Optional[int] = None  # Target memory budget

    # Sliding window
    window_size: int = 4096             # Recent tokens to keep in full

    # StreamingLLM / Attention sinks
    num_sink_tokens: int = 4            # Initial tokens to always keep

    # H2O parameters
    heavy_hitter_ratio: float = 0.1     # Fraction of tokens as heavy hitters
    accumulate_attention: bool = True   # Accumulate attention scores

    # Quantization
    quantization: QuantizationType = QuantizationType.NONE
    quantize_after_tokens: int = 1024   # Quantize cache older than this

    # Pyramid compression
    pyramid_levels: int = 4             # Compression levels
    pyramid_ratio: float = 0.5          # Compression ratio per level

    # Dynamic/adaptive
    min_tokens_to_compress: int = 512
    compression_trigger_ratio: float = 0.8  # Compress when cache > 80% full

    # Per-layer settings
    per_layer_config: bool = False
    layers_full_cache: List[int] = field(default_factory=list)  # Layers without compression

    def estimate_memory_mb(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype_bytes: int = 2  # BF16
    ) -> float:
        """Estimate memory usage in MB."""
        # KV cache: 2 (K+V) * layers * heads * head_dim * seq_len * dtype
        tokens = self.max_cache_tokens
        if self.strategy == CompressionStrategy.SLIDING_WINDOW:
            tokens = min(tokens, self.window_size)
        elif self.strategy == CompressionStrategy.STREAMING_LLM:
            tokens = min(tokens, self.window_size + self.num_sink_tokens)

        bytes_total = 2 * num_layers * num_heads * head_dim * tokens * dtype_bytes

        if self.quantization == QuantizationType.INT8:
            bytes_total //= 2
        elif self.quantization == QuantizationType.INT4:
            bytes_total //= 4

        return bytes_total / (1024 * 1024)


@dataclass
class CacheStats:
    """Statistics for KV-cache."""

    current_tokens: int = 0
    max_tokens_seen: int = 0
    evictions: int = 0
    compressions: int = 0
    memory_mb: float = 0.0
    compression_ratio: float = 1.0

    # H2O stats
    heavy_hitters: int = 0
    attention_concentration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": self.current_tokens,
            "max_tokens": self.max_tokens_seen,
            "memory_mb": f"{self.memory_mb:.1f}",
            "compression": f"{self.compression_ratio:.2f}x",
            "evictions": self.evictions
        }


class KVCache:
    """
    Base KV-cache with compression support.

    Stores key-value pairs for transformer attention layers.
    Supports various compression strategies for long-context inference.
    """

    def __init__(
        self,
        config: KVCacheConfig,
        num_layers: int,
        num_heads: int,
        head_dim: int
    ):
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Initialize cache storage
        self.keys: List[Optional[Any]] = [None] * num_layers
        self.values: List[Optional[Any]] = [None] * num_layers

        # Attention scores for H2O
        self.attention_scores: List[Optional[Any]] = [None] * num_layers

        # Stats
        self.stats = CacheStats()

        # Token positions
        self.position = 0

        self._initialized = False

    def update(
        self,
        layer_idx: int,
        key: Any,
        value: Any,
        attention_weights: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Update cache with new key-value pair.

        Args:
            layer_idx: Layer index
            key: New key tensor [batch, heads, seq, head_dim]
            value: New value tensor [batch, heads, seq, head_dim]
            attention_weights: Optional attention weights for H2O

        Returns:
            Tuple of (full_keys, full_values) including cached
        """
        if self.keys[layer_idx] is None:
            # First update
            self.keys[layer_idx] = key
            self.values[layer_idx] = value
        else:
            # Concatenate with existing
            self.keys[layer_idx] = self._concat(self.keys[layer_idx], key)
            self.values[layer_idx] = self._concat(self.values[layer_idx], value)

        # Update attention scores for H2O
        if attention_weights is not None and self.config.strategy == CompressionStrategy.H2O:
            self._update_attention_scores(layer_idx, attention_weights)

        # Check if compression needed
        current_len = self._get_cache_len(layer_idx)
        if current_len > self.config.max_cache_tokens * self.config.compression_trigger_ratio:
            self._compress_cache(layer_idx)

        # Update stats
        self.stats.current_tokens = current_len
        self.stats.max_tokens_seen = max(self.stats.max_tokens_seen, current_len)

        return self.keys[layer_idx], self.values[layer_idx]

    def _concat(self, cached: Any, new: Any) -> Any:
        """Concatenate cached and new tensors."""
        try:
            import numpy as np
            if isinstance(cached, np.ndarray):
                return np.concatenate([cached, new], axis=2)
        except ImportError:
            pass

        try:
            import torch
            if isinstance(cached, torch.Tensor):
                return torch.cat([cached, new], dim=2)
        except ImportError:
            pass

        return new

    def _get_cache_len(self, layer_idx: int) -> int:
        """Get current cache length."""
        if self.keys[layer_idx] is None:
            return 0

        if hasattr(self.keys[layer_idx], 'shape'):
            return self.keys[layer_idx].shape[2]
        return 0

    def _compress_cache(self, layer_idx: int) -> None:
        """Apply compression strategy to cache."""
        strategy = self.config.strategy

        if strategy == CompressionStrategy.NONE:
            return
        elif strategy == CompressionStrategy.SLIDING_WINDOW:
            self._apply_sliding_window(layer_idx)
        elif strategy == CompressionStrategy.STREAMING_LLM:
            self._apply_streaming_llm(layer_idx)
        elif strategy == CompressionStrategy.H2O:
            self._apply_h2o(layer_idx)
        elif strategy == CompressionStrategy.PYRAMID:
            self._apply_pyramid(layer_idx)
        elif strategy == CompressionStrategy.DYNAMIC:
            self._apply_dynamic(layer_idx)

        self.stats.compressions += 1

    def _apply_sliding_window(self, layer_idx: int) -> None:
        """Keep only recent window_size tokens."""
        window = self.config.window_size

        if self.keys[layer_idx] is not None:
            self.keys[layer_idx] = self._slice_cache(self.keys[layer_idx], -window)
            self.values[layer_idx] = self._slice_cache(self.values[layer_idx], -window)
            self.stats.evictions += 1

    def _apply_streaming_llm(self, layer_idx: int) -> None:
        """
        StreamingLLM: Keep sink tokens + sliding window.

        Key insight: First few tokens act as "attention sinks" and should
        always be kept for stable attention patterns.
        """
        sinks = self.config.num_sink_tokens
        window = self.config.window_size

        if self.keys[layer_idx] is None:
            return

        cache_len = self._get_cache_len(layer_idx)
        if cache_len <= sinks + window:
            return

        # Keep: [0:sinks] + [-window:]
        sink_keys = self._slice_cache(self.keys[layer_idx], 0, sinks)
        window_keys = self._slice_cache(self.keys[layer_idx], -window)
        self.keys[layer_idx] = self._concat(sink_keys, window_keys)

        sink_values = self._slice_cache(self.values[layer_idx], 0, sinks)
        window_values = self._slice_cache(self.values[layer_idx], -window)
        self.values[layer_idx] = self._concat(sink_values, window_values)

        self.stats.evictions += 1

    def _apply_h2o(self, layer_idx: int) -> None:
        """
        Heavy Hitter Oracle: Keep tokens with highest accumulated attention.

        Tracks which tokens receive most attention and preserves those,
        evicting tokens that are rarely attended to.
        """
        if self.attention_scores[layer_idx] is None:
            # Fallback to sliding window
            self._apply_sliding_window(layer_idx)
            return

        try:
            import numpy as np

            scores = self.attention_scores[layer_idx]
            cache_len = self._get_cache_len(layer_idx)

            # Number of tokens to keep
            keep_count = int(cache_len * self.config.heavy_hitter_ratio)
            keep_count = max(keep_count, self.config.num_sink_tokens + self.config.window_size // 2)

            # Always keep sinks
            sinks = self.config.num_sink_tokens

            # Find heavy hitters (excluding sinks and recent)
            recent = min(self.config.window_size // 2, cache_len // 4)
            middle_scores = scores[sinks:-recent] if cache_len > sinks + recent else scores[sinks:]

            if len(middle_scores) > 0:
                threshold = np.percentile(middle_scores, 100 - self.config.heavy_hitter_ratio * 100)
                heavy_hitter_mask = middle_scores >= threshold

                # Build keep indices
                keep_indices = list(range(sinks))  # Sinks
                keep_indices.extend([i + sinks for i, keep in enumerate(heavy_hitter_mask) if keep])
                keep_indices.extend(range(cache_len - recent, cache_len))  # Recent
                keep_indices = sorted(set(keep_indices))

                # Apply mask
                self.keys[layer_idx] = self._index_cache(self.keys[layer_idx], keep_indices)
                self.values[layer_idx] = self._index_cache(self.values[layer_idx], keep_indices)
                self.attention_scores[layer_idx] = scores[keep_indices]

                self.stats.heavy_hitters = len(keep_indices)

            self.stats.evictions += 1

        except ImportError:
            self._apply_sliding_window(layer_idx)

    def _apply_pyramid(self, layer_idx: int) -> None:
        """
        Pyramid compression: Progressive compression of older tokens.

        Recent tokens: full resolution
        Older tokens: averaged/compressed representations
        """
        try:
            import numpy as np

            cache_len = self._get_cache_len(layer_idx)
            levels = self.config.pyramid_levels
            ratio = self.config.pyramid_ratio

            # Divide cache into levels
            level_size = cache_len // levels

            new_keys = []
            new_values = []

            for level in range(levels):
                start = level * level_size
                end = (level + 1) * level_size if level < levels - 1 else cache_len

                level_keys = self._slice_cache(self.keys[layer_idx], start, end)
                level_values = self._slice_cache(self.values[layer_idx], start, end)

                # Compress older levels more
                compression = int(1 / (ratio ** (levels - level - 1)))
                if compression > 1 and level < levels - 1:
                    level_keys = self._average_pool(level_keys, compression)
                    level_values = self._average_pool(level_values, compression)

                new_keys.append(level_keys)
                new_values.append(level_values)

            # Concatenate all levels
            self.keys[layer_idx] = self._concat_list(new_keys)
            self.values[layer_idx] = self._concat_list(new_values)

            self.stats.compressions += 1

        except ImportError:
            self._apply_sliding_window(layer_idx)

    def _apply_dynamic(self, layer_idx: int) -> None:
        """
        Dynamic compression: Adapt strategy based on content.

        Uses H2O for important tokens, pyramid for less important.
        """
        # Use H2O if attention scores available
        if self.attention_scores[layer_idx] is not None:
            self._apply_h2o(layer_idx)
        else:
            # Fallback to StreamingLLM
            self._apply_streaming_llm(layer_idx)

    def _update_attention_scores(
        self,
        layer_idx: int,
        attention_weights: Any
    ) -> None:
        """Update accumulated attention scores for H2O."""
        try:
            import numpy as np

            # Sum attention across heads and queries
            if hasattr(attention_weights, 'numpy'):
                weights = attention_weights.detach().cpu().numpy()
            else:
                weights = attention_weights

            # Average over batch and heads, sum over query positions
            scores = weights.mean(axis=(0, 1)).sum(axis=0)

            if self.attention_scores[layer_idx] is None:
                self.attention_scores[layer_idx] = scores
            else:
                # Exponential moving average
                alpha = 0.9
                old_len = len(self.attention_scores[layer_idx])
                new_len = len(scores)

                if new_len > old_len:
                    # Pad old scores
                    padded = np.zeros(new_len)
                    padded[:old_len] = self.attention_scores[layer_idx]
                    self.attention_scores[layer_idx] = alpha * padded + (1 - alpha) * scores
                else:
                    self.attention_scores[layer_idx][:new_len] = (
                        alpha * self.attention_scores[layer_idx][:new_len] +
                        (1 - alpha) * scores
                    )

        except ImportError:
            pass

    def _slice_cache(
        self,
        cache: Any,
        start: int,
        end: Optional[int] = None
    ) -> Any:
        """Slice cache along sequence dimension."""
        try:
            import numpy as np
            if isinstance(cache, np.ndarray):
                if end is None:
                    return cache[:, :, start:, :]
                return cache[:, :, start:end, :]
        except ImportError:
            pass

        try:
            import torch
            if isinstance(cache, torch.Tensor):
                if end is None:
                    return cache[:, :, start:, :]
                return cache[:, :, start:end, :]
        except ImportError:
            pass

        return cache

    def _index_cache(self, cache: Any, indices: List[int]) -> Any:
        """Index cache with specific positions."""
        try:
            import numpy as np
            if isinstance(cache, np.ndarray):
                return cache[:, :, indices, :]
        except ImportError:
            pass

        try:
            import torch
            if isinstance(cache, torch.Tensor):
                return cache[:, :, indices, :]
        except ImportError:
            pass

        return cache

    def _average_pool(self, cache: Any, factor: int) -> Any:
        """Average pool cache to compress."""
        try:
            import numpy as np
            if isinstance(cache, np.ndarray):
                # [batch, heads, seq, dim]
                seq_len = cache.shape[2]
                new_len = seq_len // factor
                reshaped = cache[:, :, :new_len * factor, :].reshape(
                    cache.shape[0], cache.shape[1], new_len, factor, cache.shape[3]
                )
                return reshaped.mean(axis=3)
        except ImportError:
            pass

        return cache

    def _concat_list(self, caches: List[Any]) -> Any:
        """Concatenate list of caches."""
        if not caches:
            return None

        result = caches[0]
        for c in caches[1:]:
            result = self._concat(result, c)
        return result

    def clear(self) -> None:
        """Clear all cached data."""
        self.keys = [None] * self.num_layers
        self.values = [None] * self.num_layers
        self.attention_scores = [None] * self.num_layers
        self.position = 0
        self.stats = CacheStats()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        total_bytes = 0

        for layer_idx in range(self.num_layers):
            if self.keys[layer_idx] is not None:
                if hasattr(self.keys[layer_idx], 'nbytes'):
                    total_bytes += self.keys[layer_idx].nbytes
                    total_bytes += self.values[layer_idx].nbytes
                elif hasattr(self.keys[layer_idx], 'element_size'):
                    total_bytes += (
                        self.keys[layer_idx].numel() *
                        self.keys[layer_idx].element_size()
                    )
                    total_bytes += (
                        self.values[layer_idx].numel() *
                        self.values[layer_idx].element_size()
                    )

        self.stats.memory_mb = total_bytes / (1024 * 1024)
        return self.stats.memory_mb


class QuantizedKVCache(KVCache):
    """
    KV-cache with quantization support.

    Quantizes older cache entries to INT8/INT4 to reduce memory.
    Recent entries stay in full precision for quality.
    """

    def __init__(
        self,
        config: KVCacheConfig,
        num_layers: int,
        num_heads: int,
        head_dim: int
    ):
        super().__init__(config, num_layers, num_heads, head_dim)

        # Quantization state
        self.quantized_keys: List[Optional[Any]] = [None] * num_layers
        self.quantized_values: List[Optional[Any]] = [None] * num_layers
        self.scales: List[Optional[Any]] = [None] * num_layers
        self.zero_points: List[Optional[Any]] = [None] * num_layers

    def _quantize_tensor(self, tensor: Any) -> Tuple[Any, Any, Any]:
        """Quantize tensor to INT8."""
        try:
            import numpy as np

            # Compute scale and zero point
            min_val = tensor.min()
            max_val = tensor.max()

            if self.config.quantization == QuantizationType.INT8:
                qmin, qmax = -128, 127
            elif self.config.quantization == QuantizationType.INT4:
                qmin, qmax = -8, 7
            else:
                return tensor, None, None

            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale

            quantized = np.clip(
                np.round(tensor / scale + zero_point),
                qmin, qmax
            ).astype(np.int8)

            return quantized, scale, zero_point

        except ImportError:
            return tensor, None, None

    def _dequantize_tensor(
        self,
        quantized: Any,
        scale: Any,
        zero_point: Any
    ) -> Any:
        """Dequantize tensor back to float."""
        try:
            import numpy as np
            return scale * (quantized.astype(np.float32) - zero_point)
        except ImportError:
            return quantized

    def update(
        self,
        layer_idx: int,
        key: Any,
        value: Any,
        attention_weights: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """Update cache with optional quantization of old entries."""
        # First, do normal update
        full_keys, full_values = super().update(
            layer_idx, key, value, attention_weights
        )

        # Check if we should quantize older entries
        cache_len = self._get_cache_len(layer_idx)

        if (self.config.quantization != QuantizationType.NONE and
            cache_len > self.config.quantize_after_tokens):

            # Quantize entries older than threshold
            threshold = self.config.quantize_after_tokens

            old_keys = self._slice_cache(full_keys, 0, -threshold)
            old_values = self._slice_cache(full_values, 0, -threshold)

            q_keys, scale_k, zp_k = self._quantize_tensor(old_keys)
            q_values, scale_v, zp_v = self._quantize_tensor(old_values)

            self.quantized_keys[layer_idx] = q_keys
            self.quantized_values[layer_idx] = q_values
            self.scales[layer_idx] = (scale_k, scale_v)
            self.zero_points[layer_idx] = (zp_k, zp_v)

            # Keep only recent in full precision
            self.keys[layer_idx] = self._slice_cache(full_keys, -threshold)
            self.values[layer_idx] = self._slice_cache(full_values, -threshold)

        return self.get_full_cache(layer_idx)

    def get_full_cache(self, layer_idx: int) -> Tuple[Any, Any]:
        """Get full cache, dequantizing as needed."""
        if self.quantized_keys[layer_idx] is None:
            return self.keys[layer_idx], self.values[layer_idx]

        # Dequantize and concatenate
        scale_k, scale_v = self.scales[layer_idx]
        zp_k, zp_v = self.zero_points[layer_idx]

        dq_keys = self._dequantize_tensor(
            self.quantized_keys[layer_idx], scale_k, zp_k
        )
        dq_values = self._dequantize_tensor(
            self.quantized_values[layer_idx], scale_v, zp_v
        )

        full_keys = self._concat(dq_keys, self.keys[layer_idx])
        full_values = self._concat(dq_values, self.values[layer_idx])

        return full_keys, full_values


def create_kv_cache_config(
    strategy: str = "streaming",
    max_tokens: int = 32768,
    window_size: int = 4096,
    num_sinks: int = 4,
    quantization: str = "none",
    heavy_hitter_ratio: float = 0.1
) -> KVCacheConfig:
    """
    Factory function for KV-cache configuration.

    Args:
        strategy: One of "none", "sliding", "streaming", "h2o", "pyramid", "dynamic"
        max_tokens: Maximum tokens to cache
        window_size: Sliding window size
        num_sinks: Number of attention sink tokens
        quantization: "none", "int8", or "int4"
        heavy_hitter_ratio: Fraction of tokens for H2O

    Returns:
        Configured KVCacheConfig
    """
    strategy_map = {
        "none": CompressionStrategy.NONE,
        "sliding": CompressionStrategy.SLIDING_WINDOW,
        "streaming": CompressionStrategy.STREAMING_LLM,
        "h2o": CompressionStrategy.H2O,
        "pyramid": CompressionStrategy.PYRAMID,
        "dynamic": CompressionStrategy.DYNAMIC
    }

    quant_map = {
        "none": QuantizationType.NONE,
        "int8": QuantizationType.INT8,
        "int4": QuantizationType.INT4,
        "fp8": QuantizationType.FP8
    }

    return KVCacheConfig(
        strategy=strategy_map.get(strategy, CompressionStrategy.STREAMING_LLM),
        max_cache_tokens=max_tokens,
        window_size=window_size,
        num_sink_tokens=num_sinks,
        quantization=quant_map.get(quantization, QuantizationType.NONE),
        heavy_hitter_ratio=heavy_hitter_ratio
    )


def create_kv_cache(
    config: KVCacheConfig,
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128
) -> KVCache:
    """
    Create KV-cache with appropriate type.

    Args:
        config: Cache configuration
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head

    Returns:
        Configured KVCache instance
    """
    if config.quantization != QuantizationType.NONE:
        return QuantizedKVCache(config, num_layers, num_heads, head_dim)
    else:
        return KVCache(config, num_layers, num_heads, head_dim)


# Convenience exports
__all__ = [
    "KVCacheConfig",
    "CompressionStrategy",
    "QuantizationType",
    "CacheStats",
    "KVCache",
    "QuantizedKVCache",
    "create_kv_cache_config",
    "create_kv_cache"
]
