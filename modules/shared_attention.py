"""
Shared Attention Module for CapibaraGPT

This module implements mecanismos of attention compartida optimizados para TPU,
incluyendo multi-head attention, cross-attention, y attention caching para
mejorar la eficiencia computacional.

Características:
- Multi-head attention optimizada para TPU
- Cross-attention para modelos encoder-decoder
- Attention caching para inferencia eficiente
- Sparse attention patterns
- Memory-efficient attention computation
- Gradient checkpointing support
"""

import os
import sys
import logging
import math
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from functools import partial

logger = logging.getLogger(__name__)

# JAX imports with fallbacks
try:
    import jax
    import jax.numpy as jnp
    from jax import partial as jax_partial
    HAS_JAX = True
    logger.info("✅ JAX available for shared attention")
except ImportError:
    HAS_JAX = False
    logger.warning("⚠️ JAX not available, using numpy fallbacks")
    import numpy as jnp
    
    # Create fallback partial
    def jax_partial(func, *args, **kwargs):
        def wrapper(*inner_args, **inner_kwargs):
            return func(*args, *inner_args, **kwargs, **inner_kwargs)
        return wrapper

@dataclass
class AttentionConfig:
    """Configuration for attention modules."""
    
    # Basic attention parameters
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    max_position_embeddings: int = 2048
    
    # Attention behavior
    attention_dropout: float = 0.1
    use_bias: bool = True
    scale_attention: bool = True
    
    # Memory optimization
    use_memory_efficient_attention: bool = True
    use_flash_attention: bool = False  # Not available without specific hardware
    gradient_checkpointing: bool = False
    
    # Sparse attention
    use_sparse_attention: bool = False
    sparsity_pattern: str = "local"  # local, global, random
    sparse_block_size: int = 64
    
    # Caching
    use_kv_cache: bool = True
    max_cache_size: int = 1024
    
    # TPU optimization
    use_bfloat16: bool = True
    shard_attention: bool = True

class AttentionCache:
    """Cache para keys y values en inferencia."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.cache = {}
        self.cache_size = 0
        self.max_size = config.max_cache_size
        
    def get(self, layer_id: int, head_id: int) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Gets keys and values from cache."""
        cache_key = (layer_id, head_id)
        return self.cache.get(cache_key)
    
    def set(self, layer_id: int, head_id: int, keys: jnp.ndarray, values: jnp.ndarray):
        """Saves keys y values en el cache."""
        cache_key = (layer_id, head_id)
        
        # Evict if cache is full
        if self.cache_size >= self.max_size and cache_key not in self.cache:
            self._evict_oldest()
        
        self.cache[cache_key] = (keys, values)
        if cache_key not in self.cache:
            self.cache_size += 1
    
    def clear(self):
        """Limpia el cache."""
        self.cache.clear()
        self.cache_size = 0
    
    def _evict_oldest(self):
        """Evict the oldest entry (simple FIFO)."""
        if self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.cache_size -= 1

class MultiHeadAttention:
    """
    Multi-head attention optimizada para TPU con soporte para caching
    y patrones of attention sparse.
    """
    
    def __init__(self, config: AttentionConfig, layer_id: int = 0):
        self.config = config
        self.layer_id = layer_id
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim) if config.scale_attention else 1.0
        
        # Initialize cache
        self.cache = AttentionCache(config) if config.use_kv_cache else None
        
        # Attention weights (would be actual parameters in real implementation)
        self.query_weight = self._init_weight((config.hidden_size, config.hidden_size))
        self.key_weight = self._init_weight((config.hidden_size, config.hidden_size))
        self.value_weight = self._init_weight((config.hidden_size, config.hidden_size))
        self.output_weight = self._init_weight((config.hidden_size, config.hidden_size))
        
        logger.info(f"🔍 MultiHeadAttention initialized (layer {layer_id})")
        logger.info(f"   Heads: {self.num_heads}, Head dim: {self.head_dim}")
        logger.info(f"   Memory efficient: {config.use_memory_efficient_attention}")
    
    def _init_weight(self, shape: Tuple[int, ...]) -> jnp.ndarray:
        """Initializes weights with Xavier initialization."""
        if HAS_JAX:
            key = jax.random.PRNGKey(42)
            return jax.random.normal(key, shape) * math.sqrt(2.0 / sum(shape))
        else:
            return np.random.normal(0, math.sqrt(2.0 / sum(shape)), shape).astype(np.float32)
    
    def __call__(self, 
                 query: jnp.ndarray,
                 key: Optional[jnp.ndarray] = None,
                 value: Optional[jnp.ndarray] = None,
                 mask: Optional[jnp.ndarray] = None,
                 training: bool = True,
                 use_cache: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Forward pass del multi-head attention.
        
        Args:
            query: Query tensor [batch, seq_len, hidden_size]
            key: Key tensor [batch, seq_len, hidden_size] (None for self-attention)
            value: Value tensor [batch, seq_len, hidden_size] (None for self-attention)
            mask: Attention mask [batch, seq_len, seq_len]
            training: Whether in training mode
            use_cache: Whether to use KV cache
            
        Returns:
            Dictionary with 'output', 'attention_weights', and optionally 'cache_updated'
        """
        batch_size, seq_len, hidden_size = query.shape
        
        # Self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = key
        
        # Linear projections
        q = self._linear_projection(query, self.query_weight)
        k = self._linear_projection(key, self.key_weight)
        v = self._linear_projection(value, self.value_weight)
        
        # Reshape for multi-head attention
        q = self._reshape_for_attention(q, batch_size, seq_len)
        k = self._reshape_for_attention(k, batch_size, seq_len)
        v = self._reshape_for_attention(v, batch_size, seq_len)
        
        # Use cache if available and requested
        if use_cache and self.cache is not None:
            k, v = self._apply_cache(k, v)
        
        # Compute attention
        if self.config.use_memory_efficient_attention:
            attention_output, attention_weights = self._memory_efficient_attention(
                q, k, v, mask, training
            )
        else:
            attention_output, attention_weights = self._standard_attention(
                q, k, v, mask, training
            )
        
        # Reshape back to [batch, seq_len, hidden_size]
        attention_output = self._reshape_from_attention(attention_output, batch_size, seq_len)
        
        # Output projection
        output = self._linear_projection(attention_output, self.output_weight)
        
        result = {
            'output': output,
            'attention_weights': attention_weights
        }
        
        if use_cache and self.cache is not None:
            result['cache_updated'] = True
        
        return result
    
    def _linear_projection(self, x: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
        """Proyección linear."""
        if HAS_JAX:
            return jnp.dot(x, weight)
        else:
            return np.dot(x, weight)
    
    def _reshape_for_attention(self, x: jnp.ndarray, batch_size: int, seq_len: int) -> jnp.ndarray:
        """Reshape tensor para multi-head attention."""
        # [batch, seq_len, hidden] -> [batch, seq_len, num_heads, head_dim]
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        # -> [batch, num_heads, seq_len, head_dim]
        if HAS_JAX:
            return jnp.transpose(x, (0, 2, 1, 3))
        else:
            return np.transpose(x, (0, 2, 1, 3))
    
    def _reshape_from_attention(self, x: jnp.ndarray, batch_size: int, seq_len: int) -> jnp.ndarray:
        """Reshape tensor de vuelta desde multi-head attention."""
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        if HAS_JAX:
            x = jnp.transpose(x, (0, 2, 1, 3))
        else:
            x = np.transpose(x, (0, 2, 1, 3))
        # -> [batch, seq_len, hidden]
        return x.reshape(batch_size, seq_len, -1)
    
    def _standard_attention(self, 
                          q: jnp.ndarray, 
                          k: jnp.ndarray, 
                          v: jnp.ndarray,
                          mask: Optional[jnp.ndarray] = None,
                          training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Implementation standard attention."""
        # Compute attention scores
        if HAS_JAX:
            scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * self.scale
        else:
            scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head
            mask = mask[:, None, :, :]  # [batch, 1, seq_len, seq_len]
            scores = scores + mask * -1e9
        
        # Softmax
        attention_weights = self._softmax(scores, axis=-1)
        
        # Apply dropout if training
        if training and self.config.attention_dropout > 0:
            attention_weights = self._dropout(attention_weights, self.config.attention_dropout)
        
        # Apply attention to values
        if HAS_JAX:
            attention_output = jnp.matmul(attention_weights, v)
        else:
            attention_output = np.matmul(attention_weights, v)
        
        return attention_output, attention_weights
    
    def _memory_efficient_attention(self,
                                  q: jnp.ndarray,
                                  k: jnp.ndarray, 
                                  v: jnp.ndarray,
                                  mask: Optional[jnp.ndarray] = None,
                                  training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Implementation memory-efficient attention using chunking."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        chunk_size = min(512, seq_len)  # Chunk size for memory efficiency
        
        if seq_len <= chunk_size:
            return self._standard_attention(q, k, v, mask, training)
        
        # Initialize output
        if HAS_JAX:
            attention_output = jnp.zeros_like(q)
            attention_weights = jnp.zeros((batch_size, num_heads, seq_len, seq_len))
        else:
            attention_output = np.zeros_like(q)
            attention_weights = np.zeros((batch_size, num_heads, seq_len, seq_len))
        
        # Process in chunks
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            
            # Extract query chunk
            q_chunk = q[:, :, i:end_i, :]
            
            # Compute attention for this chunk
            if HAS_JAX:
                scores = jnp.matmul(q_chunk, jnp.transpose(k, (0, 1, 3, 2))) * self.scale
            else:
                scores = np.matmul(q_chunk, np.transpose(k, (0, 1, 3, 2))) * self.scale
            
            # Apply mask if provided
            if mask is not None:
                mask_chunk = mask[:, None, i:end_i, :]
                scores = scores + mask_chunk * -1e9
            
            # Softmax
            weights_chunk = self._softmax(scores, axis=-1)
            
            # Apply dropout if training
            if training and self.config.attention_dropout > 0:
                weights_chunk = self._dropout(weights_chunk, self.config.attention_dropout)
            
            # Apply attention to values
            if HAS_JAX:
                output_chunk = jnp.matmul(weights_chunk, v)
            else:
                output_chunk = np.matmul(weights_chunk, v)
            
            # Store results
            attention_output = attention_output.at[:, :, i:end_i, :].set(output_chunk) if HAS_JAX else \
                              self._set_slice(attention_output, output_chunk, i, end_i)
            attention_weights = attention_weights.at[:, :, i:end_i, :].set(weights_chunk) if HAS_JAX else \
                               self._set_slice_2d(attention_weights, weights_chunk, i, end_i)
        
        return attention_output, attention_weights
    
    def _set_slice(self, array: np.ndarray, value: np.ndarray, start: int, end: int) -> np.ndarray:
        """Helper for numpy slice assignment."""
        array[:, :, start:end, :] = value
        return array
    
    def _set_slice_2d(self, array: np.ndarray, value: np.ndarray, start: int, end: int) -> np.ndarray:
        """Helper for numpy slice assignment in 2D."""
        array[:, :, start:end, :] = value
        return array
    
    def _softmax(self, x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        """Softmax implementation."""
        if HAS_JAX:
            return jax.nn.softmax(x, axis=axis)
        else:
            # Numerical stability
            x_max = np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _dropout(self, x: jnp.ndarray, rate: float) -> jnp.ndarray:
        """Dropout implementation."""
        if HAS_JAX:
            key = jax.random.PRNGKey(42)  # In real implementation, use proper key
            return jax.random.dropout(key, x, rate)
        else:
            # Simple dropout for numpy fallback
            mask = np.random.random(x.shape) > rate
            return x * mask / (1 - rate)
    
    def _apply_cache(self, k: jnp.ndarray, v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Applies cache de keys y values."""
        if self.cache is None:
            return k, v
        
        batch_size, num_heads, seq_len, head_dim = k.shape
        
        # For each head, try to use cache
        cached_k_list = []
        cached_v_list = []
        
        for head_id in range(num_heads):
            cached_kv = self.cache.get(self.layer_id, head_id)
            
            if cached_kv is not None:
                cached_k, cached_v = cached_kv
                # Concatenate with new keys/values
                if HAS_JAX:
                    new_k = jnp.concatenate([cached_k, k[:, head_id:head_id+1, :, :]], axis=2)
                    new_v = jnp.concatenate([cached_v, v[:, head_id:head_id+1, :, :]], axis=2)
                else:
                    new_k = np.concatenate([cached_k, k[:, head_id:head_id+1, :, :]], axis=2)
                    new_v = np.concatenate([cached_v, v[:, head_id:head_id+1, :, :]], axis=2)
            else:
                new_k = k[:, head_id:head_id+1, :, :]
                new_v = v[:, head_id:head_id+1, :, :]
            
            # Actualizar cache
            self.cache.set(self.layer_id, head_id, new_k, new_v)
            
            cached_k_list.append(new_k)
            cached_v_list.append(new_v)
        
        # Concatenate all heads
        if HAS_JAX:
            final_k = jnp.concatenate(cached_k_list, axis=1)
            final_v = jnp.concatenate(cached_v_list, axis=1)
        else:
            final_k = np.concatenate(cached_k_list, axis=1)
            final_v = np.concatenate(cached_v_list, axis=1)
        
        return final_k, final_v

class CrossAttention(MultiHeadAttention):
    """
    Cross-attention para modelos encoder-decoder.
    Hereda de MultiHeadAttention pero especializada para cross-attention.
    """
    
    def __init__(self, config: AttentionConfig, layer_id: int = 0):
        super().__init__(config, layer_id)
        logger.info(f"🔗 CrossAttention initialized (layer {layer_id})")
    
    def __call__(self,
                 query: jnp.ndarray,
                 encoder_output: jnp.ndarray,
                 encoder_mask: Optional[jnp.ndarray] = None,
                 training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Cross-attention entre decoder query y encoder output.
        
        Args:
            query: Decoder query [batch, dec_seq_len, hidden_size]
            encoder_output: Encoder output [batch, enc_seq_len, hidden_size]
            encoder_mask: Encoder padding mask [batch, enc_seq_len]
            training: Whether in training mode
        """
        # In cross-attention, key and value come from encoder
        return super().__call__(
            query=query,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask,
            training=training,
            use_cache=False  # Cross-attention typically doesn't use cache
        )

class SparseAttention(MultiHeadAttention):
    """
    Sparse attention patterns para reducir complejidad computacional.
    """
    
    def __init__(self, config: AttentionConfig, layer_id: int = 0):
        super().__init__(config, layer_id)
        self.sparsity_pattern = config.sparsity_pattern
        self.block_size = config.sparse_block_size
        logger.info(f"🕸️ SparseAttention initialized (layer {layer_id})")
        logger.info(f"   Pattern: {self.sparsity_pattern}, Block size: {self.block_size}")
    
    def _create_sparse_mask(self, seq_len: int, batch_size: int) -> jnp.ndarray:
        """Creates sparse mask according to configured pattern."""
        if self.sparsity_pattern == "local":
            return self._create_local_mask(seq_len, batch_size)
        elif self.sparsity_pattern == "global":
            return self._create_global_mask(seq_len, batch_size)
        elif self.sparsity_pattern == "random":
            return self._create_random_mask(seq_len, batch_size)
        else:
            # Full attention as fallback
            if HAS_JAX:
                return jnp.zeros((batch_size, seq_len, seq_len))
            else:
                return np.zeros((batch_size, seq_len, seq_len))
    
    def _create_local_mask(self, seq_len: int, batch_size: int) -> jnp.ndarray:
        """Creates mask for local attention (sliding window)."""
        window_size = self.block_size
        
        if HAS_JAX:
            mask = jnp.full((batch_size, seq_len, seq_len), -1e9)
        else:
            mask = np.full((batch_size, seq_len, seq_len), -1e9)
        
        # Allow attention within window
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            if HAS_JAX:
                mask = mask.at[:, i, start:end].set(0.0)
            else:
                mask[:, i, start:end] = 0.0
        
        return mask
    
    def _create_global_mask(self, seq_len: int, batch_size: int) -> jnp.ndarray:
        """Creates mask for global attention (some tokens attend to all)."""
        global_tokens = min(self.block_size, seq_len // 4)
        
        if HAS_JAX:
            mask = jnp.full((batch_size, seq_len, seq_len), -1e9)
            # Global tokens can attend to everything
            mask = mask.at[:, :global_tokens, :].set(0.0)
            # Everyone can attend to global tokens
            mask = mask.at[:, :, :global_tokens].set(0.0)
        else:
            mask = np.full((batch_size, seq_len, seq_len), -1e9)
            mask[:, :global_tokens, :] = 0.0
            mask[:, :, :global_tokens] = 0.0
        
        return mask
    
    def _create_random_mask(self, seq_len: int, batch_size: int) -> jnp.ndarray:
        """Creates mask for random sparse attention."""
        sparsity_ratio = 0.1  # 10% of connections
        num_connections = int(seq_len * seq_len * sparsity_ratio)
        
        if HAS_JAX:
            mask = jnp.full((batch_size, seq_len, seq_len), -1e9)
            key = jax.random.PRNGKey(42)
            
            for b in range(batch_size):
                # Random positions to unmask
                positions = jax.random.choice(
                    key, seq_len * seq_len, (num_connections,), replace=False
                )
                rows = positions // seq_len
                cols = positions % seq_len
                mask = mask.at[b, rows, cols].set(0.0)
        else:
            mask = np.full((batch_size, seq_len, seq_len), -1e9)
            
            for b in range(batch_size):
                positions = np.random.choice(seq_len * seq_len, num_connections, replace=False)
                rows = positions // seq_len
                cols = positions % seq_len
                mask[b, rows, cols] = 0.0
        
        return mask
    
    def __call__(self, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        """Override para aplicar sparse attention."""
        # Get sequence length from query
        query = args[0] if args else kwargs['query']
        batch_size, seq_len, _ = query.shape
        
        # Create sparse mask
        sparse_mask = self._create_sparse_mask(seq_len, batch_size)
        
        # Combine with existing mask if provided
        existing_mask = kwargs.get('mask')
        if existing_mask is not None:
            if HAS_JAX:
                combined_mask = jnp.minimum(sparse_mask, existing_mask)
            else:
                combined_mask = np.minimum(sparse_mask, existing_mask)
        else:
            combined_mask = sparse_mask
        
        # Update mask in kwargs
        kwargs['mask'] = combined_mask
        
        return super().__call__(*args, **kwargs)

class SharedAttentionManager:
    """
    Manager para coordinar múltiples modules of attention compartida.
    """
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.attention_modules = {}
        self.global_cache = AttentionCache(config)
        
        logger.info("🎯 SharedAttentionManager initialized")
    
    def create_attention_module(self, 
                              module_type: str = "multi_head",
                              layer_id: int = 0,
                              **kwargs) -> MultiHeadAttention:
        """Creates an attention module of the specified type."""
        
        if module_type == "multi_head":
            module = MultiHeadAttention(self.config, layer_id)
        elif module_type == "cross_attention":
            module = CrossAttention(self.config, layer_id)
        elif module_type == "sparse_attention":
            module = SparseAttention(self.config, layer_id)
        else:
            raise ValueError(f"Unknown attention module type: {module_type}")
        
        # Register module
        module_key = f"{module_type}_{layer_id}"
        self.attention_modules[module_key] = module
        
        logger.info(f"✅ Created attention module: {module_key}")
        return module
    
    def get_attention_module(self, module_type: str, layer_id: int) -> Optional[MultiHeadAttention]:
        """Gets an existing attention module."""
        module_key = f"{module_type}_{layer_id}"
        return self.attention_modules.get(module_key)
    
    def clear_all_caches(self):
        """Limpia todos los caches of attention."""
        self.global_cache.clear()
        for module in self.attention_modules.values():
            if hasattr(module, 'cache') and module.cache is not None:
                module.cache.clear()
        
        logger.info("🧹 All attention caches cleared")
    
    def get_attention_stats(self) -> Dict[str, Any]:
        """Gets attention modules statistics."""
        stats = {
            'total_modules': len(self.attention_modules),
            'module_types': {},
            'cache_stats': {
                'global_cache_size': self.global_cache.cache_size,
                'global_cache_max': self.global_cache.max_size
            },
            'config': self.config.__dict__
        }
        
        # Count module types
        for module_key in self.attention_modules:
            module_type = module_key.rsplit('_', 1)[0]
            stats['module_types'][module_type] = stats['module_types'].get(module_type, 0) + 1
        
        return stats

# Factory functions
def create_attention_config(**kwargs) -> AttentionConfig:
    """Creates attention configuration with default values."""
    return AttentionConfig(**kwargs)

def create_multi_head_attention(config: Optional[AttentionConfig] = None, 
                              layer_id: int = 0) -> MultiHeadAttention:
    """Factory to create multi-head attention."""
    config = config or AttentionConfig()
    return MultiHeadAttention(config, layer_id)

def create_cross_attention(config: Optional[AttentionConfig] = None,
                         layer_id: int = 0) -> CrossAttention:
    """Factory to create cross-attention."""
    config = config or AttentionConfig()
    return CrossAttention(config, layer_id)

def create_sparse_attention(config: Optional[AttentionConfig] = None,
                          layer_id: int = 0,
                          sparsity_pattern: str = "local") -> SparseAttention:
    """Factory to create sparse attention."""
    config = config or AttentionConfig()
    config.sparsity_pattern = sparsity_pattern
    config.use_sparse_attention = True
    return SparseAttention(config, layer_id)

def create_attention_manager(config: Optional[AttentionConfig] = None) -> SharedAttentionManager:
    """Factory to create attention manager."""
    config = config or AttentionConfig()
    return SharedAttentionManager(config)

# Global manager instance
_global_attention_manager: Optional[SharedAttentionManager] = None

def get_global_attention_manager() -> SharedAttentionManager:
    """Gets global attention manager instance."""
    global _global_attention_manager
    if _global_attention_manager is None:
        _global_attention_manager = create_attention_manager()
    return _global_attention_manager

def main():
    """Main function for testing."""
    logger.info("🔍 Shared Attention Module - Testing Mode")
    
    # Create configuration
    config = AttentionConfig(
        hidden_size=768,
        num_heads=12,
        max_position_embeddings=1024,
        use_memory_efficient_attention=True
    )
    
    # Create manager
    manager = create_attention_manager(config)
    
    # Create different attention types
    multi_head = manager.create_attention_module("multi_head", layer_id=0)
    cross_attention = manager.create_attention_module("cross_attention", layer_id=1)
    sparse_attention = manager.create_attention_module("sparse_attention", layer_id=2)
    
    # Test básico
    batch_size, seq_len, hidden_size = 2, 128, 768
    
    if HAS_JAX:
        key = jax.random.PRNGKey(42)
        query = jax.random.normal(key, (batch_size, seq_len, hidden_size))
    else:
        query = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    
    # Test multi-head attention
    result = multi_head(query, training=False)
    logger.info(f"Multi-head attention output shape: {result['output'].shape}")
    
    # Show statistics
    stats = manager.get_attention_stats()
    logger.info(f"Attention stats: {stats}")
    
    logger.info("✅ Shared attention testing completed")

if __name__ == "__main__":
    main()