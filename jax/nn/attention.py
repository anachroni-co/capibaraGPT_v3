"""
nn attention module.

# This module provides functionality for attention.
"""

import logging

try:
    import jax.numpy as jnp
    import jax.nn as jnn
    from jax import random, lax
    import math
    from . import initializers
except Exception:
    pass


    
    class MultiHeadAttention:
        """Multi-Head Attention (Vaswani et al., 2017)."""
        
        def __init__(self, d_model, num_heads, dropout_rate=0.1, use_bias=True):
            assert d_model % num_heads == 0
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.dropout_rate = dropout_rate
            self.use_bias = use_bias
            self.scale = 1.0 / math.sqrt(self.head_dim)
            
        def init_params(self, key):
            """Initialize attention parameters."""
            k1, k2, k3, k4 = random.split(key, 4)
            
            # Query, Key, Value projections
            wq = initializers.xavier_normal(k1, (self.d_model, self.d_model))
            wk = initializers.xavier_normal(k2, (self.d_model, self.d_model))
            wv = initializers.xavier_normal(k3, (self.d_model, self.d_model))
            
            # Output projection
            wo = initializers.xavier_normal(k4, (self.d_model, self.d_model))
            
            params = {'wq': wq, 'wk': wk, 'wv': wv, 'wo': wo}
            
            if self.use_bias:
                params.update({
                    'bq': jnp.zeros((self.d_model,)),
                    'bk': jnp.zeros((self.d_model,)),
                    'bv': jnp.zeros((self.d_model,)),
                    'bo': jnp.zeros((self.d_model,))
                })
                
            return params
        
        def __call__(self, query, key, value, params, mask=None, training=True, key_rng=None):
            """Forward pass through multi-head attention."""
            batch_size, seq_len, _ = query.shape
            
            # Linear projections
            q = jnp.dot(query, params['wq'])
            k = jnp.dot(key, params['wk'])
            v = jnp.dot(value, params['wv'])
            
            if self.use_bias:
                q = q + params['bq']
                k = k + params['bk']
                v = v + params['bv']
            
            # Reshape for multi-head attention
            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            
            # Scaled dot-product attention
            attention_output = self._scaled_dot_product_attention(
                q, k, v, mask, training, key_rng
            )
            
            # Reshape and concatenate heads
            attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
                batch_size, seq_len, self.d_model
            )
            
            # Output projection
            output = jnp.dot(attention_output, params['wo'])
            if self.use_bias:
                output = output + params['bo']
                
            return output
        
        def _scaled_dot_product_attention(self, q, k, v, mask=None, training=True, key_rng=None):
            """Scaled dot-product attention."""
            # Compute attention scores
            scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
            
            # Apply mask if provided
            if mask is not None:
                scores = jnp.where(mask, scores, -jnp.inf)
            
            # Softmax
            attention_weights = jnn.softmax(scores, axis=-1)
            
            # Apply dropout
            if training and self.dropout_rate > 0 and key_rng is not None:
                from .layers import dropout
                attention_weights = dropout(attention_weights, self.dropout_rate, key_rng)
            
            # Apply attention to values
            output = jnp.matmul(attention_weights, v)
            
            return output
    
    class SelfAttention:
        """Self-attention layer (special case of multi-head attention)."""
        
        def __init__(self, d_model, num_heads, dropout_rate=0.1, use_bias=True):
            self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate, use_bias)
            
        def init_params(self, key):
            return self.mha.init_params(key)
        
        def __call__(self, x, params, mask=None, training=True, key_rng=None):
            """Self-attention: query, key, value are all the same."""
            return self.mha(x, x, x, params, mask, training, key_rng)
    
    class CrossAttention:
        """Cross-attention layer for encoder-decoder architectures."""
        
        def __init__(self, d_model, num_heads, dropout_rate=0.1, use_bias=True):
            self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate, use_bias)
            
        def init_params(self, key):
            return self.mha.init_params(key)
        
        def __call__(self, query, key_value, params, mask=None, training=True, key_rng=None):
            """Cross-attention: query from decoder, key/value from encoder."""
            return self.mha(query, key_value, key_value, params, mask, training, key_rng)
    
    class FlashAttention:
        """Flash Attention (approximate implementation for JAX)."""
        
        def __init__(self, d_model, num_heads, block_size=128, dropout_rate=0.1):
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.block_size = block_size
            self.dropout_rate = dropout_rate
            self.scale = 1.0 / math.sqrt(self.head_dim)
            
        def init_params(self, key):
            """Initialize Flash Attention parameters."""
            k1, k2, k3, k4 = random.split(key, 4)
            
            wq = initializers.xavier_normal(k1, (self.d_model, self.d_model))
            wk = initializers.xavier_normal(k2, (self.d_model, self.d_model))
            wv = initializers.xavier_normal(k3, (self.d_model, self.d_model))
            wo = initializers.xavier_normal(k4, (self.d_model, self.d_model))
            
            return {'wq': wq, 'wk': wk, 'wv': wv, 'wo': wo}
        
        def __call__(self, query, key, value, params, mask=None, training=True, key_rng=None):
            """Memory-efficient attention computation."""
            batch_size, seq_len, _ = query.shape
            
            # Linear projections
            q = jnp.dot(query, params['wq']).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            k = jnp.dot(key, params['wk']).reshape(batch_size, -1, self.num_heads, self.head_dim)
            v = jnp.dot(value, params['wv']).reshape(batch_size, -1, self.num_heads, self.head_dim)
            
            # Block-wise attention (simplified version)
            output = self._block_attention(q, k, v, mask)
            
            # Reshape and project
            output = output.reshape(batch_size, seq_len, self.d_model)
            output = jnp.dot(output, params['wo'])
            
            return output
        
        def _block_attention(self, q, k, v, mask=None):
            """Block-wise attention computation."""
            batch_size, seq_len, num_heads, head_dim = q.shape
            kv_seq_len = k.shape[1]
            
            output = jnp.zeros_like(q)
            
            # Process in blocks (simplified - real Flash Attention is more complex)
            for i in range(0, seq_len, self.block_size):
                end_i = min(i + self.block_size, seq_len)
                q_block = q[:, i:end_i]
                
                # Compute attention for this block
                scores = jnp.einsum('bqhd,bkhd->bhqk', q_block, k) * self.scale
                
                if mask is not None:
                    mask_block = mask[:, :, i:end_i, :]
                    scores = jnp.where(mask_block, scores, -jnp.inf)
                
                attention_weights = jnn.softmax(scores, axis=-1)
                block_output = jnp.einsum('bhqk,bkhd->bqhd', attention_weights, v)
                
                # Ensure output is a JAX array before using .at
                if not hasattr(output, 'at'):
                    output = jnp.array(output)
                output = output.at[:, i:end_i].set(block_output)
            
            return output
    
    class GroupedQueryAttention:
        """Grouped Query Attention (GQA) - used in LLaMA-2."""
        
        def __init__(self, d_model, num_heads, num_kv_heads, dropout_rate=0.1):
            assert d_model % num_heads == 0
            assert num_heads % num_kv_heads == 0
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = d_model // num_heads
            self.kv_head_dim = d_model // num_kv_heads
            self.groups = num_heads // num_kv_heads
            self.dropout_rate = dropout_rate
            self.scale = 1.0 / math.sqrt(self.head_dim)
            
        def init_params(self, key):
            """Initialize GQA parameters."""
            k1, k2, k3, k4 = random.split(key, 4)
            
            wq = initializers.xavier_normal(k1, (self.d_model, self.d_model))
            wk = initializers.xavier_normal(k2, (self.d_model, self.num_kv_heads * self.kv_head_dim))
            wv = initializers.xavier_normal(k3, (self.d_model, self.num_kv_heads * self.kv_head_dim))
            wo = initializers.xavier_normal(k4, (self.d_model, self.d_model))
            
            return {'wq': wq, 'wk': wk, 'wv': wv, 'wo': wo}
        
        def __call__(self, query, key, value, params, mask=None, training=True, key_rng=None):
            """Forward pass through grouped query attention."""
            batch_size, seq_len, _ = query.shape
            kv_seq_len = key.shape[1]
            
            # Linear projections
            q = jnp.dot(query, params['wq']).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            k = jnp.dot(key, params['wk']).reshape(batch_size, kv_seq_len, self.num_kv_heads, self.kv_head_dim)
            v = jnp.dot(value, params['wv']).reshape(batch_size, kv_seq_len, self.num_kv_heads, self.kv_head_dim)
            
            # Repeat key and value for each group
            k = jnp.repeat(k, self.groups, axis=2)
            v = jnp.repeat(v, self.groups, axis=2)
            
            # Transpose for attention computation
            q = q.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
            k = k.transpose(0, 2, 1, 3)  # (batch, heads, kv_seq_len, head_dim)
            v = v.transpose(0, 2, 1, 3)  # (batch, heads, kv_seq_len, head_dim)
            
            # Scaled dot-product attention
            scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
            
            if mask is not None:
                scores = jnp.where(mask, scores, -jnp.inf)
            
            attention_weights = jnn.softmax(scores, axis=-1)
            
            if training and self.dropout_rate > 0 and key_rng is not None:
                from .layers import dropout
                attention_weights = dropout(attention_weights, self.dropout_rate, key_rng)
            
            output = jnp.matmul(attention_weights, v)
            
            # Reshape and concatenate
            output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
            
            # Output projection
            output = jnp.dot(output, params['wo'])
            
            return output
    
    class RotaryAttention:
        """Attention with Rotary Position Embedding (RoPE)."""
        
        def __init__(self, d_model, num_heads, max_seq_len=2048, dropout_rate=0.1):
            self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
            self.rope = None  # Will be initialized with layers.RotaryPositionalEmbedding
            
        def init_params(self, key):
            return self.mha.init_params(key)
        
        def __call__(self, query, key, value, params, seq_len, mask=None, training=True, key_rng=None):
            """Attention with rotary position embeddings."""
            if self.rope is None:
                from .layers import RotaryPositionalEmbedding
                self.rope = RotaryPositionalEmbedding(self.mha.head_dim)
            
            # Get rotary embeddings
            cos, sin = self.rope(seq_len)
            
            # Apply RoPE to query and key before attention
            batch_size, seq_len_actual, _ = query.shape
            
            # Reshape for applying RoPE
            q = jnp.dot(query, params['wq']).reshape(batch_size, seq_len_actual, self.mha.num_heads, self.mha.head_dim)
            k = jnp.dot(key, params['wk']).reshape(batch_size, -1, self.mha.num_heads, self.mha.head_dim)
            v = jnp.dot(value, params['wv'])
            
            # Apply rotary embeddings
            q = self.rope.apply_rotary_emb(q, cos[:seq_len_actual], sin[:seq_len_actual])
            k = self.rope.apply_rotary_emb(k, cos[:k.shape[1]], sin[:k.shape[1]])
            
            # Reshape back and continue with normal attention
            q = q.reshape(batch_size, seq_len_actual, self.mha.d_model)
            k = k.reshape(batch_size, k.shape[1], self.mha.d_model)
            
            # Use modified params
            rope_params = params.copy()
            rope_params['wq'] = jnp.eye(self.mha.d_model)  # Identity since we already projected
            rope_params['wk'] = jnp.eye(self.mha.d_model)
            
            return self.mha(q, k, v, rope_params, mask, training, key_rng)
    
    # Utility functions for attention patterns
    
    def causal_mask(seq_len):
        """Create causal (autoregressive) attention mask."""
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        return mask[None, None, :, :]  # Add batch and head dimensions
    
    def create_padding_mask(lengths, max_len):
        """Create padding mask from sequence lengths."""
        mask = jnp.arange(max_len)[None, :] < lengths[:, None]
        return mask[:, None, None, :]  # Add head and query dimensions
    
    def combine_masks(mask1, mask2):
        """Combine two attention masks."""
        if mask1 is None:
            return mask2
        if mask2 is None:
            return mask1
        return mask1 & mask2

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module attention.py starting")
    return True

if __name__ == "__main__":
    main()
