"""
Flax Decorators Examples
=======================

Practical examples showing how to use Flax-specific decorators
for building modern transformer architectures.
"""

import logging
logger = logging.getLogger(__name__)

try:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from jax import random
    from .flax_decorators import (
        flax_module,
        auto_param,
        with_state,
        flax_cached,
        flax_training_step,
        flax_scan_layer,
        multihead_attention,
        causal_attention,
        layer_norm_wrapper,
        batch_norm_wrapper,
        flax_jit,
        flax_vmap,
        flax_remat,
        transformer_block,
        embedding_layer,
        stateful_module,
        flax_init_only,
        flax_apply_only,
        flax_conditional,
    )
    
    # Example 1: simple Module Creation
    
    @flax_module(compact=True, name="MyDense")
    def dense_layer(self, x, features=128):
        """Create a dense layer using decorator."""
        return nn.Dense(features=features)(x)
    
    # Example 2: Auto Parameter Management
    
    @flax_module(compact=True)
    def custom_attention(self, x):
        """Attention with automatic parameter creation."""
        
        @auto_param('scale', lambda key, shape: jnp.ones(shape) * 0.1, ())
        def scaled_attention(self, query, key, value):
            scale = getattr(self, '_scale', 1.0)
            scores = jnp.matmul(query, key.transpose(-1, -2)) * scale
            attention_weights = nn.softmax(scores, axis=-1)
            return jnp.matmul(attention_weights, value)
        
        # Split into Q, K, V
        qkv = nn.Dense(features=3 * x.shape[-1])(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        return scaled_attention(self, q, k, v)
    
    # Example 3: Stateful Module
    
    @stateful_module(state_vars={'running_mean': 0.0, 'step_count': 0})
    def adaptive_norm_layer(self, x):
        """Layer with evolving normalization."""
        
        @with_state('running_mean', 0.0)
        @with_state('step_count', 0)
        def update_stats(self, current_mean):
            step = self.variable('state', 'step_count')
            running_mean = self.variable('state', 'running_mean')
            
            # Update running statistics
            step.value += 1
            alpha = 1.0 / step.value
            running_mean.value = (1 - alpha) * running_mean.value + alpha * current_mean
            
            return running_mean.value
        
        current_mean = jnp.mean(x)
        adaptive_mean = update_stats(self, current_mean)
        return x - adaptive_mean
    
    # Example 4: Complete Transformer Block
    
    @transformer_block(d_model=512, num_heads=8, d_ff=2048)
    def custom_transformer_block(self, x, deterministic=True):
        """Custom FFN for transformer block."""
        # Custom feed-forward with GELU and layer scaling
        y = nn.Dense(features=2048)(x)
        y = nn.gelu(y)
        y = nn.Dense(features=512)(x)
        
        # Layer scaling (like in CaiT)
        scale = self.param('layer_scale', 
                          nn.initializers.constant(1e-4), 
                          (512,))
        return y * scale
    
    # Example 5: Causal GPT-style Attention
    
    @causal_attention(num_heads=12, qkv_features=768)
    def gpt_attention(self, x):
        """GPT-style causal attention with preprocessing."""
        # Pre-attention normalization
        return nn.LayerNorm()(x)
    
    # Example 6: Embedding with Position Encoding
    
    @embedding_layer(vocab_size=50000, embed_dim=768, max_len=2048)
    def gpt_embeddings(self, token_emb):
        """Custom embedding processing."""
        # Add dropout to embeddings
        return nn.Dropout(rate=0.1)(token_emb, deterministic=False)
    
    # Example 7: Training Step with BatchNorm
    
    @flax_training_step(has_state=True)
    def train_step_with_bn(variables, batch):
        """Training step that handles BatchNorm state."""
        inputs, targets = batch
        
        def model_fn(x, training=True):
            x = nn.Dense(128)(x)
            x = nn.BatchNorm(use_running_average=not training)(x, use_running_average=not training)
            x = nn.relu(x)
            x = nn.Dense(10)(x)
            return x
        
        # Apply model
        logits, new_state = nn.apply(model_fn, variables)(inputs, training=True, mutable=['batch_stats'])
        
        # Compute loss
        loss = jnp.mean((logits - targets) ** 2)
        
        return loss, new_state
    
    # Example 8: Scanned Transformer Layers
    
    @flax_scan_layer(length=12)  # 12 layers
    class ScannedTransformerLayer(nn.Module):
        d_model: int = 768
        num_heads: int = 12
        
        @nn.compact
        def __call__(self, x, mask=None):
            # Self-attention
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.d_model
            )
            
            # Pre-norm
            normed_x = nn.LayerNorm()(x)
            attn_out = attn(normed_x, normed_x, mask=mask)
            x = x + attn_out
            
            # FFN
            normed_x = nn.LayerNorm()(x)
            ff_out = nn.Dense(4 * self.d_model)(normed_x)
            ff_out = nn.gelu(ff_out)
            ff_out = nn.Dense(self.d_model)(ff_out)
            x = x + ff_out
            
            return x, None  # not carry for scan
    
    # Example 9: JIT-compiled Flax Functions
    
    @flax_jit(static_argnames=['training'])
    def fast_model_apply(params, inputs, training=False):
        """JIT-compiled model application."""
        
        def model_fn(x):
            x = nn.Dense(512)(x)
            x = nn.Dropout(0.1, deterministic=not training)(x)
            x = nn.relu(x)
            x = nn.Dense(256)(x)
            return x
        
        return model_fn.apply({'params': params}, inputs)
    
    # Example 10: Conditional Execution
    
    @flax_module(compact=True)
    def conditional_layer(self, x, use_attention=True):
        """Layer with conditional attention."""
        
        @flax_conditional('use_attention', 
                         true_fn=lambda x: x + 0.1,  # Attention residual
                         false_fn=lambda x: x)       # Skip connection
        def maybe_attention(self, x):
            if self.use_attention:
                attn = nn.MultiHeadDotProductAttention(num_heads=8)
                return attn(x, x)
            return x
        
        self.use_attention = use_attention
        return maybe_attention(self, x)
    
    # Example 11: Complete GPT Model
    
    class FlaxGPT(nn.Module):
        """Complete GPT model using Flax decorators."""
        vocab_size: int = 50000
        d_model: int = 768
        num_heads: int = 12
        num_layers: int = 12
        max_len: int = 2048
        
        def setup(self):
            # Embeddings with decorator
            self.embeddings = embedding_layer(
                vocab_size=self.vocab_size,
                embed_dim=self.d_model,
                max_len=self.max_len
            )(lambda self, x: nn.Dropout(0.1)(x))
            
            # Transformer blocks with decorators
            self.layers = [
                transformer_block(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=4 * self.d_model
                )(None)  # Use default FFN
                for _ in range(self.num_layers)
            ]
            
            # end layer norm
            self.final_norm = nn.LayerNorm()
            
            # Output head
            self.lm_head = nn.Dense(self.vocab_size, use_bias=False)
        
        @flax_jit(static_argnames=['training'])
        def __call__(self, tokens, training=False):
            # Embeddings
            x = self.embeddings(tokens)
            
            # Causal mask
            seq_len = tokens.shape[1]
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            mask = causal_mask[None, None, :, :]
            
            # Transformer layers
            for layer in self.layers:
                x = layer(x, mask=mask, deterministic=not training)
            
            # end norm and projection
            x = self.final_norm(x)
            logits = self.lm_head(x)
            
            return logits
    
    # Example 12: Usage demo
    
    def demo_flax_decorators():
        """Demonstrate Flax decorators in action."""
        
        # Initialize model
        model = FlaxGPT()
        
        # Dummy input
        key = random.PRNGKey(42)
        tokens = jnp.ones((2, 128), dtype=jnp.int32)  # batch_size=2, seq_len=128
        
        # Initialize parameters
        variables = model.init(key, tokens, training=False)
        
        # Forward pass
        logits = model.apply(variables, tokens, training=False)
        
        logger.info(f"Input shape: {tokens.shape}")
        logger.info(f"Output shape: {logits.shape}")
        logger.info(f"Model parameters: {sum(x.size for x in jax.tree_leaves(variables))}")
        
        return model, variables, logits

except ImportError:
    logger.warning("️  Flax not available - examples are stubs")
    
    def demo_flax_decorators():
        logger.info("Flax decorators demo requires Flax installation")
        return None, None, None
    
    # Stub classes
    class FlaxGPT:
        def __init__(self, *args, **kwargs):
            pass

__all__ = [
    'dense_layer', 'custom_attention', 'adaptive_norm_layer',
    'custom_transformer_block', 'gpt_attention', 'gpt_embeddings',
    'train_step_with_bn', 'ScannedTransformerLayer', 'fast_model_apply',
    'conditional_layer', 'FlaxGPT', 'demo_flax_decorators'
]