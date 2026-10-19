"""
layers meta_la module.

# This module provides functionality for meta_la.
"""

import os
import sys

import logging
# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../CapibaraGPT v3
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

import jax
from flax import linen as nn
from dataclasses import dataclass
from jax import numpy as jnp
from typing import Tuple, Optional, Dict, Any

from capibara.interfaces.ilayer import ILayer
from capibara.layers.base import BaseLayer, LayerConfig

logger = logging.getLogger(__name__)

@dataclass
class MetaLAConfig(LayerConfig):
    """Configuration for Meta-Learning Attention (Meta-LA) layer"""
    hidden_dim: int = 768  # Use main's larger default
    hidden_size: int = 768  # Compatibility alias
    num_heads: int = 12  # Use main's larger default
    num_attention_heads: int = 12  # Compatibility alias
    dropout_rate: float = 0.1
    meta_learning_rate: float = 0.001
    adaptation_steps: int = 5  # From main branch

class MetaLA(BaseLayer):
    """
    Meta-Learning Attention (Meta-LA) Layer
    
    Advanced attention mechanism that adapts based on meta-learning
    principles for improved context understanding.
    """
    config: MetaLAConfig
    
    def setup(self):
        # Use the larger of the two head counts for better performance
        num_heads = max(getattr(self.config, 'num_heads', 8), 
                       getattr(self.config, 'num_attention_heads', 12))
        hidden_dim = max(getattr(self.config, 'hidden_dim', 512),
                        getattr(self.config, 'hidden_size', 768))
        
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=num_heads,
            qkv_features=hidden_dim,
            dropout_rate=self.config.dropout_rate
        )
        
        # Combine both approaches: meta_adapter from PR + meta_weights from main
        self.meta_adapter = nn.Dense(hidden_dim)
        self.meta_weights = self.param(
            'meta_weights',
            nn.initializers.normal(0.02),
            (hidden_dim, hidden_dim)
        )
        self.adaptation_layer = nn.Dense(hidden_dim)
        self.layer_norm = nn.LayerNorm()
        
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
                 training: bool = False) -> jnp.ndarray:
        """Forward pass with enhanced meta-learning adaptation"""
        batch_size, seq_len, hidden_size = x.shape
        
        # Apply meta-adaptation (from PR branch)
        adapted_x = self.meta_adapter(x)
        
        # Apply attention with adaptation
        attended = self.attention(adapted_x, adapted_x, mask=mask, 
                                deterministic=not training)
        
        # Meta-learning adaptation (from main branch)
        meta_context = jnp.dot(x, self.meta_weights)
        adapted_features = self.adaptation_layer(meta_context)
        
        # Combine all features: attention + PR meta-adaptation + main meta-features
        combined = attended + adapted_features
        output = self.layer_norm(x + combined)
        
        return output
    
    def fast_adapt(self, params, support_x, support_y, query_x, num_steps=None):
        """Fast adaptation for few-shot learning (MAML-style inner loop).

        Args:
            params: Current model parameters (from ``self.variables['params']``).
            support_x: Support set inputs  — (batch, support_len, hidden).
            support_y: Support set targets — (batch, support_len, hidden).
            query_x:   Query set inputs    — (batch, query_len, hidden).
            num_steps: Number of inner-loop gradient steps (default: config value).

        Returns:
            Predictions on ``query_x`` using adapted parameters.
        """
        num_steps = num_steps or self.config.adaptation_steps
        adapted_params = self.adapt_parameters(
            params, support_x, support_y, num_steps
        )
        return self.apply_adapted_params(query_x, adapted_params)

    def adapt_parameters(self, params, support_x, support_y, num_steps=None):
        """Adapt meta_weights via gradient descent on the support set.

        Performs ``num_steps`` gradient updates using the MSE between the
        meta-projection of ``support_x`` and ``support_y``.
        """
        num_steps = num_steps or self.config.adaptation_steps
        lr = self.config.meta_learning_rate
        meta_weights = params['meta_weights']

        def _loss(mw):
            pred = jnp.dot(support_x, mw)
            return jnp.mean((pred - support_y) ** 2)

        for _ in range(num_steps):
            grad = jax.grad(_loss)(meta_weights)
            meta_weights = meta_weights - lr * grad

        return meta_weights

    def apply_adapted_params(self, query_x, adapted_meta_weights):
        """Apply adapted meta_weights to the query set.

        Args:
            query_x: Query inputs — (batch, query_len, hidden).
            adapted_meta_weights: Meta-weights after inner-loop adaptation.

        Returns:
            Projected query features — (batch, query_len, hidden).
        """
        return jnp.dot(query_x, adapted_meta_weights)

# Alias for backward compatibility
class MetaLearningAttention(MetaLA):
    """Backward compatibility alias."""
    pass

def main():
    # Main function for this module.
    logger.info("Module meta_la.py starting")
    # Test MetaLA configuration with updated parameter names
    config = MetaLAConfig(hidden_dim=768, num_heads=12)
    logger.info(f" MetaLA config created: {config}")
    return True

if __name__ == "__main__":
    main()
