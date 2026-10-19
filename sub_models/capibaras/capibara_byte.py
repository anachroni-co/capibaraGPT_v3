"""Optimized Capibara implementation for TPUs."""

import os
import sys

import logging

# Gets the path of the current directory (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Goes up one level to get the project root -> /.../CapibaraGPT v3
project_root = os.path.dirname(script_dir)
# Adds the project root to sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from functools import partial

from .tpu_base_config import TPUBaseConfig
from capibara.interfaces.isub_models import nfigTPU, PrecisionMode
from capibara.core.arm_optimizations import ARM_OPTIMIZATIONS, HARDWARE_INFO

logger = logging.getLogger(__name__)

class CapibaraByteConfig(TPUBaseConfig):
    """Specific configuration for CapibaraByte."""
    context_window: int = 128
    cache_size: int = 1024
    use_hybrid_sharding: bool = True

class CapibaraByte(nn.Module, ISubModel):
    """Optimized Capibara implementation for TPUs.

    Features:
    - Hybrid sharding for TPUs
    - Mixed precision
    - JIT-compatible cache
    - Memory optimizations
    """
    config: CapibaraByteConfig
    
    def setup(self):
        """Initializes optimized components for TPU."""
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.config.dtype,
            name="byte_dense"
        )
        self.norm = nn.LayerNorm(
            dtype=self.config.dtype,
            name="byte_norm"
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def _compute_context_weight(self, context_embedding: jnp.ndarray) -> jnp.ndarray:
        """Computes context weight in a JIT-compatible way."""
        return jnp.mean(context_embedding, axis=-1)
        
    def __call__(
        self,
        x: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        training: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Forward pass optimized for ARM Axion.

        Args:
            x: Input tensor
            context: Optional context tensor
            training: Training mode
            **kwargs: Additional arguments

        Returns:
            Dict with output and metrics
        """
        # Detect ARM Axion
        use_arm = not training and HARDWARE_INFO["is_axion"]

        # Normalize input
        x = self.norm(x)

        # Process context if it exists
        if context is not None:
            if use_arm and ARM_OPTIMIZATIONS.get("sve2_vectorization"):
                # Use SVE2 for weight calculation
                context_weight = ARM_OPTIMIZATIONS["arm_optimization_suite"].sve2_context_weight(
                    context,
                    self.dense.weight
                )
            else:
                context_weight = self._compute_context_weight(context)
            x = x * context_weight[..., None]
        
        # Dense projection with ARM optimizations
        if use_arm:
            if ARM_OPTIMIZATIONS.get("arm_kleidi"):
                # Use Kleidi for matmul
                x = ARM_OPTIMIZATIONS["arm_optimization_suite"].kleidi_forward(
                    x,
                    self.dense.weight
                )
            elif ARM_OPTIMIZATIONS.get("arm_quantization_available"):
                # Use ARM quantization
                x = ARM_OPTIMIZATIONS["arm_optimization_suite"].quantized_matmul(
                    x,
                    self.dense.weight
                )
            else:
                x = self.dense(x)
        else:
            x = self.dense(x)
        
        # Apply dropout if necessary
        if training and self.config.dropout_rate > 0:
            x = nn.Dropout(self.config.dropout_rate)(
                x, deterministic=False
            )
        
        # Metrics
        metrics = {
            "input_norm": jnp.linalg.norm(x),
            "output_norm": jnp.linalg.norm(x),
            "context_weight": context_weight if context is not None else 0.0,
            "arm_optimizations_used": use_arm,
            "arm_features": list(ARM_OPTIMIZATIONS.keys()) if use_arm else []
        }
        
        return {
            "output": x,
            "metrics": metrics
        }

# Extended configuration
config = ByteConfigTPU(
    input_dim=256,
    hidden_size=512,
    dtype=jnp.bfloat16,  # better for TPUs
    precision=PrecisionMode.BFLOAT16,
    update_rate=0.1,
    use_memory_efficient=True,
    gradient_checkpointing=True,
    residual_connections=True
)

# Model initialization
model = TPUCapibaraByte(config)
input_data = jnp.ones((1, 128, 256), dtype=config.dtype)  # (batch, seq_len, input_dim)
context_data = None  # Pass context here if your forward accepts it

# Initialize parameters
params = model.init(jax.random.PRNGKey(0), input_data)

# Forward pass with all features
final_state, all_states = model.apply(
    params,
    x=input_data,
    initial_state=None,  # Or your initial state if you have one
    training=True
)
logger.info("Final state shape:", final_state.shape)
logger.info("All states shape:", all_states.shape)