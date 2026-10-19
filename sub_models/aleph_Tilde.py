"""Aleph-TILDE (Neural Rule Induction Submodel) – TRAINABLE"""

import logging
from typing import Dict, Any, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
from capibara.interfaces.isub_models import ISubModel
from capibara.core.arm_optimizations import HARDWARE_INFO

logger = logging.getLogger(__name__)

# ============================================================================
# SUBMODEL
# ============================================================================

class AlephTilde(nn.Module, ISubModel):
    """
    Neural proxy of Aleph-TILDE for symbolic rule induction.
    """
    hidden_size: int
    dropout_rate: float = 0.1

    def setup(self):
        self.hypothesis = nn.Dense(self.hidden_size)
        self.background = nn.Dense(self.hidden_size)
        self.combine = nn.Dense(self.hidden_size)
        self.rules = nn.Dense(self.hidden_size)
        self.output = nn.Dense(self.hidden_size)

        self.norm = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
        **kwargs: Any
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:

        if x.ndim != 3:
            raise ValueError(
                f"AlephTilde expects (batch, seq, dim), got {x.shape}"
            )

        # ARM Axion detection (INFO ONLY – no graph break)
        use_arm = deterministic and HARDWARE_INFO.get("is_axion", False)

        # Hypothesis generation
        h = self.hypothesis(x)
        h = self.norm(h)
        h = self.dropout(h, deterministic=deterministic)
        h = jax.nn.gelu(h)

        # Background knowledge projection
        b = self.background(x)
        b = self.norm(b)
        b = self.dropout(b, deterministic=deterministic)

        # Combine hypothesis + background
        combined = jnp.concatenate([h, b], axis=-1)
        combined = self.combine(combined)
        combined = jax.nn.gelu(combined)

        # Rule induction
        r = self.rules(combined)
        r = self.norm(r)
        r = self.dropout(r, deterministic=deterministic)

        # Final symbolic embedding
        out = self.output(r)

        # Metrics (JAX-safe)
        rule_energy = jnp.mean(jnp.linalg.norm(r, axis=-1))
        sparsity = jnp.mean(jnp.abs(r) < 1e-3)

        metrics = {
            "rule_energy": rule_energy,
            "rule_sparsity": sparsity,
            "arm_backend": jnp.array(use_arm, dtype=jnp.int32),
        }

        return out, metrics