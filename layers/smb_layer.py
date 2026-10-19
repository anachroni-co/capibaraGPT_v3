"""
SMB (Simple MLP Block) layer for CapibaraGPT v3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from layers.jax_compat import jnp, nn, JAX_AVAILABLE
from layers.base import LayerConfig

logger = logging.getLogger(__name__)


@dataclass
class SMBLayerConfig(LayerConfig):
    """Configuration for SMBLayer."""
    expansion: int = 4
    dropout_rate: float = 0.1


class SMBLayer(nn.Module):
    """Simple feed-forward block."""
    config: SMBLayerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        hidden = nn.Dense(self.config.hidden_size * self.config.expansion)(x)
        hidden = nn.relu(hidden)
        hidden = nn.Dropout(rate=self.config.dropout_rate)(hidden, deterministic=not training)
        out = nn.Dense(self.config.hidden_size)(hidden)
        return out


def main() -> bool:
    logger.info("Module smb_layer.py starting")
    if not JAX_AVAILABLE:
        logger.warning("JAX/Flax not available; SMBLayer will not be usable.")
    return True


if __name__ == "__main__":
    main()
