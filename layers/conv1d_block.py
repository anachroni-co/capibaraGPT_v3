"""
Conv1D block for CapibaraGPT v3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from layers.jax_compat import jnp, nn, JAX_AVAILABLE
from layers.base import LayerConfig

logger = logging.getLogger(__name__)


@dataclass
class Conv1DBlockConfig(LayerConfig):
    """Configuration for Conv1DBlock."""
    features: int = 512
    kernel_size: int = 3
    use_layer_norm: bool = True


class Conv1DBlock(nn.Module):
    """Simple 1D convolutional block with optional layer norm."""
    config: Conv1DBlockConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        conv = nn.Conv(
            features=self.config.features,
            kernel_size=(self.config.kernel_size,),
            use_bias=self.config.use_bias,
        )
        y = conv(x)
        if self.config.use_layer_norm:
            y = nn.LayerNorm()(y)
        return nn.relu(y)


def main() -> bool:
    logger.info("Module conv1d_block.py starting")
    if not JAX_AVAILABLE:
        logger.warning("JAX/Flax not available; Conv1DBlock will not be usable.")
    return True


if __name__ == "__main__":
    main()
