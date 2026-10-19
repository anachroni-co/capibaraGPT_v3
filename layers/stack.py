"""
Layer stack utilities for CapibaraGPT v3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from layers.jax_compat import jnp, nn, JAX_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class LayerStackConfig:
    """Configuration for a simple stacked layer block."""
    num_layers: int = 2


class LayerStack(nn.Module):
    """Apply a sequence of layers to an input."""
    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for layer in self.layers:
            try:
                x = layer(x, training=training)
            except TypeError:
                x = layer(x)
        return x


def main() -> bool:
    logger.info("Module stack.py starting")
    if not JAX_AVAILABLE:
        logger.warning("JAX/Flax not available; LayerStack will not be usable.")
    return True


if __name__ == "__main__":
    main()
