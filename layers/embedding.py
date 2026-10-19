"""
Embedding layer for CapibaraGPT v3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from layers.jax_compat import jnp, nn, JAX_AVAILABLE
from layers.base import LayerConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig(LayerConfig):
    """Configuration for EmbeddingLayer."""
    vocab_size: int = 50257
    embedding_dim: int = 768


class EmbeddingLayer(nn.Module):
    """Token embedding layer."""
    config: EmbeddingConfig

    @nn.compact
    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.embedding_dim,
        )
        return embed(token_ids)


def main() -> bool:
    logger.info("Module embedding.py starting")
    if not JAX_AVAILABLE:
        logger.warning("JAX/Flax not available; EmbeddingLayer will not be usable.")
    return True


if __name__ == "__main__":
    main()
