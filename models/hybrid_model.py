"""CapibaraHybridLM — trainable language model over the hybrid SSM+attention stack.

This wires the previously disconnected building blocks into a real model:

- ``layers/ssm_hybrid_layers.HybridLayerStack`` — interleaved SSM/attention
  depth (Jamba/Zamba-style), the intended core architecture.
- ``layers/sparsity/bitnet.BitNet158`` — optional 1.58-bit LM head for
  cheap inference on CPU/ARM (``lm_head="bitnet"``).
- ``sub_models/experimental/spike_ssm.SpikeSSM`` — optional spiking block
  appended after the stack for energy-efficiency experiments
  (``use_spike_block=True``).

Usage:
    from models.hybrid_model import CapibaraHybridLM, HybridLMConfig

    cfg = HybridLMConfig(vocab_size=50257, num_layers=12, hidden_size=768)
    model = CapibaraHybridLM(config=cfg)
    params = model.init(jax.random.PRNGKey(0), tokens)   # tokens: [B, T] int32
    logits = model.apply(params, tokens)                  # [B, T, vocab]

Select from config/config.yaml with ``model.architecture.type: "hybrid"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import jax.numpy as jnp
from flax import linen as nn

from layers.ssm_hybrid_layers import HybridLayerStack, HybridLayerStackConfig
from layers.sparsity.bitnet import BitNet158
from sub_models.experimental.spike_ssm import SpikeSSM

logger = logging.getLogger(__name__)

__all__ = ["CapibaraHybridLM", "HybridLMConfig"]


@dataclass
class HybridLMConfig:
    """Configuration for CapibaraHybridLM.

    Attributes:
        vocab_size: Token vocabulary size.
        num_layers: Depth of the hybrid stack.
        hidden_size: Residual stream width.
        num_heads: Attention heads (attention layers only).
        d_state: SSM state dimension (SSM layers only).
        max_seq_length: Maximum sequence length (positional embeddings).
        dropout_rate: Dropout inside attention blocks (0.0 for pretraining).
        ffn_mult: FFN width multiplier.
        ssm_layers / attention_layers: Explicit layer-index assignment;
            None = interleaved default (SSM even, attention odd).
        lm_head: "dense" (default) or "bitnet" (1.58-bit BitNet158 head
            for cheap CPU/ARM inference).
        use_spike_block: Append a SpikeSSM block after the stack
            (energy-efficiency experiments; surrogate gradients flow).
        spike_state_dim: State dimension of the optional spike block.
        tie_embeddings: Reuse the token-embedding matrix as LM head
            (ignored when lm_head="bitnet").
    """

    vocab_size: int = 50257
    num_layers: int = 12
    hidden_size: int = 768
    num_heads: int = 12
    d_state: int = 64
    max_seq_length: int = 2048
    dropout_rate: float = 0.0
    ffn_mult: int = 4
    ssm_layers: Optional[List[int]] = field(default=None)
    attention_layers: Optional[List[int]] = field(default=None)
    lm_head: str = "dense"
    use_spike_block: bool = False
    spike_state_dim: int = 64
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.lm_head not in ("dense", "bitnet"):
            raise ValueError(f"lm_head must be 'dense' or 'bitnet', got {self.lm_head!r}")

    def stack_config(self) -> HybridLayerStackConfig:
        return HybridLayerStackConfig(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            d_state=self.d_state,
            dropout_rate=self.dropout_rate,
            ffn_mult=self.ffn_mult,
            ssm_layers=self.ssm_layers,
            attention_layers=self.attention_layers,
        )


class CapibaraHybridLM(nn.Module):
    """Hybrid SSM+attention causal language model (Jamba-style interleaving)."""

    config: HybridLMConfig

    def setup(self) -> None:
        cfg = self.config
        self.token_embed = nn.Embed(
            num_embeddings=cfg.vocab_size, features=cfg.hidden_size
        )
        self.pos_embed = nn.Embed(
            num_embeddings=cfg.max_seq_length, features=cfg.hidden_size
        )
        self.stack = HybridLayerStack(config=cfg.stack_config())
        if cfg.use_spike_block:
            self.spike_block = SpikeSSM(
                hidden_size=cfg.hidden_size, state_dim=cfg.spike_state_dim
            )
        self.final_norm = nn.LayerNorm()
        if cfg.lm_head == "bitnet":
            self.head = BitNet158(features=cfg.vocab_size)
        elif not cfg.tie_embeddings:
            self.head = nn.Dense(cfg.vocab_size, use_bias=False)

    def __call__(
        self, tokens: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """Compute next-token logits.

        Args:
            tokens: [batch, seq_len] int token ids.
            deterministic: disable dropout when True.

        Returns:
            logits: [batch, seq_len, vocab_size].
        """
        cfg = self.config
        _, seq_len = tokens.shape
        if seq_len > cfg.max_seq_length:
            raise ValueError(
                f"seq_len {seq_len} > max_seq_length {cfg.max_seq_length}"
            )

        positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        x = self.token_embed(tokens) + self.pos_embed(positions)

        x = self.stack(x, training=not deterministic)

        if cfg.use_spike_block:
            x = x + self.spike_block(x)  # residual spiking refinement

        x = self.final_norm(x)

        if cfg.lm_head == "bitnet":
            return self.head(x)
        if cfg.tie_embeddings:
            return self.token_embed.attend(x)
        return self.head(x)
