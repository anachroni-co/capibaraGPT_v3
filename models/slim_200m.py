"""Capibara Slim 200M — Flax decoder-only transformer (byte-level).

Architecture (pre-norm, LLaMA-style):
  Token embedding + learned positional embedding
  × N layers of: RMSNorm → CausalMHA → RMSNorm → SwiGLU FFN
  RMSNorm → LM head (weight-tied to embedding)

~200M parameters with default config (vocab=512, d=1024, L=18, H=16).

Usage:
    from models.slim_200m import Slim200M, ModelConfig
    model = Slim200M(ModelConfig())
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 128), jnp.int32))
    logits = model.apply(params, input_ids)   # (B, T, vocab_size)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    FLAX_AVAILABLE = True
except ImportError:
    jax = None   # type: ignore[assignment]
    jnp = None   # type: ignore[assignment]
    nn = None    # type: ignore[assignment]
    FLAX_AVAILABLE = False


@dataclass
class ModelConfig:
    vocab_size: int = 512          # byte-level: 256 bytes + special tokens
    hidden_size: int = 1024
    num_layers: int = 12           # 12 × ~16.8M ≈ 203M total
    num_heads: int = 16
    max_seq_len: int = 2048
    dropout_rate: float = 0.0     # disabled during inference; use 0.1 for training
    pad_token_id: int = 256

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def ffn_dim(self) -> int:
        return self.hidden_size * 4

    def param_count_estimate(self) -> int:
        d, V, L, ffn = self.hidden_size, self.vocab_size, self.num_layers, self.ffn_dim
        emb = V * d
        attn = 4 * d * d        # Q K V O projections
        ffn_params = 3 * d * ffn  # SwiGLU: gate + up + down
        norms = 2 * d            # two RMSNorms per layer (scale only)
        per_layer = attn + ffn_params + norms
        return emb + L * per_layer + d  # +d for final norm


if FLAX_AVAILABLE:

    class RMSNorm(nn.Module):
        eps: float = 1e-6

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
            rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
            return x / rms * scale

    class CausalSelfAttention(nn.Module):
        config: ModelConfig

        @nn.compact
        def __call__(
            self,
            x: jnp.ndarray,
            deterministic: bool = True,
        ) -> jnp.ndarray:
            B, T, C = x.shape
            cfg = self.config
            H, D = cfg.num_heads, cfg.head_dim

            q = nn.Dense(C, use_bias=False, name="q_proj")(x)
            k = nn.Dense(C, use_bias=False, name="k_proj")(x)
            v = nn.Dense(C, use_bias=False, name="v_proj")(x)

            q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)  # (B,H,T,D)
            k = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)

            scale = D ** -0.5
            attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale

            # Causal mask
            mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
            attn = jnp.where(mask[None, None], attn, jnp.finfo(attn.dtype).min)
            attn = jax.nn.softmax(attn, axis=-1)

            out = jnp.einsum("bhij,bhjd->bhid", attn, v)
            out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
            return nn.Dense(C, use_bias=False, name="o_proj")(out)

    class SwiGLUFFN(nn.Module):
        config: ModelConfig

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            ffn = self.config.ffn_dim
            gate = nn.Dense(ffn, use_bias=False, name="gate")(x)
            up   = nn.Dense(ffn, use_bias=False, name="up")(x)
            return nn.Dense(self.config.hidden_size, use_bias=False, name="down")(
                jax.nn.silu(gate) * up
            )

    class TransformerBlock(nn.Module):
        config: ModelConfig

        @nn.compact
        def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
            x = x + CausalSelfAttention(self.config)(RMSNorm()(x), deterministic)
            x = x + SwiGLUFFN(self.config)(RMSNorm()(x))
            return x

    class Slim200M(nn.Module):
        """Byte-level decoder-only transformer, ~200M params."""
        config: ModelConfig

        @nn.compact
        def __call__(
            self,
            input_ids: jnp.ndarray,          # (B, T) int32
            deterministic: bool = True,
        ) -> jnp.ndarray:                    # (B, T, vocab_size)
            cfg = self.config
            B, T = input_ids.shape

            tok_emb = nn.Embed(cfg.vocab_size, cfg.hidden_size, name="tok_emb")
            pos_emb = self.param(
                "pos_emb",
                nn.initializers.normal(stddev=0.02),
                (1, cfg.max_seq_len, cfg.hidden_size),
            )

            x = tok_emb(input_ids) + pos_emb[:, :T, :]

            for i in range(cfg.num_layers):
                x = TransformerBlock(cfg, name=f"block_{i}")(x, deterministic)

            x = RMSNorm(name="final_norm")(x)

            # Weight-tied LM head
            logits = tok_emb.attend(x)      # (B, T, vocab_size)
            return logits

else:
    class Slim200M:  # type: ignore[no-redef]
        """Stub — install flax to use the real model."""
        def __init__(self, config: ModelConfig):
            raise ImportError("flax is required: pip install flax")


def build_model(config: Optional[ModelConfig] = None) -> "Slim200M":
    """Convenience factory."""
    if not FLAX_AVAILABLE:
        raise ImportError("flax is required: pip install flax")
    return Slim200M(config or ModelConfig())


def count_params(params: dict) -> int:
    """Count total trainable parameters in a param tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
