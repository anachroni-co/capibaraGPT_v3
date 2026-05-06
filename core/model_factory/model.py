"""
CapibaraMoEModel: end-to-end Mixtral-style decoder-only Transformer with MoE.

Public surface:
- CapibaraMoEModel       : flax.linen.Module
- ModelOutput            : NamedTuple-like dataclass for forward outputs
- build_model(cfg)       : factory returning an unitialized nn.Module
- init_params(model, cfg): convenience helper to materialize parameters
- count_params(params)   : runtime parameter count from a PyTree
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from .config import ModelConfig
from .blocks import RMSNorm, TransformerBlock, _rope_freqs, make_causal_mask


def _resolve_dtype(name: str) -> jnp.dtype:
    return {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
    }[name]


@dataclass
class ModelOutput:
    """Forward pass outputs."""
    logits: jnp.ndarray                # (B, T, vocab_size)
    aux_loss: jnp.ndarray              # scalar; sum of layer load-balance terms


class CapibaraMoEModel(nn.Module):
    """
    Decoder-only Transformer with Mixtral-style MoE FFN, RoPE, GQA, RMSNorm.

    Forward signature:
        logits, aux = model.apply(params, input_ids, deterministic=True)
    where `aux` is the sum of load-balance auxiliary losses across MoE layers,
    multiplied by `cfg.load_balance_weight`. Add it to your training loss.
    """

    cfg: ModelConfig

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray, deterministic: bool = True) -> ModelOutput:
        cfg = self.cfg
        compute_dtype = _resolve_dtype(cfg.dtype)
        param_dtype = _resolve_dtype(cfg.param_dtype)

        B, T = input_ids.shape
        if T > cfg.max_seq_len:
            raise ValueError(
                f"input length {T} exceeds cfg.max_seq_len={cfg.max_seq_len}"
            )

        # Token embeddings (kept in param_dtype for stability, cast for compute).
        embed = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.d_model,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            name="embed",
        )
        x = embed(input_ids)

        # Precompute RoPE tables for max_seq_len; slice to T inside attention.
        cos, sin = _rope_freqs(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        mask = make_causal_mask(T, dtype=compute_dtype)

        aux_total = jnp.float32(0.0)
        for layer_idx in range(cfg.n_layers):
            x, aux = TransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                ff_dim=cfg.ff_dim,
                n_experts=cfg.n_experts,
                top_k=cfg.top_k,
                router_jitter=cfg.router_jitter,
                rms_norm_eps=cfg.rms_norm_eps,
                dtype=compute_dtype,
                param_dtype=param_dtype,
                name=f"layer_{layer_idx}",
            )(x, cos=cos, sin=sin, mask=mask, deterministic=deterministic)
            aux_total = aux_total + aux

        x = RMSNorm(
            cfg.d_model,
            eps=cfg.rms_norm_eps,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            name="final_norm",
        )(x)

        if cfg.tie_word_embeddings:
            # Reuse the embedding matrix as the output projection.
            logits = embed.attend(x.astype(param_dtype)).astype(jnp.float32)
        else:
            logits = nn.Dense(
                cfg.vocab_size,
                use_bias=False,
                dtype=jnp.float32,
                param_dtype=param_dtype,
                name="lm_head",
            )(x)

        return ModelOutput(
            logits=logits,
            aux_loss=cfg.load_balance_weight * aux_total,
        )


# ---------------------------------------------------------------------------
# factory + helpers
# ---------------------------------------------------------------------------


def build_model(cfg: ModelConfig) -> CapibaraMoEModel:
    """Return an uninitialized Flax module bound to the given config."""
    return CapibaraMoEModel(cfg=cfg)


def init_params(
    model: CapibaraMoEModel,
    cfg: ModelConfig,
    seed: int = 0,
    batch_size: int = 1,
    seq_len: int | None = None,
) -> Any:
    """Materialize parameters using a dummy forward pass."""
    seq_len = seq_len or min(8, cfg.max_seq_len)
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    dummy = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    variables = model.init(
        {"params": keys[0], "router": keys[1], "dropout": keys[2]},
        dummy,
        deterministic=True,
    )
    return variables["params"]


def count_params(params: Any) -> int:
    """Count total scalar parameters in a Flax param PyTree."""
    leaves, _ = jax.tree_util.tree_flatten(params)
    return int(sum(int(jnp.size(x)) for x in leaves))
