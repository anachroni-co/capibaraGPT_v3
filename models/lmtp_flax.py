"""models/lmtp_flax.py

Capibara Slim — L-MTP Flax/JAX model (arXiv:2505.17505, NeurIPS 2025).

Provides a full Flax implementation of the backbone (SlimFlaxModel) and the
L-MTP look-backward prediction heads (LMTPHeads), plus a combined wrapper
(LMTPWrapper) ready for two-stage training and JIT-compiled inference.

Architecture recap
------------------
  SlimFlaxModel  — hybrid Transformer + Mamba backbone
    ├─ Embedding (vocab → hidden)
    ├─ blocks[0..N]
    │    ├─ FlaxTransformerBlock  (RMSNorm → RoPE-Attention → SwiGLU MLP)
    │    └─ FlaxMambaBlock        (conv1d → ZOH SSM via lax.scan → SiLU gate)
    ├─ RMSNorm
    └─ lm_head  (weight-tied to embedding)

  LMTPHeads — n_head linear heads; head i takes [h_prev; h_curr] → logits
    tokens_per_step = leap_k × (n_head − 1) + 1   (e.g. k=4,n=4 → 13)

  LMTPWrapper — SlimFlaxModel + LMTPHeads, exposes
    • __call__(input_ids) → (ntp_logits, lmtp_logits_list, hidden)
    • decode_step(tokens, h_prev) → (step_tokens, h_curr)

Usage
-----
    from models.lmtp_flax import LMTPConfig, LMTPWrapper, create_lmtp_model
    from models.architecture import SlimConfig

    cfg = LMTPConfig(backbone=SlimConfig.preset("1.5b"), n_head=4, leap_k=2)
    model = LMTPWrapper(cfg)

    # Initialise params
    import jax, jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 16), dtype=jnp.int32)
    params = model.init(key, dummy)

Requires
--------
    pip install flax optax jax[cpu]    # CPU
    pip install jax[cuda12]            # GPU
    pip install jax[tpu]               # TPU / Axion
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional imports — every public symbol gets a no-op stub if unavailable
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    _JAX = True
except ImportError:
    _JAX = False

try:
    import flax.linen as nn
    _FLAX = True
except ImportError:
    _FLAX = False

_AVAILABLE = _JAX and _FLAX


def _require(name: str = "LMTPWrapper") -> None:
    if not _AVAILABLE:
        missing = []
        if not _JAX:
            missing.append("jax")
        if not _FLAX:
            missing.append("flax")
        raise ImportError(
            f"{name} requires JAX + Flax. Install with: "
            f"pip install {' '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LMTPConfig:
    """Combined config for backbone + L-MTP heads.

    Attributes:
        backbone:       SlimConfig for the transformer/mamba backbone.
        n_head:         Number of L-MTP prediction heads (≥ 1).
        leap_k:         Stride between consecutive head predictions.
                        tokens_per_step = leap_k × (n_head − 1) + 1.
        warmup_frac:    Fraction of total steps used for head-only warm-up.
        dtype:          Compute dtype (jnp.float32 / jnp.bfloat16).
    """
    # Import here to avoid circular at module level
    backbone: object = None          # SlimConfig; set in create_lmtp_model
    n_head: int = 4
    leap_k: int = 2
    warmup_frac: float = 0.25
    dtype: str = "float32"           # "float32" | "bfloat16"

    def tokens_per_step(self) -> int:
        return self.leap_k * (self.n_head - 1) + 1


# ---------------------------------------------------------------------------
# Building blocks (Flax modules) — defined only when Flax is available
# ---------------------------------------------------------------------------

if _AVAILABLE:

    # ---- RMSNorm -----------------------------------------------------------

    class FlaxRMSNorm(nn.Module):
        dim: int
        eps: float = 1e-5
        dtype: jnp.dtype = jnp.float32

        @nn.compact
        def __call__(self, x):
            scale = self.param("scale", nn.initializers.ones, (self.dim,))
            rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
            return scale * x / rms

    # ---- Rotary Positional Embedding ----------------------------------------

    def _make_rope_freqs(dim: int, max_len: int, dtype) -> jnp.ndarray:
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        t = jnp.arange(max_len, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)                       # (max_len, dim/2)
        emb = jnp.concatenate([freqs, freqs], axis=-1)       # (max_len, dim)
        return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)

    def _rotate_half(x):
        d = x.shape[-1] // 2
        return jnp.concatenate([-x[..., d:], x[..., :d]], axis=-1)

    def _apply_rope(q, k, cos, sin):
        # q, k: (B, H, L, D);  cos/sin: (L, D)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

    # ---- Attention ---------------------------------------------------------

    class FlaxSlimAttention(nn.Module):
        hidden_size: int
        num_heads: int
        max_seq_len: int
        rms_norm_eps: float = 1e-5
        dropout_rate: float = 0.0
        dtype: jnp.dtype = jnp.float32

        @nn.compact
        def __call__(self, x, mask=None, deterministic: bool = True):
            B, L, _ = x.shape
            H = self.num_heads
            D = self.hidden_size // H

            residual = x
            x = FlaxRMSNorm(self.hidden_size, self.rms_norm_eps, self.dtype)(x)

            q = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)(x)
            k = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)(x)
            v = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)(x)

            # Reshape to (B, H, L, D)
            q = q.reshape(B, L, H, D).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, H, D).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, H, D).transpose(0, 2, 1, 3)

            cos, sin = _make_rope_freqs(D, self.max_seq_len, self.dtype)
            cos, sin = cos[:L], sin[:L]
            q, k = _apply_rope(q, k, cos, sin)

            scale = D ** -0.5
            attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale

            if mask is not None:
                attn = jnp.where(mask, jnp.finfo(self.dtype).min, attn)
            attn = jax.nn.softmax(attn, axis=-1)
            if self.dropout_rate > 0 and not deterministic:
                attn = nn.Dropout(self.dropout_rate)(attn, deterministic=False)

            out = jnp.einsum("bhij,bhjd->bhid", attn, v)
            out = out.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
            out = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)(out)
            return out + residual

    # ---- SwiGLU MLP --------------------------------------------------------

    class FlaxSlimMLP(nn.Module):
        hidden_size: int
        intermediate_size: int
        rms_norm_eps: float = 1e-5
        dtype: jnp.dtype = jnp.float32

        @nn.compact
        def __call__(self, x):
            residual = x
            x = FlaxRMSNorm(self.hidden_size, self.rms_norm_eps, self.dtype)(x)
            gate = nn.Dense(self.intermediate_size, use_bias=False, dtype=self.dtype)(x)
            up   = nn.Dense(self.intermediate_size, use_bias=False, dtype=self.dtype)(x)
            x    = jax.nn.silu(gate) * up
            x    = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)(x)
            return x + residual

    # ---- Transformer Block -------------------------------------------------

    class FlaxTransformerBlock(nn.Module):
        hidden_size: int
        num_heads: int
        intermediate_size: int
        max_seq_len: int
        rms_norm_eps: float = 1e-5
        dropout_rate: float = 0.0
        dtype: jnp.dtype = jnp.float32

        @nn.compact
        def __call__(self, x, mask=None, deterministic: bool = True):
            x = FlaxSlimAttention(
                self.hidden_size, self.num_heads, self.max_seq_len,
                self.rms_norm_eps, self.dropout_rate, self.dtype,
            )(x, mask=mask, deterministic=deterministic)
            x = FlaxSlimMLP(
                self.hidden_size, self.intermediate_size,
                self.rms_norm_eps, self.dtype,
            )(x)
            return x

    # ---- Mamba Block (lax.scan SSM) ----------------------------------------

    class FlaxMambaBlock(nn.Module):
        """Selective SSM via lax.scan — O(L) memory, fully JIT-able.

        Uses zero-order-hold discretisation (same as architecture.py MambaBlock)
        but replaces the Python for-loop with jax.lax.scan for XLA compilation.
        """
        hidden_size: int
        d_state: int = 16
        d_conv: int = 4
        expand: int = 2
        rms_norm_eps: float = 1e-5
        dtype: jnp.dtype = jnp.float32

        @nn.compact
        def __call__(self, x, mask=None, deterministic: bool = True):
            B, L, d = x.shape
            d_inner = self.expand * self.hidden_size
            residual = x
            x = FlaxRMSNorm(self.hidden_size, self.rms_norm_eps, self.dtype)(x)

            # in_proj splits into x_part and z gate
            xz = nn.Dense(d_inner * 2, use_bias=False, dtype=self.dtype)(x)
            x_part, z = xz[..., :d_inner], xz[..., d_inner:]

            # Depthwise conv1d (causal)
            conv_w = self.param(
                "conv_w",
                nn.initializers.normal(0.02),
                (self.d_conv, 1, d_inner),
            )
            conv_b = self.param("conv_b", nn.initializers.zeros, (d_inner,))
            # Pad left by (d_conv-1) for causality
            x_pad = jnp.pad(x_part, ((0, 0), (self.d_conv - 1, 0), (0, 0)))
            x_conv = lax.conv_general_dilated(
                x_pad.transpose(0, 2, 1),          # (B, d_inner, L+pad)
                conv_w.transpose(1, 2, 0),          # (d_inner, 1, d_conv)
                window_strides=(1,),
                padding="VALID",
                feature_group_count=d_inner,
            ).transpose(0, 2, 1) + conv_b           # (B, L, d_inner)
            x_conv = jax.nn.silu(x_conv)

            # Input-dependent SSM params
            x_dbl = nn.Dense(
                self.d_state * 2 + 1, use_bias=False, dtype=self.dtype
            )(x_conv)
            dt_raw = x_dbl[..., :1]
            B_mat  = x_dbl[..., 1: 1 + self.d_state]
            C_mat  = x_dbl[..., 1 + self.d_state:]

            dt_proj = nn.Dense(d_inner, use_bias=True, dtype=self.dtype)
            dt = jax.nn.softplus(dt_proj(dt_raw))   # (B, L, d_inner)

            A_log = self.param(
                "A_log",
                lambda _, s: jnp.log(
                    jnp.arange(1, s[1] + 1, dtype=jnp.float32).reshape(1, -1).repeat(s[0], 0)
                ),
                (d_inner, self.d_state),
            )
            A = -jnp.exp(A_log.astype(jnp.float32))  # (d_inner, d_state)
            D_param = self.param("D", nn.initializers.ones, (d_inner,))

            # ZOH discretisation
            # dA: (B, L, d_inner, d_state)
            dA = jnp.exp(jnp.einsum("bld,dn->bldn", dt, A))
            # dB: (B, L, d_inner, d_state)
            dB = jnp.einsum("bld,bln->bldn", dt, B_mat)

            # Sequential scan via lax.scan (replaces Python for-loop)
            def scan_fn(h, t_inputs):
                dA_t, dB_t, x_t, C_t = t_inputs   # shapes: (B,d,N), (B,d,N), (B,d), (B,N)
                h = dA_t * h + dB_t * x_t[..., None]
                y = jnp.einsum("bdn,bn->bd", h, C_t)
                return h, y

            init_h = jnp.zeros((B, d_inner, self.d_state), dtype=self.dtype)
            # Transpose time to front for scan: (L, B, ...)
            _, ys = lax.scan(
                scan_fn,
                init_h,
                (
                    dA.transpose(1, 0, 2, 3),           # (L, B, d_inner, d_state)
                    dB.transpose(1, 0, 2, 3),
                    x_conv.transpose(1, 0, 2),           # (L, B, d_inner)
                    C_mat.transpose(1, 0, 2),            # (L, B, d_state)
                ),
            )
            y = ys.transpose(1, 0, 2)                   # (B, L, d_inner)
            y = y + x_conv * D_param
            y = y * jax.nn.silu(z)

            out = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)(y)
            return out + residual

    # ---- SlimFlaxModel (backbone) ------------------------------------------

    class SlimFlaxModel(nn.Module):
        """Capibara Slim backbone — hybrid Transformer + Mamba in Flax."""
        hidden_size: int
        num_layers: int
        num_heads: int
        intermediate_size: int
        vocab_size: int
        max_seq_len: int
        rms_norm_eps: float = 1e-5
        dropout: float = 0.0
        mamba_every_n: int = 0
        mamba_d_state: int = 16
        mamba_d_conv: int = 4
        mamba_expand: int = 2
        tie_embeddings: bool = True
        dtype: jnp.dtype = jnp.float32

        @classmethod
        def from_slim_config(cls, cfg, dtype=None):
            from models.architecture import SlimConfig  # noqa
            kw = dict(
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                intermediate_size=cfg.intermediate_size,
                vocab_size=cfg.vocab_size,
                max_seq_len=cfg.max_seq_len,
                rms_norm_eps=cfg.rms_norm_eps,
                dropout=cfg.dropout,
                mamba_every_n=cfg.mamba_every_n,
                mamba_d_state=cfg.mamba_d_state,
                mamba_d_conv=cfg.mamba_d_conv,
                mamba_expand=cfg.mamba_expand,
                tie_embeddings=cfg.tie_embeddings,
            )
            if dtype is not None:
                kw["dtype"] = dtype
            return cls(**kw)

        @nn.compact
        def __call__(
            self,
            input_ids,
            attention_mask=None,
            deterministic: bool = True,
            return_hidden: bool = False,
        ):
            B, L = input_ids.shape

            # Causal mask
            mask = jnp.triu(jnp.ones((L, L), dtype=bool), k=1)  # (L, L)
            mask = mask[None, None, :, :]                          # (1, 1, L, L)
            if attention_mask is not None:
                # Also mask padding tokens
                pad = (attention_mask[:, None, None, :] == 0)
                mask = mask | pad

            x = nn.Embed(
                self.vocab_size, self.hidden_size,
                dtype=self.dtype, name="embed",
            )(input_ids)

            for i in range(self.num_layers):
                use_mamba = self.mamba_every_n > 0 and (i % self.mamba_every_n == 1)
                if use_mamba:
                    x = FlaxMambaBlock(
                        self.hidden_size,
                        d_state=self.mamba_d_state,
                        d_conv=self.mamba_d_conv,
                        expand=self.mamba_expand,
                        rms_norm_eps=self.rms_norm_eps,
                        dtype=self.dtype,
                        name=f"mamba_{i}",
                    )(x, deterministic=deterministic)
                else:
                    x = FlaxTransformerBlock(
                        self.hidden_size,
                        self.num_heads,
                        self.intermediate_size,
                        self.max_seq_len,
                        self.rms_norm_eps,
                        self.dropout,
                        self.dtype,
                        name=f"transformer_{i}",
                    )(x, mask=mask, deterministic=deterministic)

            hidden = FlaxRMSNorm(
                self.hidden_size, self.rms_norm_eps, self.dtype, name="norm"
            )(x)

            # LM head — weight tying handled externally via params sharing
            logits = nn.Dense(
                self.vocab_size, use_bias=False, dtype=self.dtype, name="lm_head"
            )(hidden)

            if return_hidden:
                return logits, hidden
            return logits

    # ---- L-MTP Heads -------------------------------------------------------

    class LMTPHeads(nn.Module):
        """n_head linear heads for look-backward L-MTP.

        Each head i takes the concatenation [h_prev; h_curr] (2×hidden_size)
        and predicts token at offset (i+1)×leap_k ahead of the current position.
        """
        hidden_size: int
        vocab_size: int
        n_head: int = 4
        leap_k: int = 2
        dtype: jnp.dtype = jnp.float32

        def tokens_per_step(self) -> int:
            return self.leap_k * (self.n_head - 1) + 1

        @nn.compact
        def __call__(
            self,
            h_prev: jnp.ndarray,    # (B, T, hidden)
            h_curr: jnp.ndarray,    # (B, T, hidden)
        ) -> List[jnp.ndarray]:
            """Return list of n_head logit tensors, each (B, T, vocab)."""
            x = jnp.concatenate([h_prev, h_curr], axis=-1)   # (B, T, 2H)
            logits_list = []
            for i in range(self.n_head):
                logits_list.append(
                    nn.Dense(
                        self.vocab_size, use_bias=False, dtype=self.dtype,
                        name=f"head_{i}",
                    )(x)
                )
            return logits_list

    # ---- LMTPWrapper -------------------------------------------------------

    class LMTPWrapper(nn.Module):
        """Backbone + L-MTP heads — the combined model for training & inference.

        Forward pass returns:
            (ntp_logits, lmtp_logits_list, hidden_states)

        where:
            ntp_logits      (B, T, V)   — backbone next-token prediction
            lmtp_logits_list            — list of n_head (B, T, V) tensors
            hidden_states   (B, T, H)   — final backbone hidden states
        """
        cfg: LMTPConfig

        def setup(self):
            _require("LMTPWrapper")
            from models.architecture import SlimConfig  # noqa
            sc = self.cfg.backbone
            dtype = (jnp.bfloat16 if self.cfg.dtype == "bfloat16" else jnp.float32)
            self.backbone = SlimFlaxModel.from_slim_config(sc, dtype=dtype)
            self.heads = LMTPHeads(
                hidden_size=sc.hidden_size,
                vocab_size=sc.vocab_size,
                n_head=self.cfg.n_head,
                leap_k=self.cfg.leap_k,
                dtype=dtype,
            )

        def __call__(
            self,
            input_ids,
            attention_mask=None,
            deterministic: bool = True,
        ):
            ntp_logits, hidden = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                deterministic=deterministic,
                return_hidden=True,
            )
            # look-backward: shift hidden right (zeros for first position)
            h_prev = jnp.concatenate(
                [jnp.zeros_like(hidden[:, :1, :]), hidden[:, :-1, :]],
                axis=1,
            )
            lmtp_logits = self.heads(h_prev, hidden)
            return ntp_logits, lmtp_logits, hidden

        def decode_step(
            self,
            token_ids: jnp.ndarray,   # (B, T) — full context so far
            h_prev: jnp.ndarray,      # (B, 1, H) — previous step hidden
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Single look-backward decode step.

            Returns:
                step_tokens  (B, tokens_per_step)  — greedy token candidates
                h_curr       (B, 1, H)             — hidden state at last pos
            """
            _, hidden = self.backbone(token_ids, deterministic=True, return_hidden=True)
            h_curr = hidden[:, -1:, :]     # (B, 1, H)
            # Heads use the last hidden state position
            lmtp_logits = self.heads(h_prev, h_curr)   # list of (B, 1, V)

            tps = self.heads.tokens_per_step()
            # Interleave head predictions with leap_k-1 gaps (greedy repeat-fill)
            tokens: List[jnp.ndarray] = []
            for i, logits in enumerate(lmtp_logits):
                tok = logits[:, 0, :].argmax(-1, keepdims=True)  # (B, 1)
                tokens.append(tok)
                if i < self.cfg.n_head - 1:
                    for _ in range(self.cfg.leap_k - 1):
                        tokens.append(tok)
            step_tokens = jnp.concatenate(tokens[:tps], axis=1)  # (B, tps)
            return step_tokens, h_curr


else:
    # Stubs when JAX/Flax unavailable

    class FlaxRMSNorm:          # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require("FlaxRMSNorm")

    class FlaxSlimAttention:    # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require("FlaxSlimAttention")

    class FlaxMambaBlock:       # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require("FlaxMambaBlock")

    class FlaxTransformerBlock: # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require("FlaxTransformerBlock")

    class SlimFlaxModel:        # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require("SlimFlaxModel")

    class LMTPHeads:            # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require("LMTPHeads")

    class LMTPWrapper:          # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require("LMTPWrapper")


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_lmtp_model(
    backbone_preset: str = "1.5b",
    n_head: int = 4,
    leap_k: int = 2,
    dtype: str = "bfloat16",
) -> "LMTPWrapper":
    """Create an LMTPWrapper from a SlimConfig preset.

    Args:
        backbone_preset:  "1.5b" | "3b" | "7b"
        n_head:           Number of L-MTP prediction heads.
        leap_k:           Leap stride; tokens_per_step = leap_k*(n_head-1)+1.
        dtype:            "float32" | "bfloat16" (bfloat16 recommended for TPU).

    Returns:
        LMTPWrapper (Flax Module — call .init() to obtain parameters).
    """
    _require("create_lmtp_model")
    from models.architecture import SlimConfig
    cfg = LMTPConfig(
        backbone=SlimConfig.preset(backbone_preset),
        n_head=n_head,
        leap_k=leap_k,
        dtype=dtype,
    )
    return LMTPWrapper(cfg)


__all__ = [
    "LMTPConfig",
    "LMTPWrapper",
    "LMTPHeads",
    "SlimFlaxModel",
    "FlaxTransformerBlock",
    "FlaxMambaBlock",
    "FlaxRMSNorm",
    "create_lmtp_model",
    "_AVAILABLE",
]
