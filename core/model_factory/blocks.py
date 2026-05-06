"""
Flax nn.Module building blocks for the CapibaraGPT MoE Transformer.

Components:
- RMSNorm                  : Llama/Mixtral-style RMS layer norm.
- apply_rotary             : RoPE applied to (Q, K) tensors.
- RotaryFreqs              : precomputed cos/sin tables.
- GroupedQueryAttention    : GQA with KV-head broadcast.
- SwiGLUExpert             : single-expert FFN (gate, up, down).
- TopKRouter               : softmax router with load-balance auxiliary loss.
- SparseMoEBlock           : top-k expert dispatch over a SwiGLU expert pool.
- TransformerBlock         : pre-norm residual block (attn + MoE).

All modules are pure nn.Module: no I/O, no global state, no random calls outside
of `make_rng`. Param counts match the analytic estimates in config.ModelConfig.
"""
from __future__ import annotations

from dataclasses import field
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

# ----------------------------------------------------------------------------
# RMSNorm
# ----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (no bias, learned scale)."""

    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param(
            "scale",
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )
        x32 = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
        x32 = x32 * jax.lax.rsqrt(var + self.eps)
        return (x32 * scale.astype(jnp.float32)).astype(self.dtype)


# ----------------------------------------------------------------------------
# Rotary positional embedding (RoPE)
# ----------------------------------------------------------------------------


def _rope_freqs(seq_len: int, head_dim: int, theta: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute cos/sin tables of shape (seq_len, head_dim/2)."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("i,j->ij", pos, inv_freq)        # (seq_len, half)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rotary(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """
    Apply RoPE to a tensor of shape (..., seq, n_heads, head_dim).
    Splits the last dim into pairs and rotates each pair by the matching angle.
    """
    # x: (..., seq, n_heads, head_dim)  ; cos/sin: (seq, head_dim/2)
    x1, x2 = jnp.split(x, 2, axis=-1)
    cos_b = cos[..., None, :]  # (seq, 1, head_dim/2) -> broadcast over heads
    sin_b = sin[..., None, :]
    rotated = jnp.concatenate([x1 * cos_b - x2 * sin_b, x1 * sin_b + x2 * cos_b], axis=-1)
    return rotated.astype(x.dtype)


# ----------------------------------------------------------------------------
# Grouped-Query Attention
# ----------------------------------------------------------------------------


class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with grouped K/V heads.

    Shapes:
      input  : (B, T, d_model)
      Q      : (B, T, n_heads,    head_dim)
      K, V   : (B, T, n_kv_heads, head_dim)   -- broadcast to n_heads via tile
      out    : (B, T, d_model)
    """

    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        B, T, _ = x.shape
        kv_dim = self.head_dim * self.n_kv_heads

        q = nn.Dense(
            features=self.d_model,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="q_proj",
        )(x).reshape(B, T, self.n_heads, self.head_dim)

        k = nn.Dense(
            features=kv_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="k_proj",
        )(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        v = nn.Dense(
            features=kv_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="v_proj",
        )(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        # RoPE on Q and K (V is not rotated).
        q = apply_rotary(q, cos[:T], sin[:T])
        k = apply_rotary(k, cos[:T], sin[:T])

        # Broadcast K/V from n_kv_heads to n_heads.
        repeat = self.n_heads // self.n_kv_heads
        if repeat != 1:
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        # Scores: (B, n_heads, T, T)
        scale = 1.0 / jnp.sqrt(jnp.float32(self.head_dim))
        scores = jnp.einsum("bthd,bThd->bhtT", q, k) * scale

        if mask is not None:
            # mask: (1, 1, T, T) of 0/-inf
            scores = scores + mask

        weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        out = jnp.einsum("bhtT,bThd->bthd", weights, v)
        out = out.reshape(B, T, self.d_model)

        return nn.Dense(
            features=self.d_model,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="o_proj",
        )(out)


# ----------------------------------------------------------------------------
# SwiGLU expert
# ----------------------------------------------------------------------------


class SwiGLUExpert(nn.Module):
    """
    SwiGLU FFN: y = down(silu(gate(x)) * up(x)).

    Llama/Mixtral parameterization. No bias on any of the three projections.
    """

    d_model: int
    ff_dim: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.Dense(
            self.ff_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="w_gate",
        )(x)
        up = nn.Dense(
            self.ff_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="w_up",
        )(x)
        h = jax.nn.silu(gate) * up
        return nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="w_down",
        )(h)


# ----------------------------------------------------------------------------
# Top-K router + Sparse MoE block
# ----------------------------------------------------------------------------


def _load_balance_loss(probs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Switch Transformer / Mixtral load-balance auxiliary loss.

      probs : (T, n_experts)  routing probabilities (full softmax)
      mask  : (T, n_experts)  one_hot of selected experts (top_k)

    Returns scalar L = n_experts * sum_e (P_e * f_e), where
      P_e = mean prob mass routed to e   (smooth)
      f_e = fraction of tokens that selected e in top_k  (hard)
    """
    n_experts = probs.shape[-1]
    P = jnp.mean(probs, axis=0)               # (n_experts,)
    f = jnp.mean(mask, axis=0)                # (n_experts,)
    return n_experts * jnp.sum(P * f)


class TopKRouter(nn.Module):
    """
    Linear top-k router with optional training-time jitter.

    Returns:
      gates   (T, top_k)        : softmax-normalized weights of selected experts
      indices (T, top_k)        : expert ids selected per token
      aux     scalar            : load-balance auxiliary loss
    """

    n_experts: int
    top_k: int
    jitter: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_flat: jnp.ndarray, deterministic: bool) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        # x_flat: (T, d_model) where T = B*seq.
        logits = nn.Dense(
            self.n_experts,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="router",
        )(x_flat).astype(jnp.float32)

        if (not deterministic) and self.jitter > 0.0:
            noise = jax.random.uniform(
                self.make_rng("router"),
                logits.shape,
                minval=1.0 - self.jitter,
                maxval=1.0 + self.jitter,
            )
            logits = logits * noise

        probs = jax.nn.softmax(logits, axis=-1)                 # (T, n_experts)
        top_vals, top_idx = jax.lax.top_k(probs, self.top_k)    # (T, k) each

        # Re-normalize selected weights so they sum to 1 per token.
        gates = top_vals / (jnp.sum(top_vals, axis=-1, keepdims=True) + 1e-9)

        mask = jax.nn.one_hot(top_idx, self.n_experts, dtype=jnp.float32).sum(axis=1)
        aux = _load_balance_loss(probs, mask)
        return gates.astype(x_flat.dtype), top_idx, aux


class SparseMoEBlock(nn.Module):
    """
    Top-k MoE block. Implementation: build all experts as a single batched
    SwiGLU (n_experts, d_model, ff_dim) and dispatch per-token by index.

    Memory-friendly path (good enough for 1B/3B per-host training): for each of
    the top_k slots we materialize the per-token expert id and run a single
    expert call by gathering. This avoids Python-level fan-out and keeps the
    block jit-compileable.
    """

    d_model: int
    ff_dim: int
    n_experts: int
    top_k: int
    router_jitter: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        gates, indices, aux = TopKRouter(
            n_experts=self.n_experts,
            top_k=self.top_k,
            jitter=self.router_jitter,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="router",
        )(x_flat, deterministic=deterministic)

        # Stack experts as parameters (n_experts, d, ff)/(n_experts, ff, d).
        w_gate = self.param(
            "w_gate",
            nn.initializers.lecun_normal(),
            (self.n_experts, D, self.ff_dim),
            self.param_dtype,
        )
        w_up = self.param(
            "w_up",
            nn.initializers.lecun_normal(),
            (self.n_experts, D, self.ff_dim),
            self.param_dtype,
        )
        w_down = self.param(
            "w_down",
            nn.initializers.lecun_normal(),
            (self.n_experts, self.ff_dim, D),
            self.param_dtype,
        )

        # For each top-k slot, gather the expert weights per token and run.
        out = jnp.zeros_like(x_flat)
        for k in range(self.top_k):
            idx_k = indices[:, k]                              # (T,)
            gate_k = gates[:, k:k + 1].astype(x_flat.dtype)    # (T, 1)
            wg = w_gate[idx_k].astype(self.dtype)              # (T, D, ff)
            wu = w_up[idx_k].astype(self.dtype)
            wd = w_down[idx_k].astype(self.dtype)
            # token-wise FFN: y_t = (silu(x_t @ wg_t) * (x_t @ wu_t)) @ wd_t
            h_g = jnp.einsum("td,tdh->th", x_flat, wg)
            h_u = jnp.einsum("td,tdh->th", x_flat, wu)
            h = jax.nn.silu(h_g) * h_u
            y = jnp.einsum("th,thd->td", h, wd)
            out = out + gate_k * y

        return out.reshape(B, T, D), aux


# ----------------------------------------------------------------------------
# Transformer block (pre-norm, MoE replaces the FFN)
# ----------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: x + Attn(RMSNorm(x)) + MoE(RMSNorm(x))."""

    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ff_dim: int
    n_experts: int
    top_k: int
    router_jitter: float = 0.0
    rms_norm_eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        mask: jnp.ndarray | None,
        deterministic: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = RMSNorm(self.d_model, eps=self.rms_norm_eps,
                    dtype=self.dtype, param_dtype=self.param_dtype,
                    name="attn_norm")(x)
        h = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="attn",
        )(h, cos=cos, sin=sin, mask=mask)
        x = x + h

        h = RMSNorm(self.d_model, eps=self.rms_norm_eps,
                    dtype=self.dtype, param_dtype=self.param_dtype,
                    name="moe_norm")(x)
        h, aux = SparseMoEBlock(
            d_model=self.d_model,
            ff_dim=self.ff_dim,
            n_experts=self.n_experts,
            top_k=self.top_k,
            router_jitter=self.router_jitter,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="moe",
        )(h, deterministic=deterministic)
        return x + h, aux


def make_causal_mask(seq_len: int, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """Causal mask: (1, 1, T, T) with 0 on/below diagonal, -inf above."""
    i = jnp.arange(seq_len)[:, None]
    j = jnp.arange(seq_len)[None, :]
    mask = jnp.where(j <= i, 0.0, jnp.finfo(dtype).min)
    return mask[None, None, :, :].astype(dtype)
