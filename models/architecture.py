"""Capibara Slim — model architecture (T5.1).

Config-driven hybrid Transformer + Mamba model.

Presets:
  SlimConfig.preset("1.5b")  hidden=2048, layers=24  → ~1.3B params
  SlimConfig.preset("3b")    hidden=2560, layers=32  → ~3.1B params
  SlimConfig.preset("7b")    hidden=4096, layers=32  → ~6.7B params

Block layout (configurable via mamba_every_n):
  Default: Transformer everywhere.
  Set mamba_every_n=2 to alternate Transformer/Mamba.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Config (no torch required)
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict] = {
    "1.5b": dict(
        hidden_size=2048, num_layers=24, num_heads=16,
        intermediate_size=5504, vocab_size=32000, max_seq_len=2048,
    ),
    "3b": dict(
        hidden_size=2560, num_layers=32, num_heads=20,
        intermediate_size=6912, vocab_size=32000, max_seq_len=2048,
    ),
    "7b": dict(
        hidden_size=4096, num_layers=32, num_heads=32,
        intermediate_size=11008, vocab_size=32000, max_seq_len=4096,
    ),
}


@dataclass
class SlimConfig:
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 5504  # SwiGLU inner dim
    vocab_size: int = 32000
    max_seq_len: int = 2048
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    mamba_every_n: int = 0        # 0 = no Mamba layers; 2 = every 2nd layer
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    tie_embeddings: bool = True   # tie input/output embedding weights

    @classmethod
    def preset(cls, name: str) -> "SlimConfig":
        key = name.lower().replace("-", "").replace("_", "")
        if key not in _PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(_PRESETS)}")
        return cls(**_PRESETS[key])

    def estimate_params(self) -> int:
        """Rough parameter count (M)."""
        embed = self.vocab_size * self.hidden_size
        h = self.hidden_size
        attn = 4 * h * h          # Q K V O
        mlp = 3 * h * self.intermediate_size  # SwiGLU: gate + up + down
        layer = attn + mlp
        return embed + self.num_layers * layer


# ---------------------------------------------------------------------------
# PyTorch components
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


def _require_torch(name: str) -> None:
    if not _TORCH:
        raise ImportError(f"{name} requires PyTorch. Install with: pip install torch")


# ---- RMSNorm ---------------------------------------------------------------

if _TORCH:
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
            return self.weight * x / rms
else:
    class RMSNorm:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("RMSNorm")


# ---- Rotary Embeddings -----------------------------------------------------

if _TORCH:
    class RotaryEmbedding(nn.Module):
        def __init__(self, dim: int, max_seq_len: int = 4096) -> None:
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_seq_len).float()
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("cos", emb.cos()[None, None, :, :])
            self.register_buffer("sin", emb.sin()[None, None, :, :])

        def forward(self, seq_len: int):
            return self.cos[:, :, :seq_len, :], self.sin[:, :, :seq_len, :]

    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(q, k, cos, sin):
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

else:
    class RotaryEmbedding:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("RotaryEmbedding")


# ---- Attention -------------------------------------------------------------

if _TORCH:
    class SlimAttention(nn.Module):
        def __init__(self, cfg: SlimConfig) -> None:
            super().__init__()
            assert cfg.hidden_size % cfg.num_heads == 0
            self.num_heads = cfg.num_heads
            self.head_dim = cfg.hidden_size // cfg.num_heads
            self.scale = self.head_dim ** -0.5

            self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
            self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
            self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
            self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

            self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len)
            self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(self, x, mask=None):
            residual = x
            x = self.norm(x)
            B, L, _ = x.shape
            H, D = self.num_heads, self.head_dim

            def _split(t):
                return t.view(B, L, H, D).transpose(1, 2)  # (B, H, L, D)

            q, k, v = _split(self.q_proj(x)), _split(self.k_proj(x)), _split(self.v_proj(x))
            cos, sin = self.rope(L)
            q, k = _apply_rope(q, k, cos, sin)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask, float("-inf"))
            attn = self.dropout(F.softmax(attn, dim=-1))

            out = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)
            return self.o_proj(out) + residual

else:
    class SlimAttention:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("SlimAttention")


# ---- SwiGLU MLP ------------------------------------------------------------

if _TORCH:
    class SlimMLP(nn.Module):
        def __init__(self, cfg: SlimConfig) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
            self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        def forward(self, x):
            residual = x
            x = self.norm(x)
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) + residual

else:
    class SlimMLP:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("SlimMLP")


# ---- TransformerBlock ------------------------------------------------------

if _TORCH:
    class TransformerBlock(nn.Module):
        def __init__(self, cfg: SlimConfig) -> None:
            super().__init__()
            self.attn = SlimAttention(cfg)
            self.mlp  = SlimMLP(cfg)

        def forward(self, x, mask=None):
            return self.mlp(self.attn(x, mask))

else:
    class TransformerBlock:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("TransformerBlock")


# ---- Mamba Block (simplified selective SSM) --------------------------------

if _TORCH:
    class MambaBlock(nn.Module):
        """Simplified Mamba SSM block (pure PyTorch — no CUDA kernel required).

        Implements the core selective SSM: input-dependent dt/B/C,
        zero-order-hold discretisation, sequential scan, and SiLU gate.
        Slower than the official CUDA kernel but functionally equivalent
        and trainable on any hardware.
        """

        def __init__(self, cfg: SlimConfig) -> None:
            super().__init__()
            d = cfg.hidden_size
            self.d_state = cfg.mamba_d_state
            d_inner = cfg.mamba_expand * d

            self.norm = RMSNorm(d, cfg.rms_norm_eps)
            self.in_proj = nn.Linear(d, d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                d_inner, d_inner, cfg.mamba_d_conv,
                padding=cfg.mamba_d_conv - 1, groups=d_inner, bias=True,
            )
            self.x_proj = nn.Linear(d_inner, self.d_state * 2 + 1, bias=False)
            self.dt_proj = nn.Linear(1, d_inner)
            self.out_proj = nn.Linear(d_inner, d, bias=False)

            A = torch.arange(1, self.d_state + 1, dtype=torch.float).repeat(d_inner, 1)
            self.A_log = nn.Parameter(A.log())
            self.D = nn.Parameter(torch.ones(d_inner))

        def forward(self, x, mask=None):
            B_batch, L, _ = x.shape
            residual = x
            x = self.norm(x)

            xz = self.in_proj(x)
            x_part, z = xz.chunk(2, dim=-1)

            # Local context via 1-D depthwise conv
            x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, :L].transpose(1, 2)
            x_conv = F.silu(x_conv)

            # Input-dependent SSM parameters
            x_dbl = self.x_proj(x_conv)
            dt_raw, B_mat, C_mat = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(self.dt_proj(dt_raw))            # (B, L, d_inner)

            A = -torch.exp(self.A_log.float())               # (d_inner, d_state)

            # Discretise: zero-order hold
            # dA: (B, L, d_inner, d_state)
            dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))
            # dB: (B, L, d_inner, d_state)
            dB = torch.einsum("bld,bln->bldn", dt, B_mat)

            # Sequential scan (O(L) memory, no parallel scan for simplicity)
            h = torch.zeros(B_batch, x_part.shape[-1], self.d_state,
                            device=x.device, dtype=x.dtype)
            ys = []
            for t in range(L):
                h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
                y = torch.einsum("bdn,bn->bd", h, C_mat[:, t])
                ys.append(y)

            y = torch.stack(ys, dim=1)                       # (B, L, d_inner)
            y = y + x_conv * self.D
            y = y * F.silu(z)
            return self.out_proj(y) + residual

else:
    class MambaBlock:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("MambaBlock")


# ---- SlimModel -------------------------------------------------------------

if _TORCH:
    class SlimModel(nn.Module):
        """Capibara Slim — hybrid Transformer + Mamba language model."""

        def __init__(self, cfg: SlimConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

            blocks: list[nn.Module] = []
            for i in range(cfg.num_layers):
                use_mamba = cfg.mamba_every_n > 0 and (i % cfg.mamba_every_n == 1)
                blocks.append(MambaBlock(cfg) if use_mamba else TransformerBlock(cfg))
            self.blocks = nn.ModuleList(blocks)

            self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

            if cfg.tie_embeddings:
                self.lm_head.weight = self.embed.weight

            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, input_ids, attention_mask=None):
            B, L = input_ids.shape
            mask = None
            if attention_mask is None:
                # Causal mask
                mask = torch.triu(
                    torch.ones(L, L, device=input_ids.device, dtype=torch.bool), diagonal=1
                ).unsqueeze(0).unsqueeze(0)

            x = self.embed(input_ids)
            for block in self.blocks:
                x = block(x, mask)
            x = self.norm(x)
            return self.lm_head(x)

        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

else:
    class SlimModel:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("SlimModel")


__all__ = [
    "SlimConfig",
    "SlimModel",
    "TransformerBlock",
    "MambaBlock",
    "SlimAttention",
    "SlimMLP",
    "RMSNorm",
]
