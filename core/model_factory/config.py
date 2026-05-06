"""
ModelConfig dataclass + size presets for the CapibaraGPT MoE factory.

Independent from core/modular_model.py: imports only stdlib + tomllib so it
can be loaded in test environments without the full capibara graph.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:  # py3.11+
    import tomllib as _toml_loader
    _TOML_BINARY = True
except ModuleNotFoundError:  # py3.10
    import tomli as _toml_loader  # type: ignore
    _TOML_BINARY = True


@dataclass(frozen=True)
class ModelConfig:
    """
    Static architecture description for a Mixtral-style decoder-only MoE
    Transformer. All fields are required at construction time; presets and
    TOML loaders only fill in the values, they do not patch defaults later.

    Param-count helpers (`active_params_estimate`, `total_params_estimate`)
    return analytic estimates so configs can be sanity-checked without
    materializing weights.
    """

    # ---- vocabulary ----
    vocab_size: int
    max_seq_len: int

    # ---- transformer trunk ----
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int        # for grouped-query attention; n_heads % n_kv_heads must be 0
    head_dim: int          # explicit so we do not silently divide d_model//n_heads
    ff_dim: int            # SwiGLU intermediate size

    # ---- MoE ----
    n_experts: int
    top_k: int             # number of experts dispatched per token
    router_jitter: float = 0.0    # additive noise on routing logits during training
    load_balance_weight: float = 0.01

    # ---- attention/positional ----
    rope_theta: float = 500_000.0
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0

    # ---- model-wide ----
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    dtype: str = "bfloat16"
    param_dtype: str = "float32"

    # ---- metadata ----
    name: str = "capibara-moe"

    # ---------------------------------------------------------------- validation
    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads "
                f"({self.n_kv_heads}) for grouped-query attention."
            )
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError(
                f"head_dim*n_heads ({self.head_dim}*{self.n_heads}="
                f"{self.head_dim * self.n_heads}) must equal d_model ({self.d_model})."
            )
        if self.top_k > self.n_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed n_experts ({self.n_experts})."
            )
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.param_dtype not in ("bfloat16", "float32"):
            raise ValueError(f"unsupported param_dtype: {self.param_dtype}")

    # --------------------------------------------------------------- estimates
    def _attention_params(self) -> int:
        """Params of one GQA attention block (Q, K, V, O projections)."""
        d = self.d_model
        kv_dim = self.head_dim * self.n_kv_heads
        # Q: d * d, K: d * kv_dim, V: d * kv_dim, O: d * d
        return d * d + 2 * d * kv_dim + d * d

    def _expert_params(self) -> int:
        """Params of one SwiGLU expert (gate, up, down projections)."""
        return 3 * self.d_model * self.ff_dim

    def _router_params(self) -> int:
        return self.d_model * self.n_experts

    def _embedding_params(self) -> int:
        return self.vocab_size * self.d_model

    def active_params_estimate(self) -> int:
        """
        Active params per token: embedding + n_layers * (attn + top_k experts +
        router) + lm_head (untied) or 0 (tied). Excludes layer-norm scalars.
        """
        per_layer_active = (
            self._attention_params()
            + self.top_k * self._expert_params()
            + self._router_params()
        )
        head = 0 if self.tie_word_embeddings else self._embedding_params()
        return self._embedding_params() + self.n_layers * per_layer_active + head

    def total_params_estimate(self) -> int:
        """Total params: same as active but with all experts, not just top_k."""
        per_layer_total = (
            self._attention_params()
            + self.n_experts * self._expert_params()
            + self._router_params()
        )
        head = 0 if self.tie_word_embeddings else self._embedding_params()
        return self._embedding_params() + self.n_layers * per_layer_total + head

    # --------------------------------------------------------------- builders
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        # Strip unknown keys so user-extended TOMLs do not crash the loader.
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def from_toml(cls, path: os.PathLike | str) -> "ModelConfig":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"config TOML not found: {path}")
        with open(path, "rb") as fh:
            data = _toml_loader.load(fh)
        # Accept either flat keys or a [model] section.
        section = data.get("model", data)
        return cls.from_dict(section)

    # --------------------------------------------------------------- presets
    @classmethod
    def preset_1b(cls) -> "ModelConfig":
        """
        ~1.1B active / ~2.0B total. 4 experts top-2 Mixtral-style. d_model=1536,
        24 layers, GQA 3:1. Targeted at single-host TPU v4 / 8x A100 training.
        """
        return cls(
            vocab_size=32000,
            max_seq_len=4096,
            d_model=1536,
            n_layers=24,
            n_heads=24,
            n_kv_heads=8,
            head_dim=64,
            ff_dim=4096,
            n_experts=4,
            top_k=2,
            name="capibara-moe-1b",
        )

    @classmethod
    def preset_3b(cls) -> "ModelConfig":
        """
        ~3.0B active / ~5.4B total. 4 experts top-2. d_model=2048, 36 layers,
        GQA 4:1. Targeted at TPU v4-32 pod / 16x H100 training.
        """
        return cls(
            vocab_size=32000,
            max_seq_len=4096,
            d_model=2048,
            n_layers=36,
            n_heads=32,
            n_kv_heads=8,
            head_dim=64,
            ff_dim=5632,
            n_experts=4,
            top_k=2,
            name="capibara-moe-3b",
        )

    @classmethod
    def preset_smoke(cls) -> "ModelConfig":
        """Minimal config for unit tests / CPU smoke. ~5M params."""
        return cls(
            vocab_size=512,
            max_seq_len=64,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            ff_dim=128,
            n_experts=2,
            top_k=2,
            name="capibara-moe-smoke",
        )


PRESETS: Dict[str, "ModelConfig"] = {
    "1b": ModelConfig.preset_1b(),
    "3b": ModelConfig.preset_3b(),
    "smoke": ModelConfig.preset_smoke(),
}


def get_preset(name: str) -> ModelConfig:
    key = name.lower()
    if key not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[key]
