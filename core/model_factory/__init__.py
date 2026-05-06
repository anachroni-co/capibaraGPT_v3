"""
core.model_factory: Mixtral-style MoE Transformer factory.

Public surface (kept small on purpose - the rest is implementation detail):

    from core.model_factory import (
        ModelConfig, get_preset,
        CapibaraMoEModel, build_model, init_params, count_params,
    )

Loading the trunk does NOT touch capibara/* or any heavy dependency: only
jax, flax.linen, stdlib + tomli (py3.10) / tomllib (py3.11+).
"""
from .config import ModelConfig, PRESETS, get_preset
from .model import (
    CapibaraMoEModel,
    ModelOutput,
    build_model,
    count_params,
    init_params,
)

__all__ = [
    "ModelConfig",
    "PRESETS",
    "get_preset",
    "CapibaraMoEModel",
    "ModelOutput",
    "build_model",
    "count_params",
    "init_params",
]
