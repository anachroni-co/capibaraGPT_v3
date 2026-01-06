"""
Compatibilidad: reexporta tipos del esquema unificado.
"""
from .config_schema import (
    ModelConfig,
    TrainingConfig,
    PruningConfig,
    WandbConfig,
    ModulesConfig,
    PathsConfig,
    CapibaraConfig,
)

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "PruningConfig",
    "WandbConfig",
    "ModulesConfig",
    "PathsConfig",
    "CapibaraConfig",
]


