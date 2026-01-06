"""
Compatibility: exports model classes from unified_model_config.

This wrapper ensures that imports like `from capibara.config.model_config import ModelConfig`
work without changing existing code.
"""

from .unified_model_config import (
    ModelConfig,
    CapibaraConfig,
    BitNetConfig,
    NeuroAdaptiveConfig,
    ModularModelConfig,
    TrainingConfig,
)

__all__ = [
    "ModelConfig",
    "CapibaraConfig",
    "BitNetConfig",
    "NeuroAdaptiveConfig",
    "ModularModelConfig",
    "TrainingConfig",
]


