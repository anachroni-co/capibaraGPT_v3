"""
core.trainer: canonical pretraining loop for `core.model_factory.CapibaraMoEModel`.

Public surface (kept small on purpose):

    from core.trainer import (
        Recipe, TrainingConfig, CheckpointConfig, LoggingConfig, load_recipe,
        build_optimizer,
        CheckpointManager,
        Trainer, TrainState, StepMetrics,
    )

This module deliberately does NOT depend on capibara/* or training/*; it
only uses jax + flax + optax + orbax. Keeping the dependency surface
small is what lets it stand up an end-to-end training run on CPU for the
smoke preset, then scale to TPU/GPU without rewiring imports.
"""
from .config_loader import (
    Recipe,
    TrainingConfig,
    CheckpointConfig,
    LoggingConfig,
    load_recipe,
)
from .optimizer import build_optimizer
from .checkpointing import CheckpointManager
from .trainer import Trainer, TrainState, StepMetrics, make_train_step, make_eval_step

__all__ = [
    "Recipe",
    "TrainingConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "load_recipe",
    "build_optimizer",
    "CheckpointManager",
    "Trainer",
    "TrainState",
    "StepMetrics",
    "make_train_step",
    "make_eval_step",
]
