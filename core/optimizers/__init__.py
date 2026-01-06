"""Optimizers Package for CapibaraGPT Training.

This package provides optimizer implementations and configuration for training
CapibaraGPT models. It includes base optimizer classes, configuration dataclasses,
and factory functions for creating optimizers with various algorithms (Adam, SGD, AdamW).

The package is designed to work with JAX-based training loops and integrates with
Optax for production-grade optimization.

Key Components:
    - OptimizerConfig: Configuration dataclass for optimizer hyperparameters
    - create_optimizer: Factory function to create configured optimizers

Example:
    Basic usage:

    >>> from capibara.core.optimizers import create_optimizer, OptimizerConfig
    >>>
    >>> # Create optimizer with defaults (Adam, lr=0.001)
    >>> optimizer = create_optimizer()
    >>>
    >>> # Create with custom configuration
    >>> config = OptimizerConfig(
    ...     learning_rate=0.0001,
    ...     optimizer_type="adamw",
    ...     weight_decay=0.01,
    ...     clip_grad_norm=1.0
    ... )
    >>> optimizer = create_optimizer(config)
    >>>
    >>> # Use in training loop
    >>> updated_grads = optimizer.step(gradients)
    >>> optimizer.zero_grad()

Note:
    If dependencies are missing (e.g., Optax not installed), the imports
    gracefully fall back to None, allowing the package to be imported
    without errors. Check for None before using.

See Also:
    - capibara.core.optimization: Advanced optimization utilities
    - capibara.training: Training loop implementations
    - Optax documentation: https://optax.readthedocs.io/
"""

try:
    from .optimizer import OptimizerConfig, create_optimizer
except ImportError as e:
    # Fallback if dependencies are missing
    OptimizerConfig = None  # type: ignore
    create_optimizer = None  # type: ignore

__all__ = ['OptimizerConfig', 'create_optimizer']
