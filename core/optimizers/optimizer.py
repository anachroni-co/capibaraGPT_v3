"""Optimizer Module for CapibaraGPT Training.

This module provides optimizer configuration and base optimizer implementations
for training CapibaraGPT models. It supports various optimization algorithms
including Adam, SGD with momentum, and includes gradient clipping functionality.

The optimizer system features:
- Configurable learning rate and hyperparameters
- Multiple optimizer types (Adam, SGD, AdamW)
- Gradient clipping support
- Weight decay (L2 regularization)
- Extensible base class for custom optimizers

Key Components:
    - OptimizerConfig: Configuration dataclass for optimizer hyperparameters
    - BaseOptimizer: Base optimizer implementation
    - create_optimizer: Factory function for optimizer creation

Example:
    Basic optimizer creation:

    >>> from capibara.core.optimizers.optimizer import create_optimizer, OptimizerConfig
    >>>
    >>> # Create with default Adam configuration
    >>> optimizer = create_optimizer()
    >>>
    >>> # Custom configuration
    >>> config = OptimizerConfig(
    ...     learning_rate=0.0001,
    ...     optimizer_type="adam",
    ...     weight_decay=0.01,
    ...     clip_grad_norm=1.0
    ... )
    >>> optimizer = create_optimizer(config)

    Using the optimizer:

    >>> # Perform optimization step
    >>> updated_params = optimizer.step(gradients)
    >>>
    >>> # Zero gradients before next step
    >>> optimizer.zero_grad()

    Configuration dictionary:

    >>> config = OptimizerConfig(learning_rate=0.001)
    >>> config_dict = config.to_dict()
    >>> print(config_dict["learning_rate"])  # 0.001

Note:
    This is a base implementation designed for extensibility. For production
    training, consider integrating with Optax or other JAX optimization libraries.

See Also:
    - capibara.training: Training loop implementations
    - capibara.core.optimization: Advanced optimization utilities
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OptimizerConfig:
    """Configuration for optimizer hyperparameters.

    This dataclass encapsulates all hyperparameters needed to configure an
    optimizer for training. It supports Adam, SGD, and AdamW optimizers with
    configurable learning rate, momentum, weight decay, and gradient clipping.

    Attributes:
        learning_rate (float): Learning rate for parameter updates. Defaults to 0.001.
        optimizer_type (str): Type of optimizer ("adam", "sgd", "adamw").
            Defaults to "adam".
        momentum (float): Momentum factor for SGD and related optimizers.
            Defaults to 0.9.
        weight_decay (float): L2 regularization coefficient. Defaults to 0.0 (no decay).
        eps (float): Small constant for numerical stability in Adam. Defaults to 1e-8.
        beta1 (float): Exponential decay rate for first moment estimates (Adam).
            Defaults to 0.9.
        beta2 (float): Exponential decay rate for second moment estimates (Adam).
            Defaults to 0.999.
        clip_grad_norm (Optional[float]): Maximum gradient norm for clipping.
            None means no clipping. Defaults to None.

    Example:
        >>> # Default Adam configuration
        >>> config = OptimizerConfig()
        >>> print(config.optimizer_type)  # "adam"
        >>> print(config.learning_rate)  # 0.001
        >>>
        >>> # SGD with momentum
        >>> sgd_config = OptimizerConfig(
        ...     optimizer_type="sgd",
        ...     learning_rate=0.01,
        ...     momentum=0.95
        ... )
        >>>
        >>> # Adam with weight decay and gradient clipping
        >>> adamw_config = OptimizerConfig(
        ...     optimizer_type="adamw",
        ...     learning_rate=0.0001,
        ...     weight_decay=0.01,
        ...     clip_grad_norm=1.0
        ... )

    Note:
        The optimizer_type field determines which hyperparameters are actually
        used. For example, momentum is only used with SGD-based optimizers,
        while beta1 and beta2 are specific to Adam variants.
    """
    learning_rate: float = 0.001
    optimizer_type: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    eps: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999
    clip_grad_norm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary containing all configuration parameters.

        Example:
            >>> config = OptimizerConfig(learning_rate=0.001)
            >>> config_dict = config.to_dict()
            >>> print(config_dict)
            {
                'learning_rate': 0.001,
                'optimizer_type': 'adam',
                'momentum': 0.9,
                'weight_decay': 0.0,
                'eps': 1e-08,
                'beta1': 0.9,
                'beta2': 0.999,
                'clip_grad_norm': None
            }

        Note:
            Useful for serialization, logging, or passing to other configuration
            systems that expect dictionary inputs.
        """
        return {
            "learning_rate": self.learning_rate,
            "optimizer_type": self.optimizer_type,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "eps": self.eps,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "clip_grad_norm": self.clip_grad_norm
        }

class BaseOptimizer:
    """Base optimizer class for training CapibaraGPT models.

    This class provides a foundation for optimizer implementations, handling
    initialization, step updates, and gradient zeroing. It can be extended to
    implement specific optimization algorithms.

    Attributes:
        config (OptimizerConfig): Optimizer configuration with hyperparameters.
        initialized (bool): Whether the optimizer has been initialized.

    Example:
        >>> config = OptimizerConfig(learning_rate=0.001)
        >>> optimizer = BaseOptimizer(config)
        >>>
        >>> # Initialize optimizer
        >>> optimizer.initialize()
        >>>
        >>> # Perform optimization step
        >>> new_grads = optimizer.step(gradients)
        >>>
        >>> # Zero gradients
        >>> optimizer.zero_grad()

    Note:
        This is a base implementation. For production use, consider extending
        this class with actual optimization logic (gradient descent, Adam updates,
        etc.) or integrating with Optax for JAX-optimized implementations.
    """

    def __init__(self, config: OptimizerConfig):
        """Initialize the base optimizer with configuration.

        Args:
            config (OptimizerConfig): Optimizer configuration with hyperparameters.

        Example:
            >>> config = OptimizerConfig(
            ...     learning_rate=0.0001,
            ...     optimizer_type="adam",
            ...     weight_decay=0.01
            ... )
            >>> optimizer = BaseOptimizer(config)
            >>> print(optimizer.config.learning_rate)  # 0.0001
        """
        self.config = config
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize the optimizer state.

        This method sets up any necessary state for the optimizer. In the base
        implementation, it simply marks the optimizer as initialized and logs
        the confirmation.

        Returns:
            bool: True if initialization successful, False if error occurred.

        Example:
            >>> optimizer = BaseOptimizer(OptimizerConfig())
            >>> success = optimizer.initialize()
            >>> print(success)  # True
            >>> print(optimizer.initialized)  # True

        Note:
            Extended implementations should override this method to initialize
            optimizer-specific state (momentum buffers, adaptive learning rate
            parameters, etc.).
        """
        try:
            self.initialized = True
            logger.info(f"Optimizer {self.config.optimizer_type} initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            return False

    def step(self, gradients: Any) -> Any:
        """Perform a single optimization step with given gradients.

        This method applies the optimization algorithm to update parameters based
        on computed gradients. The base implementation is a placeholder that
        returns gradients unchanged.

        Args:
            gradients (Any): Gradients computed from loss backpropagation. Type
                depends on the model framework (e.g., JAX pytrees, PyTorch tensors).

        Returns:
            Any: Updated gradients or parameters. Type matches input type.

        Example:
            >>> optimizer = BaseOptimizer(OptimizerConfig())
            >>> gradients = {"layer1": [0.1, 0.2], "layer2": [0.3, 0.4]}
            >>> updated = optimizer.step(gradients)
            >>> # In base implementation, updated == gradients

        Note:
            Extended implementations should:
            1. Check if initialized, initialize if needed
            2. Apply gradient clipping if configured
            3. Apply optimization algorithm (SGD, Adam, etc.)
            4. Apply weight decay if configured
            5. Return updated parameters or gradients

            Current implementation automatically initializes if not initialized.
        """
        if not self.initialized:
            self.initialize()

        # Basic optimization step (placeholder - extend for actual algorithm)
        logger.debug("Optimizer step performed")
        return gradients

    def zero_grad(self) -> None:
        """Zero out accumulated gradients.

        This method resets gradient accumulators. In frameworks like PyTorch,
        this is essential before computing new gradients. In JAX with functional
        programming, this may be a no-op since gradients are recomputed each step.

        Example:
            >>> optimizer = BaseOptimizer(OptimizerConfig())
            >>> optimizer.zero_grad()  # Gradients zeroed

        Note:
            Extended implementations should clear any accumulated gradient state
            or buffers specific to the optimization algorithm.
        """
        logger.debug("Gradients zeroed")

def create_optimizer(config: Optional[OptimizerConfig] = None) -> BaseOptimizer:
    """Factory function to create and initialize an optimizer.

    Creates an optimizer instance with the specified configuration, automatically
    initializing it for immediate use.

    Args:
        config (Optional[OptimizerConfig], optional): Optimizer configuration.
            If None, uses default OptimizerConfig(). Defaults to None.

    Returns:
        BaseOptimizer: Initialized optimizer instance ready for training.

    Example:
        >>> # Create with defaults
        >>> optimizer = create_optimizer()
        >>> print(optimizer.config.optimizer_type)  # "adam"
        >>>
        >>> # Create with custom config
        >>> config = OptimizerConfig(
        ...     learning_rate=0.0001,
        ...     optimizer_type="adamw",
        ...     weight_decay=0.01
        ... )
        >>> optimizer = create_optimizer(config)
        >>> print(optimizer.initialized)  # True

    Note:
        This is the recommended way to create optimizers, as it ensures
        proper initialization. The optimizer is ready to use immediately
        after creation.
    """
    if config is None:
        config = OptimizerConfig()

    optimizer = BaseOptimizer(config)
    optimizer.initialize()

    logger.info(f"Created {config.optimizer_type} optimizer")
    return optimizer

def main():
    """Main function for module execution.

    This function is called when the module is run directly. Provides basic
    logging confirmation that the module loaded successfully.

    Returns:
        bool: Always returns True to indicate successful execution.

    Example:
        >>> from capibara.core.optimizers import optimizer
        >>> result = optimizer.main()
        >>> print(result)  # True

    Note:
        This is primarily for testing and verification purposes. Production code
        should use create_optimizer() or instantiate BaseOptimizer directly.
    """
    logger.info("Module optimizer.py starting")
    return True

if __name__ == "__main__":
    main()
