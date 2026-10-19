"""
Interface for submodules in CapibaraGPT.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional, Protocol, Union


class ISubModel(Protocol):
    """
    Standard interface for CapibaraGPT submodules.

    Defines the contract that all submodules must fulfill
    to be compatible with the modular system.
    """

    @abstractmethod
    def __call__(
        self,
        inputs: Any,
        *,
        training: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Forward pass of the submodule.

        Args:
            inputs: Input to the submodule (can be tensor, dict, etc.)
            training: Whether in training mode
            **kwargs: Additional arguments specific to the submodule

        Returns:
            Dict with the submodule outputs
        """
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the submodule configuration.

        Returns:
            Dictionary with configuration
        """
        ...

    @abstractmethod
    def setup_optimizations(self, device: str = "cpu") -> None:
        """
        Configure device-specific optimizations.

        Args:
            device: Device type ("cpu", "gpu", "tpu")
        """
        ...

    def get_metrics(self) -> Dict[str, float]:
        """
        Get submodule metrics.

        Returns:
            Dictionary with metrics (optional)
        """
        return {}

    def reset_state(self) -> None:
        """
        Reset the internal state of the submodule (optional).
        """
        pass
