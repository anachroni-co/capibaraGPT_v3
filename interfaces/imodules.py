"""
IModule interface for modular components.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict


class IModule(ABC):
    """Minimal interface for modular components."""

    def __call__(self, inputs: Any, training: bool = False) -> Dict[str, Any]:
        """Run a forward pass and return a result dict."""
        raise NotImplementedError

    def get_metrics(self) -> Dict[str, Any]:
        """Return module metrics."""
        return {}

    def get_config(self) -> Dict[str, Any]:
        """Return module configuration."""
        return {}
