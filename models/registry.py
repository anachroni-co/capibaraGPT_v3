"""Capibara Slim — model registry.

Registers available backends by name and returns cached instances.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from models.backend import ModelBackend

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], ModelBackend]] = {}
        self._cache: dict[str, ModelBackend] = {}

    def register(self, name: str, factory: Callable[[], ModelBackend]) -> None:
        self._factories[name] = factory
        logger.debug("ModelRegistry: registered backend '%s'", name)

    def get(self, name: str) -> ModelBackend:
        if name not in self._cache:
            if name not in self._factories:
                raise KeyError(f"No backend registered under '{name}'")
            self._cache[name] = self._factories[name]()
        return self._cache[name]

    def available(self) -> list[str]:
        return list(self._factories.keys())


# ---------------------------------------------------------------------------
# Default registry — populated once at import time
# ---------------------------------------------------------------------------

def _build_default_registry() -> ModelRegistry:
    from config.slim_loader import get as cfg_get
    from models.stub_backend import StubBackend
    from models.transformer_backend import TransformerBackend
    from models.mamba_backend import MambaBackend

    model_path: str = cfg_get("model", "path", "models/tiny-gpt2")

    reg = ModelRegistry()
    reg.register("stub", lambda: StubBackend())
    reg.register("transformer", lambda: TransformerBackend(model_path))
    reg.register("mamba", lambda: MambaBackend())
    return reg


_default: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _default
    if _default is None:
        _default = _build_default_registry()
    return _default
