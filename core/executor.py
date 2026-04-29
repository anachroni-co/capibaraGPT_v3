"""Capibara Slim — core executor (T2.1).

Central execution unit: receives input, asks the router which backend
to use, fetches the backend from the registry, and runs generation.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from config.slim_loader import get as cfg_get
from core.slim_router import SlimRouter
from models.registry import get_registry

logger = logging.getLogger(__name__)


class SlimExecutor:
    def __init__(self) -> None:
        self._router = SlimRouter()
        self._registry = get_registry()
        self._model_backend: str = cfg_get("model", "backend", "auto")

    def run(
        self,
        input_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        t0 = time.monotonic()

        backend_name = self._select_backend(input_text)
        backend = self._registry.get(backend_name)

        if not backend.is_available:
            logger.warning("backend '%s' unavailable, falling back to stub", backend_name)
            backend_name = "stub"
            backend = self._registry.get("stub")

        result = backend.generate(input_text, max_tokens=max_tokens, temperature=temperature)

        result["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        result["routed_to"] = backend_name
        logger.info(
            "executor: routed_to=%s tokens=%s latency=%.1fms",
            backend_name,
            result.get("tokens_used", "?"),
            result["latency_ms"],
        )
        return result

    def _select_backend(self, input_text: str) -> str:
        """Config 'model.backend' overrides routing; 'auto' defers to router."""
        cfg = self._model_backend
        if cfg == "auto":
            routed = self._router.route(input_text)
            # tool routing is handled upstream; fall back to stub for now
            return routed if routed != "tool" else "stub"
        return cfg
