"""Capibara Slim — simple deterministic router (T2.2).

Routes each request to a backend based on configurable strategy.

  length (default):
    - input starts with "tool:"   → "tool"
    - word count < mamba_threshold → "mamba"
    - otherwise                    → "transformer"

  always_transformer / always_mamba / always_stub:
    Force a single backend for testing.
"""
from __future__ import annotations

import logging

from config.slim_loader import get as cfg_get

logger = logging.getLogger(__name__)


class SlimRouter:
    def __init__(self) -> None:
        self._strategy: str = cfg_get("routing", "strategy", "length")
        self._mamba_threshold: int = int(cfg_get("routing", "mamba_threshold", 128))

    def route(self, input_text: str) -> str:
        """Return the backend name for the given input."""
        backend = self._route(input_text)
        logger.debug("router: '%s' → %s", input_text[:40], backend)
        return backend

    def _route(self, text: str) -> str:
        strategy = self._strategy

        if strategy == "always_transformer":
            return "transformer"
        if strategy == "always_mamba":
            return "mamba"
        if strategy == "always_stub":
            return "stub"

        # Default: length-based
        if text.lstrip().lower().startswith("tool:"):
            return "tool"
        word_count = len(text.split())
        return "mamba" if word_count < self._mamba_threshold else "transformer"
