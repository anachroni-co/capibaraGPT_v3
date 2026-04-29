"""Capibara Slim — Mamba backend.

Week 2: stub that accepts the same interface as TransformerBackend.
Swap _generate_real() in when Mamba weights + mamba-ssm are available.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MAMBA_AVAILABLE = False
try:
    import mamba_ssm  # type: ignore[import]
    _MAMBA_AVAILABLE = True
except ImportError:
    pass


class MambaBackend:
    """Mamba SSM backend for efficient short-sequence inference."""

    def __init__(self, model_path: str = "") -> None:
        self._model_path = model_path
        self._model = None
        if _MAMBA_AVAILABLE and model_path:
            self._load()

    def _load(self) -> None:
        logger.info("MambaBackend: model loading not yet implemented; using stub")

    @property
    def name(self) -> str:
        return "mamba"

    @property
    def is_available(self) -> bool:
        # Stub is always available; returns False only if a real model was
        # requested but failed to load.
        return True

    def generate(
        self,
        input_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        if _MAMBA_AVAILABLE and self._model is not None:
            return self._generate_real(input_text, max_tokens, temperature)
        return self._generate_stub(input_text)

    def _generate_stub(self, input_text: str) -> dict[str, Any]:
        tokens = input_text.split()
        return {
            "output": f"[mamba-stub] {len(tokens)} token(s): {input_text[:60]}",
            "model": self.name,
            "tokens_used": len(tokens),
        }

    def _generate_real(
        self,
        input_text: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("Real Mamba generation not yet wired")
