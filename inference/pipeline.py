"""Capibara Slim — inference pipeline.

Week 1 implementation: stub backend that echoes input with a canned reply.
Week 2 will replace _model_call() with real Transformer / Mamba execution.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from config.slim_loader import get as cfg_get

logger = logging.getLogger(__name__)


class SlimPipeline:
    def __init__(self) -> None:
        self._backend: str = cfg_get("model", "backend", "stub")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        input_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        tokens = self._preprocess(input_text)
        raw = self._model_call(tokens, max_tokens=max_tokens, temperature=temperature)
        output = self._postprocess(raw)
        elapsed = time.monotonic() - t0
        logger.debug("pipeline.run | tokens=%d | elapsed=%.3fs", len(tokens), elapsed)
        return {
            "output": output,
            "model": self._backend,
            "tokens_used": len(tokens),
            "latency_ms": round(elapsed * 1000, 1),
        }

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _preprocess(self, text: str) -> list[str]:
        """Tokenise by whitespace (placeholder until real tokeniser is wired)."""
        return text.split()

    def _model_call(
        self,
        tokens: list[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        if self._backend == "stub":
            return f"[stub] received {len(tokens)} token(s)"
        raise NotImplementedError(f"backend '{self._backend}' not yet implemented")

    def _postprocess(self, raw: str) -> str:
        return raw.strip()
