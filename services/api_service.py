"""Capibara Slim — API service."""
from __future__ import annotations

import logging
import time
from typing import Any

import utils.stats as stats
from services.model_service import ModelService

logger = logging.getLogger(__name__)


class ApiService:
    def __init__(self) -> None:
        self._model = ModelService()

    def generate(
        self,
        input_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        error = False
        try:
            result = self._model.generate(
                input_text=input_text,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            error = True
            raise
        finally:
            latency_ms = (time.monotonic() - t0) * 1000
            stats.record(
                tokens=result.get("tokens_used", 0) if not error else 0,
                latency_ms=latency_ms,
                error=error,
                blocked=result.get("blocked", False) if not error else False,
            )

        return {
            "output": result["output"],
            "model": result.get("model", "stub"),
            "tokens_used": result.get("tokens_used", 0),
        }
