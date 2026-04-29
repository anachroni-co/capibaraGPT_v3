"""Capibara Slim — API service.

Sits between the HTTP routes and the model service. Responsible for
request validation, response shaping, and (later) rate limiting.
"""
from __future__ import annotations

import logging
from typing import Any

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
        result = self._model.generate(
            input_text=input_text,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "output": result["output"],
            "model": result.get("model", "stub"),
            "tokens_used": result.get("tokens_used", 0),
        }
