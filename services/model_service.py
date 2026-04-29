"""Capibara Slim — model service.

Wraps the inference pipeline and exposes a single generate() call.
Currently backed by the stub pipeline; swap backend in config/slim.yaml.
"""
from __future__ import annotations

import logging
from typing import Any

from config.slim_loader import get as cfg_get
from inference.pipeline import SlimPipeline

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self._pipeline = SlimPipeline()
        self._backend: str = cfg_get("model", "backend", "stub")

    def generate(
        self,
        input_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        logger.info("generate | backend=%s | input_len=%d", self._backend, len(input_text))
        return self._pipeline.run(
            input_text,
            max_tokens=max_tokens,
            temperature=temperature,
        )
