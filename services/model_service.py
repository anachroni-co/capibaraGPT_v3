"""Capibara Slim — model service."""
from __future__ import annotations

import logging
from typing import Any

from config.slim_loader import get as cfg_get
from inference.pipeline import SlimPipeline
from utils.cache import ResponseCache, _cache_key, get_cache

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self._pipeline = SlimPipeline()
        self._backend: str = cfg_get("model", "backend", "auto")
        self._cache: ResponseCache = get_cache()

    def generate(
        self,
        input_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        key = _cache_key(input_text, max_tokens, temperature)
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("cache hit key=%s", key)
            cached["cached"] = True
            return cached

        logger.info("generate | backend=%s | input_len=%d", self._backend, len(input_text))
        result = self._pipeline.run(
            input_text,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._cache.set(key, result)
        return result
