"""Capibara Slim — inference pipeline.

Thin orchestration layer: preprocess → executor → postprocess.
The executor owns routing and backend selection.
"""
from __future__ import annotations

import logging
from typing import Any

from core.executor import SlimExecutor

logger = logging.getLogger(__name__)


class SlimPipeline:
    def __init__(self) -> None:
        self._executor = SlimExecutor()

    def run(
        self,
        input_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        text = self._preprocess(input_text)
        result = self._executor.run(text, max_tokens=max_tokens, temperature=temperature)
        result["output"] = self._postprocess(result.get("output", ""))
        return result

    def _preprocess(self, text: str) -> str:
        return text.strip()

    def _postprocess(self, text: str) -> str:
        return text.strip()
