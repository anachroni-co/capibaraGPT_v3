"""Capibara Slim — inference pipeline.

Orchestration flow:
  1. safety: input filter
  2. tool detection → tool executor (if tool: prefix)
  3. core executor (model inference)
  4. safety: output filter
  5. return
"""
from __future__ import annotations

import logging
from typing import Any

from config.slim_loader import get as cfg_get
from core.executor import SlimExecutor
from safety.input_filter import InputFilter
from safety.output_filter import OutputFilter
from tools.detector import detect_tool
from tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class SlimPipeline:
    def __init__(self) -> None:
        self._executor = SlimExecutor()
        self._tool_executor = ToolExecutor()
        self._input_filter = InputFilter(enabled=cfg_get("safety", "input_filter", True))
        self._output_filter = OutputFilter(enabled=cfg_get("safety", "output_filter", True))

    def run(
        self,
        input_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        text = input_text.strip()

        # 1. Input safety check
        in_check = self._input_filter.check(text)
        if not in_check.allowed:
            logger.warning("pipeline: input blocked — %s", in_check.reason)
            return {
                "output": f"[blocked] {in_check.reason}",
                "model": "safety",
                "tokens_used": 0,
                "latency_ms": 0.0,
                "blocked": True,
            }

        # 2. Tool detection
        tool_match = detect_tool(text)
        if tool_match is not None:
            tool_name, tool_input = tool_match
            tool_result = self._tool_executor.execute(tool_name, tool_input)
            raw_output = (
                tool_result["result"]
                if tool_result["error"] is None
                else f"[tool error] {tool_result['error']}"
            )
            return self._finish(raw_output, model=f"tool:{tool_name}", tokens_used=0)

        # 3. Model inference
        result = self._executor.run(text, max_tokens=max_tokens, temperature=temperature)
        return self._finish(
            result.get("output", ""),
            model=result.get("model", "unknown"),
            tokens_used=result.get("tokens_used", 0),
            extra={k: v for k, v in result.items() if k not in ("output", "model", "tokens_used")},
        )

    def _finish(
        self,
        raw: str,
        model: str,
        tokens_used: int,
        extra: dict | None = None,
    ) -> dict[str, Any]:
        # Output safety check
        out_check = self._output_filter.check(raw)
        if not out_check.allowed:
            logger.warning("pipeline: output blocked — %s", out_check.reason)
            output = "[blocked] output filtered"
        else:
            output = out_check.text

        result: dict[str, Any] = {
            "output": output,
            "model": model,
            "tokens_used": tokens_used,
        }
        if extra:
            result.update(extra)
        return result
