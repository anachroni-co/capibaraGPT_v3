"""Capibara Slim — inference pipeline.

Orchestration flow:
  1. safety: input filter
  2. tool detection → tool executor (if tool: prefix)
  3. core executor (model inference) — with configurable timeout
  4. safety: output filter
  5. return
"""
from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

from config.slim_loader import get as cfg_get
from core.executor import SlimExecutor
from safety.input_filter import InputFilter
from safety.output_filter import OutputFilter
from tools.detector import detect_tool
from tools.executor import ToolExecutor

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0  # seconds


class SlimPipeline:
    def __init__(self) -> None:
        self._executor = SlimExecutor()
        self._tool_executor = ToolExecutor()
        self._input_filter = InputFilter(enabled=cfg_get("safety", "input_filter", True))
        self._output_filter = OutputFilter(enabled=cfg_get("safety", "output_filter", True))
        self._timeout: float = float(cfg_get("inference", "timeout_seconds", _DEFAULT_TIMEOUT))

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

        # 3. Model inference (with timeout)
        try:
            result = self._run_with_timeout(text, max_tokens, temperature)
        except TimeoutError:
            logger.error("pipeline: inference timed out after %.1fs — using stub fallback", self._timeout)
            result = {"output": "[timeout] inference timed out", "model": "fallback", "tokens_used": 0}
        except Exception as exc:
            logger.exception("pipeline: inference error — %s", exc)
            result = {"output": f"[error] {exc}", "model": "fallback", "tokens_used": 0}

        return self._finish(
            result.get("output", ""),
            model=result.get("model", "unknown"),
            tokens_used=result.get("tokens_used", 0),
            extra={k: v for k, v in result.items() if k not in ("output", "model", "tokens_used")},
        )

    def _run_with_timeout(
        self, text: str, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._executor.run, text, max_tokens, temperature)
            try:
                return future.result(timeout=self._timeout)
            except concurrent.futures.TimeoutError as exc:
                raise TimeoutError from exc

    def _finish(
        self,
        raw: str,
        model: str,
        tokens_used: int,
        extra: dict | None = None,
    ) -> dict[str, Any]:
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
