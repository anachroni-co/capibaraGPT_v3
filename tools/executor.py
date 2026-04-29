"""Capibara Slim — tool executor (T3.2).

Runs a registered tool with a basic timeout so a slow or hanging tool
cannot block the inference pipeline indefinitely.
"""
from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

from tools.registry import get_tool_registry

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 5.0  # seconds


class ToolExecutor:
    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout
        self._registry = get_tool_registry()

    def execute(self, tool_name: str, tool_input: str) -> dict[str, Any]:
        fn = self._registry.get(tool_name)
        if fn is None:
            logger.warning("ToolExecutor: unknown tool '%s'", tool_name)
            return {
                "tool": tool_name,
                "result": None,
                "error": f"unknown tool '{tool_name}'",
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn, tool_input)
            try:
                result = future.result(timeout=self._timeout)
                logger.info("ToolExecutor: '%s' OK", tool_name)
                return {"tool": tool_name, "result": result, "error": None}
            except concurrent.futures.TimeoutError:
                logger.error("ToolExecutor: '%s' timed out after %.1fs", tool_name, self._timeout)
                return {"tool": tool_name, "result": None, "error": "timeout"}
            except Exception as exc:
                logger.error("ToolExecutor: '%s' raised %s", tool_name, exc)
                return {"tool": tool_name, "result": None, "error": str(exc)}
