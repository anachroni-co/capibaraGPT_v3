"""Capibara Slim — tool registry (T3.1).

Registers Python callables as named tools. Tools must accept a single
string argument (the extracted tool input) and return a string result.
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

ToolFn = Callable[[str], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn, *, description: str = "") -> None:
        self._tools[name] = fn
        logger.debug("ToolRegistry: registered '%s'", name)

    def get(self, name: str) -> ToolFn | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------------
# Default registry with built-in tools
# ---------------------------------------------------------------------------

def _tool_echo(arg: str) -> str:
    return arg


def _tool_upper(arg: str) -> str:
    return arg.upper()


def _tool_len(arg: str) -> str:
    return str(len(arg.split()))


def _build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("echo", _tool_echo, description="Return input unchanged")
    reg.register("upper", _tool_upper, description="Uppercase the input")
    reg.register("word_count", _tool_len, description="Count words in input")
    return reg


_default: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _default
    if _default is None:
        _default = _build_default_registry()
    return _default
