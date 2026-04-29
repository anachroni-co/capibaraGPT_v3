"""Capibara Slim — tool detection (T3.3).

Parses inputs of the form:

    tool: <tool_name> <tool_input>
    tool: <tool_name>

Returns a (tool_name, tool_input) tuple or None if no tool invocation
is detected.
"""
from __future__ import annotations

import re

_TOOL_PATTERN = re.compile(
    r"^\s*tool\s*:\s*(?P<name>\w+)(?:\s+(?P<input>.+))?$",
    re.IGNORECASE | re.DOTALL,
)


def detect_tool(text: str) -> tuple[str, str] | None:
    """Return (tool_name, tool_input) if text is a tool call, else None."""
    m = _TOOL_PATTERN.match(text.strip())
    if m is None:
        return None
    name = m.group("name").lower()
    arg = (m.group("input") or "").strip()
    return name, arg
