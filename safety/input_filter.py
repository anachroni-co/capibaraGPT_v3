"""Capibara Slim — input filter (T3.4).

Lightweight regex-based filter applied before the input reaches the model.
Returns a FilterResult: allowed=True means the input is clean.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FilterResult:
    allowed: bool
    reason: str = ""
    matched_patterns: list[str] = field(default_factory=list)


# Patterns that trigger a block. Extend as needed.
_BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(jailbreak|ignore previous instructions|bypass safety)\b", re.I),
    re.compile(r"\b(how to (make|build|create) (a )?(bomb|weapon|explosive))\b", re.I),
    re.compile(r"(prompt injection|system prompt|forget everything|new instructions:)", re.I),
]

_MAX_INPUT_LENGTH = 8192


class InputFilter:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def check(self, text: str) -> FilterResult:
        if not self._enabled:
            return FilterResult(allowed=True)

        if len(text) > _MAX_INPUT_LENGTH:
            return FilterResult(allowed=False, reason="input too long")

        matched = [p.pattern for p in _BLOCKED_PATTERNS if p.search(text)]
        if matched:
            return FilterResult(
                allowed=False,
                reason="blocked pattern detected",
                matched_patterns=matched,
            )

        return FilterResult(allowed=True)
