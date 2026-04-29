"""Capibara Slim — output filter (T3.5).

Applied to model/tool output before it is returned to the caller.
Sanitises or blocks content that should not be surfaced.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class OutputFilterResult:
    allowed: bool
    text: str          # cleaned text (may differ from input)
    reason: str = ""
    redacted: list[str] = field(default_factory=list)


# Patterns that trigger redaction (replace match with [REDACTED]).
_REDACT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),     # credit card-like
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),  # email
]

# Patterns that block output entirely.
_BLOCK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(step[- ]by[- ]step.*?(bomb|weapon|explosive))\b", re.I | re.S),
]


class OutputFilter:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def check(self, text: str) -> OutputFilterResult:
        if not self._enabled:
            return OutputFilterResult(allowed=True, text=text)

        for pat in _BLOCK_PATTERNS:
            if pat.search(text):
                return OutputFilterResult(
                    allowed=False,
                    text="",
                    reason="blocked output pattern",
                )

        redacted: list[str] = []
        cleaned = text
        for pat in _REDACT_PATTERNS:
            if pat.search(cleaned):
                redacted.append(pat.pattern)
                cleaned = pat.sub("[REDACTED]", cleaned)

        return OutputFilterResult(allowed=True, text=cleaned, redacted=redacted)
