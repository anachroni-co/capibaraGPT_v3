"""Capibara Slim — stub backend (no weights required)."""
from __future__ import annotations

from typing import Any


class StubBackend:
    """Always-available backend that returns a canned response."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def is_available(self) -> bool:
        return True

    def generate(
        self,
        input_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        tokens = input_text.split()
        return {
            "output": f"[stub] received {len(tokens)} token(s): {input_text[:60]}",
            "model": self.name,
            "tokens_used": len(tokens),
        }
