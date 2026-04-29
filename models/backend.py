"""Capibara Slim — backend protocol and shared types."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelBackend(Protocol):
    """Common interface every model backend must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def is_available(self) -> bool: ...

    def generate(
        self,
        input_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]: ...
