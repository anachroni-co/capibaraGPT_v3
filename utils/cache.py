"""Capibara Slim — simple in-memory response cache (T4.4).

LRU cache keyed on (input_text, max_tokens, temperature).
Disabled by default; enable via config/slim.yaml:

    cache:
      enabled: true
      max_size: 256
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

from config.slim_loader import get as cfg_get

logger = logging.getLogger(__name__)


def _cache_key(input_text: str, max_tokens: int, temperature: float) -> str:
    raw = f"{input_text}|{max_tokens}|{temperature:.4f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ResponseCache:
    def __init__(self, max_size: int = 256, enabled: bool = True) -> None:
        self._enabled = enabled
        self._max_size = max_size
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        if not self._enabled or key not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return dict(self._store[key])

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self._enabled:
            return
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = dict(value)
        if len(self._store) > self._max_size:
            evicted = next(iter(self._store))
            del self._store[evicted]

    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}


_default: ResponseCache | None = None


def get_cache() -> ResponseCache:
    global _default
    if _default is None:
        enabled: bool = cfg_get("cache", "enabled", False)
        max_size: int = int(cfg_get("cache", "max_size", 256))
        _default = ResponseCache(max_size=max_size, enabled=enabled)
    return _default
