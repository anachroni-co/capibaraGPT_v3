"""Capibara Slim — per-IP sliding window rate limiter.

Rate limiting is opt-in via `rate_limit.enabled = true` in slim.yaml.
When disabled (the default) all requests pass through unchanged.

Algorithm: sliding window — counts requests in the last 60 s per IP.
Returns HTTP 429 with `Retry-After` header when the limit is exceeded.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config.slim_loader import get as cfg_get

logger = logging.getLogger(__name__)

_UNPROTECTED = frozenset({"/health", "/metrics", "/docs", "/openapi.json", "/redoc"})
_WINDOW_SECONDS = 60


class _SlidingWindow:
    """Thread-safe sliding-window counter for one IP."""

    __slots__ = ("_lock", "_timestamps")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def check_and_record(self, limit: int) -> tuple[bool, int]:
        """Return (allowed, current_count_after_record)."""
        now = time.monotonic()
        cutoff = now - _WINDOW_SECONDS
        with self._lock:
            # Evict expired entries
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            count = len(self._timestamps)
            if count >= limit:
                return False, count
            self._timestamps.append(now)
            return True, count + 1


class RateLimiter:
    """Registry of per-IP sliding windows."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.limit = requests_per_minute
        self._windows: dict[str, _SlidingWindow] = {}
        self._lock = threading.Lock()

    def check(self, ip: str) -> tuple[bool, int]:
        with self._lock:
            if ip not in self._windows:
                self._windows[ip] = _SlidingWindow()
            window = self._windows[ip]
        return window.check_and_record(self.limit)


_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def _get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                rpm = int(cfg_get("rate_limit", "requests_per_minute", 60))
                _limiter = RateLimiter(rpm)
    return _limiter


def _rate_limit_enabled() -> bool:
    return bool(cfg_get("rate_limit", "enabled", False))


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return (request.client and request.client.host) or "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce per-IP request rate."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _UNPROTECTED or not _rate_limit_enabled():
            return await call_next(request)

        ip = _client_ip(request)
        allowed, count = _get_limiter().check(ip)
        if not allowed:
            logger.warning("rate limit exceeded for %s (%d req/min)", ip, count)
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded", "limit": _get_limiter().limit},
                headers={"Retry-After": str(_WINDOW_SECONDS)},
            )

        return await call_next(request)
