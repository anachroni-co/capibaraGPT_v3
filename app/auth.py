"""Capibara Slim — API key authentication middleware.

Auth is opt-in: it activates only when `auth.enabled = true` in slim.yaml
AND the `CAPIBARA_API_KEY` environment variable is set.
When disabled (the default) all requests pass through unchanged.

Clients must send:
    Authorization: Bearer <api-key>
"""
from __future__ import annotations

import os
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config.slim_loader import get as cfg_get

logger = logging.getLogger(__name__)

_UNPROTECTED = frozenset({"/health", "/metrics", "/docs", "/openapi.json", "/redoc"})


def _configured_key() -> str | None:
    return os.environ.get("CAPIBARA_API_KEY") or None


def _auth_enabled() -> bool:
    return bool(cfg_get("auth", "enabled", False)) and _configured_key() is not None


class AuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid Bearer token when auth is enabled."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _UNPROTECTED or not _auth_enabled():
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "missing or malformed Authorization header"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header.removeprefix("Bearer ").strip()
        if token != _configured_key():
            logger.warning("invalid API key from %s", request.client and request.client.host)
            return JSONResponse(
                status_code=403,
                content={"detail": "invalid API key"},
            )

        return await call_next(request)
