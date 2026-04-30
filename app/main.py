"""Capibara Slim — API entry point."""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.auth import AuthMiddleware
from app.ratelimit import RateLimitMiddleware
from app.routes import router
from config.slim_loader import get as cfg_get
from utils.logger import new_request_id, request_id, setup_logging

setup_logging(level=cfg_get("logging", "level", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="Capibara Slim", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)


@app.middleware("http")
async def _trace(request: Request, call_next) -> Response:
    rid = new_request_id()
    token = request_id.set(rid)
    t0 = time.monotonic()
    response = await call_next(request)
    elapsed = round((time.monotonic() - t0) * 1000, 1)
    response.headers["X-Request-Id"] = rid
    logger.info(
        "%s %s %d",
        request.method,
        request.url.path,
        response.status_code,
        extra={"req_id": rid, "latency_ms": elapsed},
    )
    request_id.reset(token)
    return response


@app.exception_handler(Exception)
async def _global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error", "type": type(exc).__name__},
    )


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
