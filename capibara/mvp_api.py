"""
MVP production API for CapibaraGPT v3.

Frozen P0 surface:
- POST /v1/generate
- GET /health/live
- GET /health/ready
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "FastAPI is required for MVP API. Install with: pip install fastapi uvicorn"
    ) from exc


def _load_hybrid_inference_symbols():
    """Load hybrid inference symbols without importing inference.__init__ side effects."""
    root = Path(__file__).resolve().parents[1]
    engine_path = root / "inference" / "hybrid_inference_engine.py"
    spec = importlib.util.spec_from_file_location("capibara_hybrid_engine", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load hybrid inference module from {engine_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HybridInferenceEngine, mod.InferenceBackend, mod.InferenceConfig


HybridInferenceEngine, InferenceBackend, InferenceConfig = _load_hybrid_inference_symbols()

logger = logging.getLogger("capibara.mvp_api")
SERVICE_NAME = os.getenv("CAPIBARA_SERVICE_NAME", "capibara-api")
SERVICE_ENV = os.getenv("CAPIBARA_ENV", "dev")
SERVICE_VERSION = os.getenv("CAPIBARA_SERVICE_VERSION", "1.0.0")

MAX_PROMPT_CHARS = int(os.getenv("CAPIBARA_MAX_PROMPT_CHARS", "12000"))
MAX_NEW_TOKENS = int(os.getenv("CAPIBARA_MAX_NEW_TOKENS", "1024"))
REQUEST_TIMEOUT_S = float(os.getenv("CAPIBARA_REQUEST_TIMEOUT_S", "30"))
STRICT_RUNTIME = os.getenv("CAPIBARA_STRICT_RUNTIME", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    max_new_tokens: int = Field(128, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class GenerateResponse(BaseModel):
    text: str
    usage: Usage
    latency_ms: float
    model_version: str
    request_id: str


class _FakeResult:
    def __init__(self, text: str, prompt_tokens: int, tokens_generated: int) -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.tokens_generated = tokens_generated


class _FakeInferenceEngine:
    """Deterministic lightweight engine for containerized smoke tests."""

    async def generate(self, prompt: str, **kwargs):
        max_tokens = int(kwargs.get("max_tokens", 32))
        token_count = max(1, min(max_tokens, 32))
        text = f"fake-response:{prompt[:64]}"
        return _FakeResult(text=text, prompt_tokens=max(1, len(prompt.split())), tokens_generated=token_count)


class MVPService:
    """Thin service wrapper around HybridInferenceEngine for API use."""

    def __init__(self) -> None:
        backend = os.getenv("CAPIBARA_BACKEND", "cpu").strip().lower()
        backend_map = {
            "cpu": InferenceBackend.CPU,
            "gpu": InferenceBackend.GPU_CUDA,
            "tpu": InferenceBackend.TPU_V6E,
            "auto": InferenceBackend.AUTO,
        }
        model_path = os.getenv("CAPIBARA_MODEL_PATH", "").strip()
        use_fake_backend = os.getenv("CAPIBARA_FAKE_BACKEND", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        self.model_version = os.getenv("CAPIBARA_MODEL_VERSION", "mvp-v1")
        if use_fake_backend:
            self.engine = _FakeInferenceEngine()
            self._ready = True
            self._ready_reason = "ready"
            self.model_version = os.getenv("CAPIBARA_MODEL_VERSION", "mvp-fake-v1")
            return

        self.engine = HybridInferenceEngine(
            InferenceConfig(
                backend=backend_map.get(backend, InferenceBackend.CPU),
                model_path=model_path,
            )
        )
        self._ready = False
        self._ready_reason = "model_not_loaded"

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        self.engine.load_model(model_path)
        self._ready = True
        self._ready_reason = "ready"

    @property
    def ready(self) -> bool:
        return self._ready and self.engine is not None

    @property
    def ready_reason(self) -> str:
        return self._ready_reason


def _log_event(event: str, **payload: object) -> None:
    """Emit structured JSON logs for API events."""
    record = {
        "event": event,
        "service": SERVICE_NAME,
        "env": SERVICE_ENV,
        "version": SERVICE_VERSION,
        **payload,
    }
    logger.info(json.dumps(record, ensure_ascii=True, sort_keys=True))


def create_app(service: Optional[MVPService] = None) -> FastAPI:
    service = service or MVPService()
    app = FastAPI(title="CapibaraGPT MVP API", version="1.0.0")

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        incoming = request.headers.get("X-Request-Id", "").strip()
        incoming_trace = request.headers.get("X-Trace-Id", "").strip()
        request_id = incoming or str(uuid.uuid4())
        trace_id = incoming_trace or request_id
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        start = time.time()
        try:
            response = await call_next(request)
        except HTTPException as exc:
            detail = exc.detail
            if isinstance(detail, dict):
                detail.setdefault("request_id", request_id)
            else:
                detail = {"error": "http_exception", "message": str(detail), "request_id": request_id}
            latency_ms = (time.time() - start) * 1000.0
            _log_event(
                "request_http_exception",
                request_id=request_id,
                trace_id=trace_id,
                path=request.url.path,
                method=request.method,
                status_code=exc.status_code,
                latency_ms=round(latency_ms, 3),
            )
            response = JSONResponse(status_code=exc.status_code, content={"detail": detail})
            response.headers["X-Request-Id"] = request_id
            response.headers["X-Trace-Id"] = trace_id
            return response
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000.0
            _log_event(
                "request_unhandled_exception",
                request_id=request_id,
                trace_id=trace_id,
                path=request.url.path,
                method=request.method,
                latency_ms=round(latency_ms, 3),
                error=str(exc),
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": {
                        "error": "internal_server_error",
                        "message": str(exc),
                        "request_id": request_id,
                    }
                },
            )
            response.headers["X-Request-Id"] = request_id
            response.headers["X-Trace-Id"] = trace_id
            return response

        response.headers["X-Request-Id"] = request_id
        response.headers["X-Trace-Id"] = trace_id
        latency_ms = (time.time() - start) * 1000.0
        _log_event(
            "request_complete",
            request_id=request_id,
            trace_id=trace_id,
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 3),
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        trace_id = getattr(request.state, "trace_id", request_id)
        detail = exc.detail
        if isinstance(detail, dict):
            detail.setdefault("request_id", request_id)
        else:
            detail = {"error": "http_exception", "message": str(detail), "request_id": request_id}
        response = JSONResponse(status_code=exc.status_code, content={"detail": detail})
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Trace-Id"] = trace_id
        return response

    @app.get("/health/live")
    async def health_live(request: Request) -> dict:
        return {
            "status": "alive",
            "strict_runtime": STRICT_RUNTIME,
            "timestamp": int(time.time()),
            "request_id": request.state.request_id,
        }

    @app.get("/health/ready")
    async def health_ready(request: Request) -> dict:
        if not service.ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "reason": service.ready_reason,
                    "strict_runtime": STRICT_RUNTIME,
                    "request_id": request.state.request_id,
                },
            )
        return {
            "status": "ready",
            "strict_runtime": STRICT_RUNTIME,
            "request_id": request.state.request_id,
        }

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
        request_id = request.state.request_id
        trace_id = request.state.trace_id

        if not service.ready:
            _log_event(
                "generate_not_ready",
                request_id=request_id,
                trace_id=trace_id,
                reason=service.ready_reason,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_not_ready",
                    "reason": service.ready_reason,
                    "request_id": request_id,
                },
            )

        start = time.time()
        try:
            result = await asyncio.wait_for(
                service.engine.generate(
                    req.prompt,
                    max_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                ),
                timeout=REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            latency_ms = (time.time() - start) * 1000.0
            _log_event(
                "generate_timeout",
                request_id=request_id,
                trace_id=trace_id,
                latency_ms=round(latency_ms, 3),
                timeout_s=REQUEST_TIMEOUT_S,
            )
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "generation_timeout",
                    "message": f"Request exceeded timeout of {REQUEST_TIMEOUT_S}s",
                    "request_id": request_id,
                },
            ) from exc
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000.0
            _log_event(
                "generate_error",
                request_id=request_id,
                trace_id=trace_id,
                latency_ms=round(latency_ms, 3),
                error=str(exc),
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "generation_failed",
                    "message": str(exc),
                    "request_id": request_id,
                },
            ) from exc

        latency_ms = (time.time() - start) * 1000.0
        _log_event(
            "generate_ok",
            request_id=request_id,
            trace_id=trace_id,
            latency_ms=round(latency_ms, 3),
            input_tokens=result.prompt_tokens,
            output_tokens=result.tokens_generated,
            model_version=service.model_version,
        )
        return GenerateResponse(
            text=result.text,
            usage=Usage(
                input_tokens=result.prompt_tokens,
                output_tokens=result.tokens_generated,
            ),
            latency_ms=latency_ms,
            model_version=service.model_version,
            request_id=request_id,
        )

    return app


app = create_app()


def run() -> None:
    """Run MVP API with uvicorn."""
    import uvicorn

    host = os.getenv("CAPIBARA_API_HOST", "0.0.0.0")
    port = int(os.getenv("CAPIBARA_API_PORT", "8000"))
    uvicorn.run("capibara.mvp_api:app", host=host, port=port, reload=False)

