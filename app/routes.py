"""Capibara Slim — HTTP route definitions."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.api_service import ApiService
from utils.cache import get_cache

logger = logging.getLogger(__name__)
router = APIRouter()
_api_service = ApiService()


class GenerateRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=8192)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    output: str
    model: str
    tokens_used: int


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "capibara-slim"}


@router.get("/metrics")
def metrics() -> dict:
    import utils.stats as _stats
    return {
        "service": "capibara-slim",
        "cache": get_cache().stats(),
        "requests": _stats.snapshot(),
    }


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        result = _api_service.generate(
            input_text=request.input,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return GenerateResponse(**result)
    except Exception as exc:
        logger.exception("generate failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/generate/stream")
def generate_stream(request: GenerateRequest) -> StreamingResponse:
    """Server-Sent Events streaming endpoint.

    Each token is emitted as an SSE data line:
        data: {"token": "hello "}\\n\\n

    The stream ends with:
        data: [DONE]\\n\\n

    Example (curl):
        curl -X POST http://localhost:8000/generate/stream \\
             -H 'Content-Type: application/json' \\
             -d '{"input": "hello"}' \\
             --no-buffer
    """
    from inference.streaming import SlimStreamer
    streamer = SlimStreamer()

    def _event_gen():
        try:
            for fragment in streamer.stream(
                request.input,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                payload = json.dumps({"token": fragment}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.exception("streaming error")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(_event_gen(), media_type="text/event-stream")
