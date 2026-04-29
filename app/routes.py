"""Capibara Slim — HTTP route definitions."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.api_service import ApiService

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
