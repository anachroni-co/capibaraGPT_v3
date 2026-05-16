#!/usr/bin/env python3
"""OpenAI-compatible HTTP server for cunca-v2 cascade inference.

Exposes:
  POST /v1/chat/completions   — chat completions (streaming + batch)
  GET  /v1/models             — list available models
  GET  /health                — liveness check

Auth: Authorization: Bearer <CUNCA_API_KEY>

Usage:
    python scripts/serve.py \\
        --draft  checkpoints/cunca_v2_small_gl/soup_uniform.pkl \\
        --draft-lora checkpoints/lora/gestoria_gl_v2_small/lora_final.pkl \\
        --target checkpoints/cunca_v2_gl/soup_uniform.pkl \\
        --lora   checkpoints/lora/gestoria_gl_v2/lora_final.pkl \\
        --port 8000 --api-key sk-cunca-xxx

    # API key via env var:
    CUNCA_API_KEY=sk-cunca-xxx python scripts/serve.py ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("serve")

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Global state ──────────────────────────────────────────────────────────────

_state: dict = {}          # draft_m, target_m, k, api_key
_lock: asyncio.Lock        # serialises JAX calls (not thread-safe)


# ── Request / response schemas ────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "cunca-v2"
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    n: int = 1


# ── Auth ──────────────────────────────────────────────────────────────────────

def _check_auth(request: Request) -> None:
    expected = _state.get("api_key")
    if not expected:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Prompt builder ────────────────────────────────────────────────────────────

def _messages_to_prompt(messages: list[Message]) -> str:
    # Model was trained on bare prompt→response pairs (no chat markers).
    # Use the last user message as the prompt; prepend system content if present.
    system = next((m.content for m in messages if m.role == "system"), None)
    user   = next((m.content for m in reversed(messages) if m.role == "user"), "")
    if system:
        return f"{system}\n\n{user}"
    return user


# ── Inference worker (runs in executor to avoid blocking event loop) ──────────

def _generate_sync(prompt: str, max_tokens: int, temperature: float) -> tuple[str, dict]:
    from scripts.cascade_inference import cascade_generate
    return cascade_generate(
        prompt,
        _state["draft_m"],
        _state["target_m"],
        k=_state["k"],
        max_tokens=max_tokens,
        temperature=temperature,
        seed=int(time.time()) % (2**31),
        stream=False,
    )


async def _generate(prompt: str, max_tokens: int, temperature: float) -> tuple[str, dict]:
    loop = asyncio.get_event_loop()
    async with _lock:
        return await loop.run_in_executor(
            None, _generate_sync, prompt, max_tokens, temperature
        )


# ── Streaming helper ──────────────────────────────────────────────────────────

async def _stream_response(
    completion_id: str,
    model: str,
    text: str,
) -> AsyncIterator[str]:
    # Send text in small chunks to simulate token streaming
    chunk_size = 4
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0)  # yield control

    # Final chunk
    data = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _lock
    _lock = asyncio.Lock()

    args = _state["args"]
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=true "
        "--xla_cpu_enable_fast_math=true "
        "--xla_force_host_platform_device_count=1")

    from scripts.cascade_inference import load_model
    logger.info("Loading draft model: %s", args.draft)
    _state["draft_m"] = load_model(args.draft, lora_path=args.draft_lora)
    logger.info("Loading target model: %s", args.target)
    _state["target_m"] = load_model(args.target, lora_path=args.lora)
    _state["k"] = args.k
    logger.info("Models ready — serving on port %d", args.port)
    yield


app = FastAPI(title="cunca-v2 API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "draft":  _state["draft_m"].preset if "draft_m" in _state else None,
        "target": _state["target_m"].preset if "target_m" in _state else None,
        "k":      _state.get("k"),
    }


@app.get("/")
async def root():
    return {"name": "cunca-v2", "version": "1.0", "docs": "/docs"}


@app.get("/v1/models")
async def list_models(request: Request):
    _check_auth(request)
    return {
        "object": "list",
        "data": [
            {
                "id": "cunca-v2",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anachroni",
            }
        ],
    }


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request, body: ChatRequest):
    _check_auth(request)

    prompt = _messages_to_prompt(body.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    text, stats = await _generate(prompt, body.max_tokens, body.temperature)

    if body.stream:
        return StreamingResponse(
            _stream_response(completion_id, body.model, text),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    prompt_tokens = len(prompt.encode("utf-8"))
    completion_tokens = stats.get("total_tokens", len(text.encode("utf-8")))

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        },
        "x_cunca_stats": stats,
    })


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft",      required=True)
    parser.add_argument("--draft-lora", default=None)
    parser.add_argument("--target",     required=True)
    parser.add_argument("--lora",       default=None)
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--k",          type=int, default=5,
                        help="Draft tokens per step (default: 5)")
    parser.add_argument("--threads",    type=int, default=32,
                        help="OMP_NUM_THREADS (default: 32)")
    parser.add_argument("--api-key",    default=None,
                        help="Bearer API key (or set CUNCA_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("CUNCA_API_KEY")
    if not api_key:
        logger.warning("No API key set — server is open to all requests")
    _state["api_key"] = api_key
    _state["args"]    = args

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
