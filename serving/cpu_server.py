"""serving/cpu_server.py

Async HTTP inference server for capibaraGPT CPU deployment.

Architecture
------------
                    ┌──────────────┐
  HTTP client  ──►  │  FastAPI app  │  (asyncio, uvicorn)
                    └──────┬───────┘
                           │  asyncio.Queue (bounded, backpressure)
                    ┌──────▼───────┐
                    │  Worker pool │  (ThreadPoolExecutor — CPU bound)
                    └──────┬───────┘
                           │
                    ┌──────▼───────────────────────┐
                    │  LMTPCachedDecoder + Int8ByteLM│
                    │  + ThinkAnywhereStreamFilter   │
                    └──────────────────────────────┘

All heavy inference runs in a thread pool so the asyncio event loop stays
responsive for health checks and new request intake.  The queue is bounded
to MAX_QUEUE_DEPTH — requests beyond that get 503 immediately (backpressure)
rather than silently piling up memory.

Usage
-----
    # With a trained ByteLM + LMTPHeads:
    from serving.cpu_server import build_app, ServerConfig
    import uvicorn

    app = build_app(backbone, heads, ServerConfig())
    uvicorn.run(app, host="0.0.0.0", port=8080)

    # Or from CLI:
    python -m serving.cpu_server --port 8080 --hidden 256

API
---
POST /generate
    Body:  {"prompt": "...", "max_new_tokens": 128, "temperature": 1.0}
    Reply: {"tokens": [...], "text": "...", "stats": {...}}

GET  /health
    Reply: {"status": "ok", "queue_depth": N, "total_served": N}

GET  /metrics
    Reply: {"avg_tok_per_s": ..., "p50_latency_ms": ..., ...}
"""
from __future__ import annotations

import asyncio
import logging
import os
import statistics
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    _FASTAPI = True
except ImportError:
    _FASTAPI = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1            # CPU-bound inference: keep = n_physical_cores
    max_queue_depth: int = 32   # requests beyond this → 503
    max_new_tokens: int = 256
    default_temperature: float = 1.0
    eos_token_id: Optional[int] = None
    gate_checkpoint: str = ""   # path to .npz gate checkpoint ("" = no gate)
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Server state (shared across requests)
# ---------------------------------------------------------------------------

class ServerState:
    def __init__(self, backbone, heads, cfg: ServerConfig) -> None:
        self.backbone = backbone
        self.heads = heads
        self.cfg = cfg

        # Try INT8 quantisation
        try:
            from inference.int8_inference import Int8ByteLM
            self.model = Int8ByteLM.from_bytelm(backbone)
            logger.info("Serving with INT8 model")
        except Exception as e:
            logger.warning("INT8 unavailable (%s); using FP32", e)
            self.model = backbone

        # Try gate
        self.gate = None
        if cfg.gate_checkpoint:
            try:
                from core.think_anywhere import ThinkAnywhereGate
                self.gate = ThinkAnywhereGate.load(cfg.gate_checkpoint)
                logger.info("Gate loaded from %s", cfg.gate_checkpoint)
            except Exception as e:
                logger.warning("Gate load failed (%s); running without gate", e)

        # Thread pool for inference
        self.executor = ThreadPoolExecutor(max_workers=cfg.workers,
                                           thread_name_prefix="inf")
        # Metrics
        self._total_served = 0
        self._latencies_ms: deque[float] = deque(maxlen=1000)
        self._tok_per_s_samples: deque[float] = deque(maxlen=1000)

    def _run_inference(self, prompt_ids: list[int], max_new_tokens: int,
                       temperature: float) -> dict:
        """Blocking inference — runs in thread pool."""
        from inference.cpu_kv_cache import LMTPCachedDecoder, CacheDecodeConfig
        from core.think_anywhere import ThinkAnywhereStreamFilter

        cfg = CacheDecodeConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            greedy=(temperature <= 0.0),
            eos_token_id=self.cfg.eos_token_id,
        )
        decoder = LMTPCachedDecoder(self.model, self.heads, cfg)
        filt = ThinkAnywhereStreamFilter(gate=self.gate)

        t0 = time.perf_counter()
        token_ids = decoder.generate(prompt_ids, max_new_tokens=max_new_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        new_ids = token_ids[len(prompt_ids):]
        # Decode bytes → text, run stream filter
        raw_text = bytes([t % 256 for t in new_ids]).decode("utf-8", errors="replace")
        clean_text = ""
        for ch in raw_text:
            clean_text += filt.feed(ch)
        clean_text += filt.flush()

        stats = decoder.last_stats
        stats["elapsed_ms"] = round(elapsed_ms, 1)

        self._total_served += 1
        self._latencies_ms.append(elapsed_ms)
        if stats.get("tok_per_s"):
            self._tok_per_s_samples.append(stats["tok_per_s"])

        return {"tokens": new_ids, "text": clean_text, "stats": stats}

    @property
    def metrics(self) -> dict:
        lats = list(self._latencies_ms)
        tps = list(self._tok_per_s_samples)
        return {
            "total_served": self._total_served,
            "avg_tok_per_s": round(statistics.mean(tps), 1) if tps else 0,
            "p50_latency_ms": round(statistics.median(lats), 1) if lats else 0,
            "p95_latency_ms": round(
                sorted(lats)[int(len(lats) * 0.95)], 1
            ) if lats else 0,
        }


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def build_app(backbone, heads, cfg: Optional[ServerConfig] = None) -> "FastAPI":
    """Build and return the FastAPI application.

    Call uvicorn.run(build_app(backbone, heads), ...) to serve.
    """
    if not _FASTAPI:
        raise ImportError("FastAPI required: pip install fastapi uvicorn")

    cfg = cfg or ServerConfig()
    state = ServerState(backbone, heads, cfg)
    _queue: asyncio.Queue = None   # initialised in lifespan

    app = FastAPI(title="CapibaraGPT CPU Server", version="1.0")

    # ── Request / response models ──────────────────────────────────────────

    class GenerateRequest(BaseModel):
        prompt: str = Field(..., min_length=1, max_length=8192)
        max_new_tokens: int = Field(128, ge=1, le=cfg.max_new_tokens)
        temperature: float = Field(cfg.default_temperature, ge=0.0, le=4.0)

    # ── Lifespan ───────────────────────────────────────────────────────────

    @app.on_event("startup")
    async def startup():
        nonlocal _queue
        _queue = asyncio.Queue(maxsize=cfg.max_queue_depth)
        logger.info("CapibaraGPT CPU server ready — workers=%d", cfg.workers)

    # ── Routes ────────────────────────────────────────────────────────────

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        if _queue.full():
            raise HTTPException(status_code=503, detail="Server queue full — retry later")

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        await _queue.put(fut)

        # Encode prompt as byte ids
        prompt_ids = [b for b in req.prompt.encode("utf-8", errors="replace")]

        try:
            result = await loop.run_in_executor(
                state.executor,
                state._run_inference,
                prompt_ids,
                req.max_new_tokens,
                req.temperature,
            )
        except Exception as exc:
            logger.exception("Inference error")
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            try:
                _queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        return JSONResponse(result)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "queue_depth": _queue.qsize() if _queue else 0,
            "total_served": state._total_served,
        }

    @app.get("/metrics")
    async def metrics():
        return state.metrics

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="CapibaraGPT CPU inference server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--leap-k", type=int, default=4)
    p.add_argument("--checkpoint", default="",
                   help="Path to .npz checkpoint (backbone+heads). "
                        "If empty, starts with untrained weights.")
    p.add_argument("--gate-checkpoint", default="")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    # Build or load backbone + heads
    from scripts.train_lmtp_cpu import ByteLM, LMTPHeads  # type: ignore
    backbone = ByteLM(vocab=512, hidden=args.hidden)
    heads = LMTPHeads(hidden=args.hidden, vocab=512,
                      n_head=args.n_head, leap_k=args.leap_k)

    if args.checkpoint:
        import pickle
        with open(args.checkpoint, "rb") as f:
            ckpt = pickle.load(f)  # nosec B301 - trusted local checkpoint, not user input
        backbone.__dict__.update(ckpt.get("backbone", {}))
        logger.info("Checkpoint loaded from %s", args.checkpoint)

    cfg = ServerConfig(
        host=args.host,
        port=args.port,
        workers=args.workers,
        gate_checkpoint=args.gate_checkpoint,
    )

    try:
        import uvicorn
    except ImportError:
        print("uvicorn required: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    app = build_app(backbone, heads, cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    _cli()
