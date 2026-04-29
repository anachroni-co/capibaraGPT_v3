"""Capibara Slim — structured JSON logging (T4.3).

Call setup_logging() once at startup. Every logger in the app then
emits JSON lines that are easy to parse by log aggregators.

Example line:
  {"ts":"2026-04-29T21:00:00.123Z","level":"INFO","logger":"inference.pipeline",
   "msg":"pipeline ok","req_id":"abc123","model":"mamba","latency_ms":12.3}
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

# Per-request ID propagated via context var
request_id: ContextVar[str] = ContextVar("request_id", default="")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = request_id.get("")
        if rid:
            doc["req_id"] = rid
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        # Attach any extra kwargs passed to the logger call
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                doc[key] = val
        return json.dumps(doc, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
