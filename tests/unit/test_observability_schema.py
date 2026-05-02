"""
Observability Schema Tests - Unit tests for observability and monitoring schemas.

This module provides tests for observability schemas, validating
metrics collection, logging formats, and monitoring data structures.

Author: Skydesk International Dev Team.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # starlette.testclient requires httpx at runtime
testclient = pytest.importorskip("fastapi.testclient")

import capibara.mvp_api as mvp_api


@dataclass
class _Result:
    text: str = "ok"
    prompt_tokens: int = 3
    tokens_generated: int = 2


class _OkEngine:
    async def generate(self, prompt: str, **kwargs):
        return _Result(text=f"echo:{prompt}")


class _TimeoutEngine:
    async def generate(self, prompt: str, **kwargs):
        await asyncio.sleep(0.01)
        return _Result(text=prompt)


class _FailEngine:
    async def generate(self, prompt: str, **kwargs):
        raise RuntimeError("forced_failure")


class _Service:
    def __init__(self, ready: bool = True, engine: Any | None = None):
        self._ready = ready
        self._ready_reason = "ready" if ready else "model_not_loaded"
        self.model_version = "test-model"
        self.engine = engine or _OkEngine()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def ready_reason(self) -> str:
        return self._ready_reason


COMMON_SCHEMA = {
    "event": str,
    "service": str,
    "env": str,
    "version": str,
}

EVENT_SCHEMA = {
    "request_complete": {
        "request_id": str,
        "trace_id": str,
        "path": str,
        "method": str,
        "status_code": int,
        "latency_ms": (int, float),
    },
    "generate_ok": {
        "request_id": str,
        "trace_id": str,
        "latency_ms": (int, float),
        "input_tokens": int,
        "output_tokens": int,
        "model_version": str,
    },
    "generate_not_ready": {
        "request_id": str,
        "trace_id": str,
        "reason": str,
    },
    "generate_timeout": {
        "request_id": str,
        "trace_id": str,
        "latency_ms": (int, float),
        "timeout_s": (int, float),
    },
    "generate_error": {
        "request_id": str,
        "trace_id": str,
        "latency_ms": (int, float),
        "error": str,
    },
}


def _collect_api_events(caplog) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for rec in caplog.records:
        if rec.name != "capibara.mvp_api":
            continue
        payload = json.loads(rec.getMessage())
        events.append(payload)
    return events


def _assert_schema(event: Dict[str, Any]) -> None:
    for key, typ in COMMON_SCHEMA.items():
        assert key in event, f"Missing common field: {key}"
        assert isinstance(event[key], typ), f"Invalid type for {key}"

    specific = EVENT_SCHEMA.get(event["event"])
    if specific:
        for key, typ in specific.items():
            assert key in event, f"Missing field `{key}` in event `{event['event']}`"
            assert isinstance(event[key], typ), f"Invalid type for `{key}` in event `{event['event']}`"


def test_observability_log_schema_for_mvp_api(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="capibara.mvp_api")

    # Scenario 1: success path
    app_ok = mvp_api.create_app(service=_Service(ready=True, engine=_OkEngine()))
    client_ok = testclient.TestClient(app_ok)
    r1 = client_ok.post("/v1/generate", json={"prompt": "hello"})
    assert r1.status_code == 200

    # Scenario 2: not ready path
    app_not_ready = mvp_api.create_app(service=_Service(ready=False, engine=_OkEngine()))
    client_not_ready = testclient.TestClient(app_not_ready)
    r2 = client_not_ready.post("/v1/generate", json={"prompt": "hello"})
    assert r2.status_code == 503

    # Scenario 3: timeout path
    monkeypatch.setattr(mvp_api, "REQUEST_TIMEOUT_S", 0.0001)
    app_timeout = mvp_api.create_app(service=_Service(ready=True, engine=_TimeoutEngine()))
    client_timeout = testclient.TestClient(app_timeout)
    r3 = client_timeout.post("/v1/generate", json={"prompt": "hello"})
    assert r3.status_code == 504

    # Scenario 4: generation failure path
    monkeypatch.setattr(mvp_api, "REQUEST_TIMEOUT_S", 30.0)
    app_fail = mvp_api.create_app(service=_Service(ready=True, engine=_FailEngine()))
    client_fail = testclient.TestClient(app_fail)
    r4 = client_fail.post("/v1/generate", json={"prompt": "hello"})
    assert r4.status_code == 500

    events = _collect_api_events(caplog)
    assert events, "No structured API logs captured"

    # Ensure all captured API events are valid JSON schema-wise
    for event in events:
        _assert_schema(event)

    observed = {event["event"] for event in events}
    required = {"generate_ok", "generate_not_ready", "generate_timeout", "generate_error", "request_complete"}
    assert required.issubset(observed), f"Missing required events. observed={observed}"
