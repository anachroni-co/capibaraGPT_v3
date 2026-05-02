"""
MVP API Tests - Unit tests for minimum viable product API.

This module provides tests for the MVP API endpoints, validating
request handling, response formatting, and error conditions.

Author: Skydesk International Dev Team.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

pytest.importorskip("httpx")  # starlette.testclient requires httpx at runtime
testclient = pytest.importorskip("fastapi.testclient")

from capibara.mvp_api import create_app


@dataclass
class _Result:
    text: str = "ok"
    prompt_tokens: int = 3
    tokens_generated: int = 2


class _FakeEngine:
    async def generate(self, prompt: str, **kwargs):
        return _Result(text=f"echo:{prompt}")


class _FakeService:
    def __init__(self, ready: bool = True):
        self._ready = ready
        self._ready_reason = "ready" if ready else "model_not_loaded"
        self.model_version = "test-model"
        self.engine = _FakeEngine()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def ready_reason(self) -> str:
        return self._ready_reason


def test_health_live():
    app = create_app(service=_FakeService(ready=True))
    client = testclient.TestClient(app)
    res = client.get("/health/live")
    assert res.status_code == 200
    assert res.json()["status"] == "alive"
    assert isinstance(res.json()["request_id"], str) and res.json()["request_id"]
    assert isinstance(res.headers.get("X-Request-Id"), str)


def test_health_ready_when_not_ready():
    app = create_app(service=_FakeService(ready=False))
    client = testclient.TestClient(app)
    res = client.get("/health/ready")
    assert res.status_code == 503
    assert isinstance(res.json()["detail"]["request_id"], str)


def test_post_v1_generate_contract():
    app = create_app(service=_FakeService(ready=True))
    client = testclient.TestClient(app)
    res = client.post(
        "/v1/generate",
        json={"prompt": "hello", "max_new_tokens": 8, "temperature": 0.2},
    )
    assert res.status_code == 200
    payload = res.json()
    assert set(payload.keys()) == {
        "text",
        "usage",
        "latency_ms",
        "model_version",
        "request_id",
    }
    assert payload["text"] == "echo:hello"
    assert payload["usage"]["input_tokens"] == 3
    assert payload["usage"]["output_tokens"] == 2


def test_generate_not_ready_includes_request_id():
    app = create_app(service=_FakeService(ready=False))
    client = testclient.TestClient(app)
    res = client.post("/v1/generate", json={"prompt": "hello"})
    assert res.status_code == 503
    detail = res.json()["detail"]
    assert detail["error"] == "service_not_ready"
    assert isinstance(detail["request_id"], str) and detail["request_id"]


def test_generate_timeout_includes_request_id(monkeypatch):
    class _SlowEngine:
        async def generate(self, prompt: str, **kwargs):
            await asyncio.sleep(0.01)
            return _Result(text=prompt)

    class _SlowService(_FakeService):
        def __init__(self):
            super().__init__(ready=True)
            self.engine = _SlowEngine()

    import capibara.mvp_api as mvp_api

    monkeypatch.setattr(mvp_api, "REQUEST_TIMEOUT_S", 0.0001)
    app = mvp_api.create_app(service=_SlowService())
    client = testclient.TestClient(app)
    res = client.post("/v1/generate", json={"prompt": "hello"})
    assert res.status_code == 504
    detail = res.json()["detail"]
    assert detail["error"] == "generation_timeout"
    assert isinstance(detail["request_id"], str) and detail["request_id"]


def test_trace_id_header_propagation():
    app = create_app(service=_FakeService(ready=True))
    client = testclient.TestClient(app)
    req_id = "req-abc"
    trace_id = "trace-xyz"
    res = client.post(
        "/v1/generate",
        json={"prompt": "hello"},
        headers={"X-Request-Id": req_id, "X-Trace-Id": trace_id},
    )
    assert res.status_code == 200
    assert res.headers.get("X-Request-Id") == req_id
    assert res.headers.get("X-Trace-Id") == trace_id
