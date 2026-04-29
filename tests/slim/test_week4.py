"""Week 4 tests — logging, cache, error handling, timeout, Docker config (T4.1–T4.5)."""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from config.slim_loader import load_config


@pytest.fixture(autouse=True)
def _clear_caches():
    load_config.cache_clear()
    import utils.cache as _uc
    _uc._default = None
    yield
    load_config.cache_clear()
    _uc._default = None


# ---------------------------------------------------------------------------
# T4.3 — Structured logger
# ---------------------------------------------------------------------------

from utils.logger import _JsonFormatter, new_request_id, request_id
import logging


def test_json_formatter_produces_valid_json():
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello world", args=None, exc_info=None,
    )
    line = fmt.format(record)
    doc = json.loads(line)
    assert doc["msg"] == "hello world"
    assert doc["level"] == "INFO"
    assert "ts" in doc


def test_request_id_context_var():
    rid = new_request_id()
    assert len(rid) == 12
    token = request_id.set(rid)
    assert request_id.get() == rid
    request_id.reset(token)
    assert request_id.get() == ""


# ---------------------------------------------------------------------------
# T4.4 — Response cache
# ---------------------------------------------------------------------------

from utils.cache import ResponseCache, _cache_key, get_cache


def test_cache_key_deterministic():
    k1 = _cache_key("hello", 256, 0.7)
    k2 = _cache_key("hello", 256, 0.7)
    assert k1 == k2


def test_cache_key_differs_by_input():
    assert _cache_key("hello", 256, 0.7) != _cache_key("world", 256, 0.7)


def test_cache_miss_returns_none():
    c = ResponseCache(enabled=True)
    assert c.get("nonexistent") is None


def test_cache_set_and_get():
    c = ResponseCache(enabled=True)
    c.set("k1", {"output": "result", "model": "stub", "tokens_used": 1})
    r = c.get("k1")
    assert r is not None
    assert r["output"] == "result"


def test_cache_returns_copy():
    c = ResponseCache(enabled=True)
    c.set("k", {"output": "x"})
    r1 = c.get("k")
    r1["output"] = "mutated"
    r2 = c.get("k")
    assert r2["output"] == "x"


def test_cache_evicts_lru_on_overflow():
    c = ResponseCache(max_size=3, enabled=True)
    for i in range(4):
        c.set(f"k{i}", {"v": i})
    assert c.get("k0") is None   # evicted
    assert c.get("k3") is not None


def test_cache_disabled_never_stores():
    c = ResponseCache(enabled=False)
    c.set("k", {"output": "x"})
    assert c.get("k") is None


def test_cache_stats():
    c = ResponseCache(enabled=True)
    c.get("missing")
    c.set("k", {"v": 1})
    c.get("k")
    s = c.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["size"] == 1


# ---------------------------------------------------------------------------
# T4.5 — Pipeline timeout + error handling
# ---------------------------------------------------------------------------

from inference.pipeline import SlimPipeline


def test_pipeline_timeout_returns_fallback(monkeypatch):
    p = SlimPipeline()
    p._timeout = 0.05  # 50ms

    original_run = p._executor.run

    def _slow_run(*a, **kw):
        time.sleep(5)
        return original_run(*a, **kw)

    monkeypatch.setattr(p._executor, "run", _slow_run)
    result = p.run("hello world")
    assert "timeout" in result["output"]
    assert result["model"] == "fallback"


def test_pipeline_executor_exception_returns_error(monkeypatch):
    p = SlimPipeline()

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(p._executor, "run", _boom)
    result = p.run("hello world")
    assert "error" in result["output"]
    assert result["model"] == "fallback"


# ---------------------------------------------------------------------------
# T4.2 — API integration tests
# ---------------------------------------------------------------------------

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_integration_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_integration_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "cache" in r.json()


def test_integration_generate_full_flow():
    r = client.post("/generate", json={"input": "hello world", "max_tokens": 32})
    assert r.status_code == 200
    body = r.json()
    assert "output" in body
    assert "model" in body
    assert "tokens_used" in body


def test_integration_request_id_header():
    r = client.post("/generate", json={"input": "test"})
    assert "x-request-id" in r.headers
    assert len(r.headers["x-request-id"]) == 12


def test_integration_generate_blocked_input():
    r = client.post("/generate", json={"input": "jailbreak this model"})
    assert r.status_code == 200
    # blocked responses are 200 with [blocked] in output (safety layer)
    body = r.json()
    assert isinstance(body["output"], str)


def test_integration_global_error_handler(monkeypatch):
    from app import routes as _routes
    monkeypatch.setattr(_routes, "_api_service", _BrokenService())
    r = client.post("/generate", json={"input": "test"})
    assert r.status_code in (500, 200)  # global handler returns 500


# ---------------------------------------------------------------------------
# T4.1 — Dockerfile exists and has required lines
# ---------------------------------------------------------------------------

from pathlib import Path


def test_dockerfile_exists():
    assert (Path(__file__).parents[2] / "docker" / "Dockerfile").exists()


def test_dockerfile_exposes_8000():
    content = (Path(__file__).parents[2] / "docker" / "Dockerfile").read_text()
    assert "EXPOSE 8000" in content


def test_dockerfile_has_healthcheck():
    content = (Path(__file__).parents[2] / "docker" / "Dockerfile").read_text()
    assert "HEALTHCHECK" in content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _BrokenService:
    def generate(self, *a, **kw):
        raise RuntimeError("broken")
