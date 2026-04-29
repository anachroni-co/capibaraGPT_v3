"""Week 1 smoke tests for Capibara Slim."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from config.slim_loader import load_config, get
from inference.pipeline import SlimPipeline
from services.api_service import ApiService


client = TestClient(app)


# ---------------------------------------------------------------------------
# T1.2 — API
# ---------------------------------------------------------------------------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_generate_returns_output():
    r = client.post("/generate", json={"input": "hello world"})
    assert r.status_code == 200
    body = r.json()
    assert "output" in body
    assert "model" in body
    assert "tokens_used" in body


def test_generate_rejects_empty_input():
    r = client.post("/generate", json={"input": ""})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# T1.3 — Config
# ---------------------------------------------------------------------------

def test_config_loads():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "api" in cfg
    assert "inference" in cfg


def test_config_get_defaults():
    assert get("api", "port") == 8000
    assert get("model", "backend") in ("auto", "stub", "transformer", "mamba")
    assert get("missing_section", "missing_key", "default") == "default"


# ---------------------------------------------------------------------------
# T1.5 — Pipeline
# ---------------------------------------------------------------------------

def test_pipeline_stub_output():
    p = SlimPipeline()
    result = p.run("test input here")
    assert isinstance(result["output"], str)
    assert len(result["output"]) > 0
    assert "tokens_used" in result
    assert result["latency_ms"] >= 0.0


def test_pipeline_preserves_token_count():
    p = SlimPipeline()
    text = "one two three four five"
    r = p.run(text)
    assert r["tokens_used"] == 5


# ---------------------------------------------------------------------------
# T1.4 — Services
# ---------------------------------------------------------------------------

def test_api_service_shapes_response():
    svc = ApiService()
    result = svc.generate("ping", max_tokens=32)
    assert set(result.keys()) == {"output", "model", "tokens_used"}
