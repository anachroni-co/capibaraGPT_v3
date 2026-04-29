"""Week 2 tests — router, registry, backends, executor (T2.1–T2.6)."""
from __future__ import annotations

import pytest
from config.slim_loader import load_config


@pytest.fixture(autouse=True)
def _clear_config_cache():
    load_config.cache_clear()
    yield
    load_config.cache_clear()


# ---------------------------------------------------------------------------
# T2.2 — SlimRouter
# ---------------------------------------------------------------------------

from core.slim_router import SlimRouter


def test_router_short_input_goes_mamba():
    r = SlimRouter()
    assert r.route("hi there") == "mamba"


def test_router_long_input_goes_transformer():
    r = SlimRouter()
    long_text = " ".join(["word"] * 200)
    assert r.route(long_text) == "transformer"


def test_router_tool_prefix():
    r = SlimRouter()
    assert r.route("tool: calculate 1+1") == "tool"
    assert r.route("TOOL: something") == "tool"


def test_router_always_transformer(monkeypatch):
    import config.slim_loader as _sl
    monkeypatch.setattr(_sl, "load_config", lambda: {
        "routing": {"strategy": "always_transformer", "mamba_threshold": 128},
        "model": {"backend": "auto", "path": ""},
    })
    r = SlimRouter()
    assert r.route("hi") == "transformer"


def test_router_always_stub(monkeypatch):
    import config.slim_loader as _sl
    monkeypatch.setattr(_sl, "load_config", lambda: {
        "routing": {"strategy": "always_stub", "mamba_threshold": 128},
        "model": {"backend": "auto", "path": ""},
    })
    r = SlimRouter()
    assert r.route("anything") == "stub"


# ---------------------------------------------------------------------------
# T2.3 — ModelRegistry
# ---------------------------------------------------------------------------

from models.registry import ModelRegistry


def test_registry_register_and_get():
    reg = ModelRegistry()
    reg.register("test", lambda: _FakeBackend("test"))
    b = reg.get("test")
    assert b.name == "test"


def test_registry_caches_instance():
    reg = ModelRegistry()
    calls = []
    def factory():
        calls.append(1)
        return _FakeBackend("cached")
    reg.register("cached", factory)
    reg.get("cached")
    reg.get("cached")
    assert len(calls) == 1


def test_registry_unknown_raises():
    reg = ModelRegistry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_registry_available_lists_registered():
    reg = ModelRegistry()
    reg.register("a", lambda: _FakeBackend("a"))
    reg.register("b", lambda: _FakeBackend("b"))
    assert set(reg.available()) == {"a", "b"}


# ---------------------------------------------------------------------------
# T2.4 / T2.5 — Backends
# ---------------------------------------------------------------------------

from models.stub_backend import StubBackend
from models.mamba_backend import MambaBackend


def test_stub_backend_always_available():
    b = StubBackend()
    assert b.is_available is True
    assert b.name == "stub"


def test_stub_backend_generate():
    b = StubBackend()
    r = b.generate("hello world")
    assert "output" in r
    assert r["tokens_used"] == 2
    assert r["model"] == "stub"


def test_mamba_backend_stub_generates():
    b = MambaBackend()
    r = b.generate("short input")
    assert "output" in r
    assert r["model"] == "mamba"


def test_transformer_backend_unavailable_without_torch():
    from models.transformer_backend import TransformerBackend
    b = TransformerBackend("models/tiny-gpt2")
    # torch absent in test env → is_available False
    if not b.is_available:
        with pytest.raises(RuntimeError):
            b.generate("test")
    else:
        # torch present: just verify output shape
        r = b.generate("hello", max_tokens=5)
        assert "output" in r


# ---------------------------------------------------------------------------
# T2.1 — SlimExecutor
# ---------------------------------------------------------------------------

from core.executor import SlimExecutor


def test_executor_returns_required_keys():
    e = SlimExecutor()
    r = e.run("hello world")
    assert {"output", "model", "tokens_used", "latency_ms", "routed_to"} <= r.keys()


def test_executor_short_routes_to_mamba():
    e = SlimExecutor()
    r = e.run("hi")
    assert r["routed_to"] == "mamba"


def test_executor_long_routes_to_transformer_or_stub():
    e = SlimExecutor()
    long_text = " ".join(["word"] * 200)
    r = e.run(long_text)
    # In no-torch environment the executor falls back to stub via TransformerBackend
    assert r["routed_to"] in ("transformer", "stub")


# ---------------------------------------------------------------------------
# T2.6 — Pipeline → executor integration
# ---------------------------------------------------------------------------

from inference.pipeline import SlimPipeline


def test_pipeline_uses_executor():
    p = SlimPipeline()
    r = p.run("test input")
    assert "output" in r
    assert "routed_to" in r


def test_pipeline_postprocess_strips_whitespace():
    p = SlimPipeline()
    r = p.run("  padded input  ")
    assert r["output"] == r["output"].strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeBackend:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_available(self) -> bool:
        return True

    def generate(self, input_text, **_):
        return {"output": "fake", "model": self._name, "tokens_used": 0}
