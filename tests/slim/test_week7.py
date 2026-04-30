"""Week 7 tests — quantization, auth, rate limiting, CLAUDE.md."""
from __future__ import annotations

import os
import pytest
import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

needs_torch = pytest.mark.skipif(not _TORCH, reason="torch not installed")

from config.slim_loader import load_config


@pytest.fixture(autouse=True)
def _clear_config():
    load_config.cache_clear()
    yield
    load_config.cache_clear()


# ---------------------------------------------------------------------------
# T7.1 — INT8 Quantization
# ---------------------------------------------------------------------------

def test_quantization_available_flag():
    from inference.quantization import QUANTIZATION_AVAILABLE
    assert QUANTIZATION_AVAILABLE is True


def test_weight_quantizer_factory_returns_instance():
    from inference.quantization import create_weight_quantizer, WeightQuantizer
    q = create_weight_quantizer()
    assert isinstance(q, WeightQuantizer)


def test_quantize_per_channel_symmetric_shape():
    from inference.quantization import create_weight_quantizer
    q = create_weight_quantizer()
    w = np.random.randn(16, 32).astype(np.float32)
    q_w, scales = q.quantize_per_channel_symmetric(w)
    assert q_w.shape == (16, 32)
    assert q_w.dtype == np.int8
    assert scales.shape == (16,)
    assert scales.dtype == np.float16


def test_quantize_per_channel_round_trip():
    from inference.quantization import create_weight_quantizer
    q = create_weight_quantizer()
    w = np.random.randn(64, 128).astype(np.float32)
    q_w, scales = q.quantize_per_channel_symmetric(w)
    # Dequantize
    w_dq = q_w.astype(np.float32) * scales[:, np.newaxis]
    # Max relative error should be small (INT8 quantization ~1/127)
    rel_err = np.mean(np.abs(w - w_dq)) / (np.mean(np.abs(w)) + 1e-8)
    assert rel_err < 0.05, f"round-trip relative error too high: {rel_err:.4f}"


def test_quantize_rejects_1d_array():
    from inference.quantization import create_weight_quantizer
    q = create_weight_quantizer()
    with pytest.raises(ValueError, match="2D"):
        q.quantize_per_channel_symmetric(np.ones(16))


def test_slim_quantizer_state_dict():
    from inference.quantization import SlimQuantizer
    sq = SlimQuantizer()
    state = {
        "layer.weight": np.random.randn(32, 64).astype(np.float32),
        "layer.bias": np.zeros(32, dtype=np.float32),
    }
    stats, q_dict = sq.quantize_state_dict(state)
    assert stats.quantized_tensors == 1     # only the 2-D weight
    assert stats.skipped_tensors == 1       # bias is 1-D
    assert "layer.weight.W_q" in q_dict
    assert "layer.weight.S" in q_dict
    assert "layer.bias" in q_dict


def test_slim_quantizer_stats_total():
    from inference.quantization import SlimQuantizer
    sq = SlimQuantizer()
    state = {f"w{i}": np.random.randn(8, 16).astype(np.float32) for i in range(5)}
    stats, _ = sq.quantize_state_dict(state)
    assert stats.total_tensors == 5
    assert stats.quantized_tensors == 5
    assert stats.skipped_tensors == 0


@needs_torch
def test_slim_quantizer_apply_to_model():
    from inference.quantization import SlimQuantizer
    from models.architecture import SlimConfig, SlimModel

    cfg = SlimConfig(hidden_size=32, num_layers=1, num_heads=2,
                     intermediate_size=64, vocab_size=50, max_seq_len=8)
    model = SlimModel(cfg)
    sq = SlimQuantizer()
    stats = sq.apply_to_model(model)
    assert stats.quantized_tensors > 0
    # Model should still run after quantization
    ids = torch.randint(0, 50, (1, 4))
    out = model(ids)
    assert out.shape == (1, 4, 50)


# ---------------------------------------------------------------------------
# T7.2 — Authentication
# ---------------------------------------------------------------------------

def test_auth_passthrough_when_disabled():
    """With no CAPIBARA_API_KEY set, all requests pass through."""
    os.environ.pop("CAPIBARA_API_KEY", None)
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.post("/generate", json={"input": "hello"})
    assert r.status_code == 200


def test_auth_disabled_health_always_ok():
    os.environ.pop("CAPIBARA_API_KEY", None)
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


def test_auth_rejects_without_key(monkeypatch):
    monkeypatch.setenv("CAPIBARA_API_KEY", "secret-test-key")
    monkeypatch.setenv("CAPIBARA_AUTH__ENABLED", "true")

    # Reload auth module to pick up env changes
    import importlib, app.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_auth_enabled", lambda: True)
    monkeypatch.setattr(auth_mod, "_configured_key", lambda: "secret-test-key")

    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/generate", json={"input": "hello"})
    assert r.status_code == 401


def test_auth_accepts_valid_key(monkeypatch):
    import app.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_auth_enabled", lambda: True)
    monkeypatch.setattr(auth_mod, "_configured_key", lambda: "my-key")

    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.post("/generate",
                    json={"input": "hello"},
                    headers={"Authorization": "Bearer my-key"})
    assert r.status_code == 200


def test_auth_rejects_wrong_key(monkeypatch):
    import app.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_auth_enabled", lambda: True)
    monkeypatch.setattr(auth_mod, "_configured_key", lambda: "correct-key")

    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/generate",
                    json={"input": "hello"},
                    headers={"Authorization": "Bearer wrong-key"})
    assert r.status_code == 403


def test_auth_health_bypasses_when_enabled(monkeypatch):
    import app.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_auth_enabled", lambda: True)
    monkeypatch.setattr(auth_mod, "_configured_key", lambda: "k")

    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.get("/health")   # /health is in _UNPROTECTED
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# T7.3 — Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limiter_allows_under_limit():
    from app.ratelimit import RateLimiter
    rl = RateLimiter(requests_per_minute=10)
    for _ in range(10):
        allowed, _ = rl.check("127.0.0.1")
        assert allowed


def test_rate_limiter_blocks_over_limit():
    from app.ratelimit import RateLimiter
    rl = RateLimiter(requests_per_minute=5)
    for _ in range(5):
        rl.check("10.0.0.1")
    allowed, _ = rl.check("10.0.0.1")
    assert not allowed


def test_rate_limiter_different_ips_independent():
    from app.ratelimit import RateLimiter
    rl = RateLimiter(requests_per_minute=2)
    for _ in range(2):
        rl.check("1.1.1.1")
    # First IP exhausted; second IP should still be allowed
    allowed, _ = rl.check("2.2.2.2")
    assert allowed


def test_rate_limit_middleware_passthrough_when_disabled():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.post("/generate", json={"input": "hi"})
    assert r.status_code == 200


def test_rate_limit_middleware_returns_429(monkeypatch):
    import app.ratelimit as rl_mod
    monkeypatch.setattr(rl_mod, "_rate_limit_enabled", lambda: True)

    # Inject a pre-exhausted limiter
    exhausted = rl_mod.RateLimiter(requests_per_minute=0)
    monkeypatch.setattr(rl_mod, "_limiter", exhausted)

    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/generate", json={"input": "hi"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers


# ---------------------------------------------------------------------------
# T7.4 — CLAUDE.md exists and is non-trivial
# ---------------------------------------------------------------------------

def test_claude_md_exists():
    from pathlib import Path
    p = Path(__file__).parents[2] / "CLAUDE.md"
    assert p.exists(), "CLAUDE.md not found at repo root"


def test_claude_md_covers_key_sections():
    from pathlib import Path
    text = (Path(__file__).parents[2] / "CLAUDE.md").read_text()
    for section in ("app/", "inference/", "training/", "rag/", "tests/"):
        assert section in text, f"CLAUDE.md missing section about {section}"
