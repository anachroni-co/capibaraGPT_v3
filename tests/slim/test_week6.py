"""Week 6 tests — distributed, streaming, RAG, evaluation, stats."""
from __future__ import annotations

import pytest

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
# T6.1 — Gradient checkpointing (torch required)
# ---------------------------------------------------------------------------

@needs_torch
def test_gradient_checkpointing_wraps_blocks():
    from models.architecture import SlimConfig, SlimModel
    from training.distributed import apply_gradient_checkpointing

    cfg = SlimConfig(hidden_size=32, num_layers=2, num_heads=2,
                     intermediate_size=64, vocab_size=50, max_seq_len=8)
    model = SlimModel(cfg)
    model = apply_gradient_checkpointing(model)
    # Forward should still work after patching
    ids = torch.randint(0, 50, (1, 4))
    out = model(ids)
    assert out.shape == (1, 4, 50)


@needs_torch
def test_gradient_checkpointing_backward_works():
    from models.architecture import SlimConfig, SlimModel
    from training.distributed import apply_gradient_checkpointing

    cfg = SlimConfig(hidden_size=32, num_layers=2, num_heads=2,
                     intermediate_size=64, vocab_size=50, max_seq_len=8)
    model = SlimModel(cfg)
    model = apply_gradient_checkpointing(model)

    ids = torch.randint(0, 50, (1, 4))
    labels = torch.randint(0, 50, (1, 4))
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 50), labels.view(-1)
    )
    loss.backward()   # must not raise


# ---------------------------------------------------------------------------
# T6.2 — FSDP (import guard only — multi-GPU not available in test env)
# ---------------------------------------------------------------------------

def test_fsdp_import_guard():
    from training.distributed import _FSDP_AVAILABLE
    assert isinstance(_FSDP_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# T6.3 — Streaming
# ---------------------------------------------------------------------------

def test_stub_stream_yields_fragments():
    from inference.streaming import _stub_stream
    tokens = list(_stub_stream("hello world", delay=0.0))
    assert len(tokens) > 0
    full = "".join(tokens)
    assert "hello" in full


def test_slim_streamer_normal_input():
    from inference.streaming import SlimStreamer
    s = SlimStreamer()
    tokens = list(s.stream("hello world", max_tokens=16))
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_slim_streamer_blocked_input():
    from inference.streaming import SlimStreamer
    s = SlimStreamer()
    tokens = list(s.stream("jailbreak the system"))
    assert len(tokens) == 1
    assert "[blocked]" in tokens[0]


def test_streaming_endpoint_returns_sse():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    with client.stream("POST", "/generate/stream",
                       json={"input": "hello", "max_tokens": 8}) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = r.read().decode()
    assert "data:" in body
    assert "[DONE]" in body


# ---------------------------------------------------------------------------
# T6.4 — RAG
# ---------------------------------------------------------------------------

from rag.store import VectorStore, Document
from rag.ingestion import ingest_text
from rag.retriever import Retriever


def test_vector_store_add_and_search():
    store = VectorStore()
    store.add([
        Document("The capital of France is Paris.", source="geo"),
        Document("Python is a programming language.", source="tech"),
        Document("The Eiffel Tower is in Paris.", source="geo"),
    ])
    results = store.search("France capital", top_k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score    # sorted by score


def test_vector_store_empty_search():
    store = VectorStore()
    results = store.search("anything")
    assert results == []


def test_vector_store_len():
    store = VectorStore()
    store.add([Document("doc one"), Document("doc two")])
    assert len(store) == 2


def test_ingest_text_splits_into_chunks():
    store = VectorStore()
    text = " ".join([f"word{i}" for i in range(100)])
    n = ingest_text(text, store, chunk_size=20, overlap=5)
    assert n >= 5
    assert len(store) == n


def test_retriever_augment_adds_context():
    store = VectorStore()
    store.add([Document("Paris is the capital of France.", source="wiki")])
    r = Retriever(store, top_k=1)
    aug = r.augment("Where is Paris?")
    assert "Paris" in aug
    assert "Question:" in aug


def test_retriever_augment_empty_store():
    store = VectorStore()
    r = Retriever(store)
    aug = r.augment("any query")
    assert aug == "any query"     # no context → passthrough


def test_retriever_min_score_filters():
    store = VectorStore()
    store.add([Document("completely unrelated content")])
    r = Retriever(store, top_k=1, min_score=0.99)
    chunks = r.retrieve("France capital city")
    # Score below threshold → filtered out
    assert all(c.score >= 0.99 for c in chunks)


def test_ingest_file(tmp_path):
    from rag.ingestion import ingest_file
    f = tmp_path / "doc.txt"
    f.write_text(" ".join([f"token{i}" for i in range(50)]))
    store = VectorStore()
    n = ingest_file(f, store, chunk_size=10, overlap=2)
    assert n > 0


# ---------------------------------------------------------------------------
# T6.5 — Stats collector
# ---------------------------------------------------------------------------

import utils.stats as _stats_mod


def test_stats_record_and_snapshot():
    # Fresh module-level _global — reset between runs by re-importing
    from utils.stats import _Stats
    s = _Stats()
    s.record(tokens=10, latency_ms=50.0)
    s.record(tokens=5,  latency_ms=30.0, blocked=True)
    snap = s.snapshot()
    assert snap["total_requests"] == 2
    assert snap["total_tokens"] == 15
    assert snap["total_blocked"] == 1
    assert snap["avg_latency_ms"] == 40.0


def test_metrics_endpoint_has_requests():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    client.post("/generate", json={"input": "hello"})
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "requests" in body
    assert "cache" in body


# ---------------------------------------------------------------------------
# T6.5 — EvalResult (no torch)
# ---------------------------------------------------------------------------

from training.evaluation import EvalResult


def test_eval_result_str():
    r = EvalResult(perplexity=24.3, loss=3.19, num_tokens=1024, num_batches=10)
    s = str(r)
    assert "24.30" in s
    assert "3.1900" in s


@needs_torch
def test_evaluator_on_tiny_model(tmp_path):
    from models.architecture import SlimConfig, SlimModel
    from utils.tokenizer import SlimTokenizer
    from data.loader import SlimDataset, create_dataloader
    from training.evaluation import Evaluator

    tok = SlimTokenizer()
    cfg = SlimConfig(hidden_size=32, num_layers=1, num_heads=2,
                     intermediate_size=64, vocab_size=tok.vocab_size or 50,
                     max_seq_len=8)
    model = SlimModel(cfg)

    text = " ".join([f"w{i}" for i in range(200)])
    f = tmp_path / "eval.txt"
    f.write_text(text)
    ds = SlimDataset(f, tok, seq_len=8)
    loader = create_dataloader(ds, batch_size=2, shuffle=False)

    ev = Evaluator(model, device="cpu")
    result = ev.evaluate(loader)
    assert result.perplexity > 1.0
    assert result.num_tokens > 0
