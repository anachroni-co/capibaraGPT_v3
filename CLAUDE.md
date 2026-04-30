# Capibara Slim — Codebase Guide

Production-ready hybrid Transformer + Mamba inference and training system.
This file documents conventions, module structure, and key invariants for
contributors and AI coding agents.

---

## Quick start

```bash
# Install dependencies
pip install -r requirements-slim.txt

# Run the API server
uvicorn app.main:app --reload

# Run all Slim tests
python -m pytest tests/slim/ -v
```

---

## Directory structure

```
app/          FastAPI application layer
  main.py     ASGI app — CORS, tracing middleware, auth, rate limiting
  routes.py   /health, /metrics, /generate, /generate/stream
  auth.py     Bearer-token API key middleware (opt-in via slim.yaml + env)
  ratelimit.py Sliding-window per-IP rate limiter (opt-in via slim.yaml)

config/
  slim.yaml       Default Capibara Slim configuration
  slim_loader.py  load_config() (lru_cache) + get(section, key, default)

core/
  slim_router.py  Routes requests: "mamba" | "transformer" | "tool"
  executor.py     SlimExecutor — selects backend, applies timeout

inference/
  pipeline.py     Full request pipeline: filter → route → execute → filter
  streaming.py    SlimStreamer — SSE token streaming (stub + HF backend)
  quantization.py WeightQuantizer (numpy INT8), SlimQuantizer (torch wrapper)

models/
  architecture.py SlimConfig + SlimModel (RMSNorm, RoPE, Attention, MambaBlock)
  backend.py      ModelBackend Protocol
  stub_backend.py Always-available stub (no weights needed)
  transformer_backend.py  HuggingFace causal LM wrapper
  mamba_backend.py        Mamba SSM stub (real weights not yet wired)
  registry.py     ModelRegistry — lazy singleton factory per backend

rag/
  store.py        VectorStore — BoW/SentenceTransformer cosine search
  ingestion.py    ingest_text / ingest_file / ingest_directory
  retriever.py    Retriever — min_score filter + context augmentation

services/
  model_service.py   ResponseCache wrapper around SlimPipeline
  api_service.py     Records latency/tokens/errors to utils.stats

tools/
  registry.py    ToolRegistry
  executor.py    ToolExecutor (ThreadPoolExecutor with timeout)
  detector.py    detect_tool() — regex parser for "tool: name(args)" syntax

training/
  trainer.py      SlimTrainer — AdamW + cosine LR + grad accum + mixed prec
  checkpoint.py   save/load checkpoint; latest symlink
  distributed.py  apply_gradient_checkpointing + wrap_fsdp
  evaluation.py   Evaluator — perplexity via sum cross-entropy
  train.py        CLI entry point (argparse)

utils/
  cache.py        ResponseCache — LRU OrderedDict
  logger.py       JSON structured logging + request_id ContextVar
  stats.py        _Stats — thread-safe counters; module-level record/snapshot
  tokenizer.py    SlimTokenizer — HF AutoTokenizer with whitespace fallback

data/
  loader.py       SlimDataset + create_dataloader

docker/
  Dockerfile                Multi-stage image; non-root capibara user
  docker-compose.slim.yml   Single api service

tests/slim/
  test_week1.py   Config, logging, health endpoint
  test_week2.py   Pipeline, routing, backends
  test_week3.py   Tools, safety filters, cache
  test_week4.py   Docker, hardening, timeouts
  test_week5.py   Training, checkpoints, tokenizer, data loader
  test_week6.py   Gradient checkpointing, FSDP, streaming, RAG, evaluation
  test_week7.py   Quantization, auth, rate limiting, CLAUDE.md
```

---

## Key invariants

### Torch is optional
Every module that uses PyTorch wraps its import in `try/except ImportError`
and provides a no-op stub or raises `ImportError` at construction time.
The test suite runs fully (with skips) in environments without torch.

### Config access
Always use `config.slim_loader.get(section, key, default)`.
Never read `slim.yaml` directly from application code.
`load_config()` is `@lru_cache`; call `load_config.cache_clear()` in tests.

### Backend fallback chain
`SlimExecutor._select_backend()` → router → config override → stub.
If a backend reports `is_available = False`, execution falls back to stub.
Never let a backend raise an exception visible to the HTTP layer.

### Safety filters
`InputFilter` blocks jailbreak/weapon patterns before any model call.
`OutputFilter` redacts PII (emails, phone numbers, SSNs) from model output.
Both filters are applied unconditionally when `safety.input_filter/output_filter = true`.

### Streaming
`SlimStreamer.stream()` yields `str` fragments (never empty strings).
When the input is blocked, it yields exactly one fragment: `"[blocked] ..."`.
The SSE endpoint wraps each fragment as `data: {"token": "..."}\n\n` and
terminates with `data: [DONE]\n\n`.

### Auth (opt-in)
Auth is **disabled by default**. It only activates when *both*:
1. `auth.enabled = true` in `slim.yaml` (or `CAPIBARA_AUTH__ENABLED=true`)
2. `CAPIBARA_API_KEY` env var is set

`/health` and `/metrics` are always unprotected.

### Rate limiting (opt-in)
Rate limiting is **disabled by default** (`rate_limit.enabled = false`).
Enable with `rate_limit.enabled = true`. Limit defaults to 60 req/min per IP.
`/health` and `/metrics` are always exempt.

### Quantization (opt-in)
`SlimQuantizer.apply_to_model(model)` replaces 2-D Linear weights in-place
with dequantized float32 tensors (INT8 → float32 round-trip, ~1% error).
Requires PyTorch. The pure-numpy `WeightQuantizer` has no torch dependency.

---

## Running tests

```bash
# Full slim suite
python -m pytest tests/slim/ -v

# Single week
python -m pytest tests/slim/test_week7.py -v

# With coverage
python -m pytest tests/slim/ --cov=. --cov-report=term-missing
```

Torch-dependent tests are automatically skipped when torch is not installed.

---

## Model architecture (SlimModel)

```
SlimModel
  └── embedding (vocab → hidden)
  └── blocks[0..N]
        ├── TransformerBlock  (RMSNorm → SlimAttention + RoPE → SwiGLU MLP)
        └── MambaBlock        (selective SSM: conv1d → ZOH discretization → sequential scan)
  └── norm (RMSNorm)
  └── lm_head (hidden → vocab, weight-tied to embedding)
```

Preset sizes: `1.5b` (24 layers, hidden=2048), `3b` (32 layers, hidden=2560),
`7b` (32 layers, hidden=4096). All share `vocab_size=32000`, `max_seq_len=2048`.

---

## Adding a new backend

1. Create `models/my_backend.py` implementing the `ModelBackend` Protocol
   (`name`, `is_available`, `generate(input_text, **kwargs) -> dict`).
2. Register it in `models/registry.py` inside `get_registry()`.
3. Add the backend name to the `model.backend` enum comment in `slim.yaml`.
4. Write tests in `tests/slim/test_weekN.py`.

---

## Environment variables

| Variable | Effect |
|---|---|
| `CAPIBARA_API_KEY` | API key checked by AuthMiddleware when auth is enabled |
| `CAPIBARA_MODEL__BACKEND` | Override `model.backend` from slim.yaml |
| `CAPIBARA_INFERENCE__TIMEOUT_SECONDS` | Override inference timeout |
| `CAPIBARA_LOGGING__LEVEL` | Override log level (DEBUG/INFO/WARNING) |
| `CAPIBARA_AUTH__ENABLED` | Set `"true"` to enable auth (requires API key) |
| `CAPIBARA_RATE_LIMIT__ENABLED` | Set `"true"` to enable rate limiting |
