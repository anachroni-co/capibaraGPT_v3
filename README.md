# Capibara Slim

JAX/Flax foundation model stack optimised for TPU v5 training.
CPU fallback (pure NumPy) available for development and CI.

---

## What this is

Capibara Slim is the production-focused branch of CapibaraGPT v3.
It keeps the full JAX/Flax training pipeline, inference engines, and
Think-Anywhere reasoning — and removes the experimental consensus systems,
federated training, blockchain audit logs, and disconnected agent frameworks.

---

## Architecture

```
User
 ↓
capibara/repl.py   (interactive REPL — history, token budget, compaction)
 ↓
app/               (FastAPI — auth, rate limiting, streaming SSE)
 ↓
inference/hybrid_inference_engine.py  (backend selection)
 ├── TPU v5  →  JAX/Flax  →  training/tpu/
 ├── GPU     →  PyTorch
 └── CPU     →  models/pretrained_backbone.py  (NumPy fallback)
 ↓
inference/verification.py  (post-generation validation)
 ↓
core/think_anywhere/   (inline reasoning tokens — GRPO training)
core/special_tokens/   (verify / plan / search / fact_check / lang / debug)
 ↓
inference/quantization/  (INT8/INT4, KV-cache quantisation — Flax layers)
 ↓
rag/  (EmbedAnything retrieval-augmented generation)
 ↓
safety/  (input/output filters)
 ↓
Response
```

---

## Directory structure

```
app/                API layer (FastAPI, auth, rate limiting, SSE streaming)
capibara/
  repl.py           Interactive REPL (history picker, token budget, compaction)
  cli.py            Minimal argparse CLI (--health, --demo, --info)
config/
  config.yaml       Centralised configuration (model, training, features)
  feature_flags.py  is_enabled() / flag_config() helpers
  config_loader.py  YAML loader with env-var override support
core/
  think_anywhere/   Think-Anywhere inline reasoning (GRPO, streaming filter)
  special_tokens/   Structured meta-token framework (search, fact_check, …)
  backends/         Backend abstraction (TPU / GPU / CPU)
  moe/              Mixture-of-Experts routing
  cot/              Chain-of-thought helpers
data/               Datasets, loaders, preprocessing
evaluation/         8-task code benchmark (exact / prefix / pass@k)
inference/
  hybrid_inference_engine.py  Main inference orchestrator
  verification.py             Post-generation output validation
  quantization/               INT8/INT4 Flax quantised layers
  engines/                    Quantised inference engine + KV cache
models/
  pretrained_backbone.py      TransformerNumpyBackbone + LlamaCppBackbone
  architecture.py             SlimModel (RMSNorm, RoPE, Mamba, Attention)
rag/
  store.py          VectorStore — numpy cosine similarity, save/load
  ingestion.py      embed_texts/embed_query, chunking, ingest_file/directory/webpage
  retriever.py      Retriever — retrieve / augment / retrieve_texts
  memory.py         MemoryExtractor — auto-ingest high-confidence responses
safety/             Input / output safety filters
scripts/
  train_and_export_gguf.py    Train → GGUF export (CPU, NumPy)
  train_lmtp_cpu.py           L-MTP training (CPU)
  train_real_cpu.py           Byte-level training (CPU)
tests/
  slim/             Slim-specific test suite (week6: RAG, week7: improvements 2-7)
  unit/             Core unit tests
training/
  tpu/              TPU v5 trainer (JAX/Flax)
  data_capture/     Auto-capture training pairs from inference
  btx_training_system.py      Branch-Train-MiX expert training
utils/              Lightweight helpers
```

---

## Quick start

### TPU v5 (JAX/Flax)

```bash
pip install -e ".[tpu]"

python -m training.tpu.tpu_v6e_trainer \
    --config config/configs_toml/production/training.toml
```

### CPU development (NumPy only)

```bash
pip install numpy gguf

# Train 2000 steps, export to GGUF, evaluate
python scripts/train_and_export_gguf.py \
    --steps 2000 --hidden 384 --n-layers 6 --n-heads 6 \
    --seq 128 --batch 4 --lr 1e-3 \
    --out models/capibara_trained.gguf
```

### Interactive REPL

```bash
pip install -e ".[dev]"
python -m capibara.repl
```

```
Capibara Slim REPL — type /help for commands, quit to exit.
you> What is JAX?
capibara> JAX is a high-performance numerical computing library…
  [tokens: +5↑ +12↓ | total=17]
you> /flags
  off  context_compaction
  off  memory_extraction
  off  quantization
  on   rag
  off  think_anywhere
you> /tokens
  [tokens: +5↑ +12↓ | total=17]
you> quit
```

### API server

```bash
pip install -e ".[api]"
uvicorn app.main:app --reload
```

---

## Feature flags

All capabilities are **off by default** and activated via `config.yaml` or
environment variables — no code changes required.

| Flag | Env var | What it does |
|---|---|---|
| `rag` | `CAPIBARA_FEATURES_RAG_ENABLED=true` | Augment prompts with retrieved context |
| `quantization` | `CAPIBARA_FEATURES_QUANTIZATION_ENABLED=true` | INT8 weight compression at load time |
| `think_anywhere` | `CAPIBARA_FEATURES_THINK_ANYWHERE_ENABLED=true` | Inline `<think>…</think>` reasoning tokens |
| `memory_extraction` | `CAPIBARA_FEATURES_MEMORY_EXTRACTION_ENABLED=true` | Auto-ingest high-confidence responses into RAG |
| `context_compaction` | `CAPIBARA_FEATURES_CONTEXT_COMPACTION_ENABLED=true` | Summarise old turns when conversation is long |

```python
from config.feature_flags import is_enabled, flag_config, all_flags

if is_enabled("rag"):
    cfg = flag_config("rag")   # {"enabled": True, "top_k": 5, "min_score": 0.3}

print(all_flags())
# {'context_compaction': False, 'memory_extraction': False,
#  'quantization': False, 'rag': True, 'think_anywhere': False}
```

Edit `config/config.yaml` to persist flags:

```yaml
features:
  rag:
    enabled: true
    top_k: 5
    min_score: 0.3
  memory_extraction:
    enabled: true
    confidence_threshold: 0.8
```

---

## RAG — retrieval-augmented generation

Three-tier embedding backend with automatic fallback:

1. **embed_anything** — Rust-native (PyO3/maturin), no PyTorch required
2. **sentence_transformers** — PyTorch-based (`all-MiniLM-L6-v2`)
3. **BoW / TF-IDF** — numpy only, always available

```python
from rag import VectorStore, Retriever, ingest_text, ingest_file, ingest_directory

store = VectorStore()

# Ingest sources
ingest_text("Capibara Slim runs on JAX and TPU v5.", store)
ingest_file("docs/manual.pdf", store)          # PDF via embed_anything
ingest_directory("docs/", store)               # recursive, all text types

# Retrieve
retriever = Retriever(store, top_k=5, min_score=0.3)
results = retriever.retrieve("How does TPU training work?")
augmented_prompt = retriever.augment("How does TPU training work?")
```

### Persistent store

```python
store.save("data/vector_store/")          # writes vectors.npy + documents.json
store = VectorStore.load("data/vector_store/")
```

### Memory extraction

Automatically ingest high-confidence model responses back into the store,
so the model learns from its own successful outputs:

```python
from rag.memory import MemoryExtractor

extractor = MemoryExtractor(store, threshold=0.8)
extractor.maybe_ingest(
    query="What is RoPE?",
    response="RoPE is rotary position embedding…",
    score=0.92,              # e.g. from a confidence estimator
)
```

Enable via feature flag to wire it into the inference pipeline automatically:

```bash
CAPIBARA_FEATURES_MEMORY_EXTRACTION_ENABLED=true python -m capibara.repl
```

---

## Post-generation verification

`inference/verification.py` validates every model output before returning it.
Pluggable checks — enable only what you need:

```python
from inference.verification import verify_output, VerificationConfig

cfg = VerificationConfig(
    non_empty=True,
    min_length=20,
    max_length=4096,
    no_repetition=True,
    no_truncation=False,   # too strict for short outputs
    coherence=True,
)
report = verify_output(model_output, cfg)

if not report.passed:
    print(report.failures)
    # ['no_repetition: excessive repetition (ratio=0.73)']
```

---

## Think-Anywhere

`core/think_anywhere/` implements the **Think-Anywhere** mechanism
([Jiang et al., 2026](https://arxiv.org/abs/2603.29957)) —
the model inserts `<thinkanywhere>` blocks at any token position,
focusing compute where generation is hardest.

| Class | Purpose |
|---|---|
| `ThinkAnywhereProcessor` | Format prompts, parse responses, strip thinking blocks |
| `ThinkAnywhereReward` | R = 0.1·R_struct + 0.9·R_correct, GRPO advantages |
| `ThinkAnywhereStreamFilter` | Real-time streaming filter — suppresses thinking tokens |

```python
from core.think_anywhere import ThinkAnywhereProcessor

proc = ThinkAnywhereProcessor()
prompt = proc.format_prompt("Write a function for edit distance.")
result = proc.parse(model_response)
print(result.clean_code)     # executable code, thinking stripped
```

---

## Special-token framework

`core/special_tokens/` provides structured meta-tokens with semantic-aware
embedding initialisation and real-time streaming filters.

| Token | Stripped? | Purpose |
|---|---|---|
| `<verify>` | yes | Self-verification before continuing |
| `<plan>` | yes | Task decomposition |
| `<uncertain>` | no | Low-confidence marker for caller |
| `<search>` | yes | Local RAG trigger |
| `<web_search>` | yes | Live internet search |
| `<fact_check>` | no | Contradiction signal |
| `<lang:XX>` | yes | Inline language switch |
| `<debug>` | yes | Error diagnosis |

---

## Training data capture

`training/data_capture/` automatically converts high-signal inference
interactions into training pairs (SFT + DPO) as the model is used.

```python
from training.data_capture import TrainingDataCapture, ConfidenceRouter

capture = TrainingDataCapture(output_dir="data/captured")
router  = ConfidenceRouter(local_fn=model.generate, capture=capture)
response = router.generate(user_prompt)   # data captured automatically
print(capture.get_stats())
```

---

## CPU benchmark (NumPy backbone)

| Metric | Value |
|---|---|
| Model | 6L / 6H / d384 — 10.9 M params |
| Steps | 2 000 |
| Corpus | 11.1 MB (.py + .md) |
| Loss | 6.30 → 3.08 nats/byte (−51 %) |
| Throughput | ~1 344 tok/s (CPU) |
| GGUF export | 44.2 MB |
| llama.cpp inference | ~570 ms/task |

---

## Requirements

| Environment | Packages |
|---|---|
| TPU training | `jax[tpu]`, `flax`, `optax` |
| GPU training | `torch`, `flash-attn`, `accelerate`, `peft` |
| RAG | `embed-anything` (Rust-native) or `sentence-transformers` |
| CPU / dev | `numpy`, `gguf`, `llama-cpp-python` (optional) |
| API server | `fastapi`, `uvicorn`, `pydantic` |

Python >= 3.9.

```bash
pip install -e ".[tpu]"    # TPU v5 (JAX/Flax)
pip install -e ".[gpu]"    # GPU (PyTorch + flash-attn)
pip install -e ".[rag]"    # EmbedAnything RAG
pip install -e ".[api]"    # FastAPI server
pip install -e ".[dev]"    # all of the above + tests
pip install numpy gguf     # CPU-only, no extras
```

---

## Tests

```bash
pytest tests/ -v
pytest tests/slim/ -v          # slim-specific: RAG + 7 improvements (94 tests)
pytest tests/ --cov=core --cov-report=term-missing
```

---

## License

Dual licensing (open + commercial). See `LICENSE`.  
© Anacroni S.Coop.Gal.

## Contact

- GitHub: `https://github.com/anachroni-co/capibaraGPT_v3`
- Website: `https://www.anachroni.co`
- Email: `info@anachroni.co`
