# Capibara Slim

JAX/Flax foundation model stack optimised for TPU v5 training.
CPU fallback (pure NumPy) available for development and CI.

---

## What this is

Capibara Slim is the production-focused branch of CapibaraGPT v3.
It keeps the full JAX/Flax training pipeline, inference engines, and
Think-Anywhere reasoning — and removes the experimental consensus systems,
federated training, blockchain audit logs, and disconnected agent frameworks.

**Removed vs main:**
- `training/consensus/` — meta-consensus, federated consensus (experimental, mock-heavy)
- `training/federated_consensus/` — distributed consensus
- `training/data_lineage/` — blockchain audit, smart contracts
- `training/cython_kernels/` — uncompiled Cython stubs
- `agents/` — agent orchestration framework (disconnected from core)
- `sub_models/` — experimental expert submodels
- `COCOMO_II/` — cost estimation model

**Kept:**
everything else — JAX/Flax training, TPU optimisations, Think-Anywhere,
special-token framework, quantised inference, data capture, RAG, API layer.

---

## Architecture

```
User
 ↓
app/  (FastAPI — auth, rate limiting, streaming SSE)
 ↓
inference/hybrid_inference_engine.py  (backend selection)
 ├── TPU v5  →  JAX/Flax  →  training/tpu/
 ├── GPU     →  PyTorch
 └── CPU     →  models/pretrained_backbone.py  (NumPy fallback)
 ↓
core/think_anywhere/   (inline reasoning tokens — GRPO training)
core/special_tokens/   (verify / plan / search / fact_check / lang / debug)
 ↓
inference/quantization/  (INT8/INT4, KV-cache quantisation — Flax layers)
 ↓
rag/  (retrieval-augmented generation)
 ↓
safety/  (input/output filters)
 ↓
Response
```

---

## Directory structure

```
app/                API layer (FastAPI, auth, rate limiting, SSE streaming)
config/             TOML configuration files
core/
  think_anywhere/   Think-Anywhere inline reasoning (GRPO, streaming filter)
  special_tokens/   Structured meta-token framework (search, fact_check, …)
  backends/         Backend abstraction (TPU / GPU / CPU)
  moe/              Mixture-of-Experts routing
  cot/              Chain-of-thought helpers
  …
data/               Datasets, loaders, preprocessing
docker/             Dockerfile + docker-compose
evaluation/         8-task code benchmark (exact / prefix / pass@k)
inference/
  hybrid_inference_engine.py  Main inference orchestrator
  quantization/               INT8/INT4 Flax quantised layers
  engines/                    Quantised inference engine + KV cache
  int8_inference.py           NumPy INT8 fallback (no JAX required)
  …
models/
  pretrained_backbone.py      TransformerNumpyBackbone + LlamaCppBackbone
  architecture.py             SlimModel (RMSNorm, RoPE, Mamba, Attention)
  …
rag/                Vector store, ingestion, retriever
safety/             Input / output safety filters
scripts/
  train_and_export_gguf.py    Train → GGUF export (CPU, NumPy)
  train_lmtp_cpu.py           L-MTP training (CPU)
  train_real_cpu.py           Byte-level training (CPU)
  train_transformer_cpu.py    Transformer training (CPU)
  create_tiny_gguf.py         Minimal GGUF for CI
services/           API services and orchestration
tests/              Unit / integration / security / benchmark tests
training/
  byte_level_training.py      Byte-level tokeniser + trainer
  jax_utils.py                JAX / NumPy compatibility shim
  tpu/                        TPU v5 trainer (JAX/Flax)
  optimizations/              TPU v4/v5 config, XLA settings
  strategies/                 Training strategies (convexity, hierarchical, …)
  data_capture/               Auto-capture training pairs from inference
  data_preprocessing/         Quality filter, deduplicator, TPU processor
  safety/                     Bias filter, legal compliance config
  monitoring_dashboard.py     Training metrics dashboard
  btx_training_system.py      Branch-Train-MiX expert training
  moe_hierarchical_router.py  Multi-tier MoE router
  …
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

### API server

```bash
pip install -e ".[dev]"
uvicorn app.main:app --reload
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
from core.think_anywhere import ThinkAnywhereProcessor, ThinkAnywhereReward

proc = ThinkAnywhereProcessor()
prompt = proc.format_prompt("Write a function for edit distance.")
result = proc.parse(model_response)
print(result.clean_code)        # executable code, thinking stripped
print(result.is_valid)          # structural validation score
```

Enable during inference:

```python
from inference.hybrid_inference_engine import InferenceConfig
config = InferenceConfig(think_anywhere_mode=True)
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
| GPU training | `torch`, `torchvision` |
| CPU / dev | `numpy`, `gguf`, `llama-cpp-python` (optional) |
| API server | `fastapi`, `uvicorn`, `pydantic` |

Python >= 3.9.

```bash
pip install -e ".[tpu]"    # TPU
pip install -e ".[gpu]"    # GPU
pip install -e ".[dev]"    # local dev + tests
pip install numpy gguf     # CPU-only, no extras
```

---

## Tests

```bash
pytest tests/ -v
pytest tests/ --cov=core --cov-report=term-missing
```

---

## Pending work

See [`BACKLOG.md`](./BACKLOG.md) for concrete pending items with scope and
exit criteria.

---

## License

Dual licensing (open + commercial). See `LICENSE`.

## Contact

- GitHub: `https://github.com/anachroni-co/capibaraGPT_v3`
- Website: `https://www.anachroni.co`
- Email: `info@anachroni.co`
