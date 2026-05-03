# CapibaraGPT v3

Experimental open-source foundation model stack for research and education.

## What this repository includes

- Multi-backend core runtime (CPU, optional GPU/TPU).
- Training modules (consensus, strategies, federated paths).
- Inference modules (including quantization/hybrid experiments).
- Data pipelines and dataset tooling.
- Optional services and integrations.
- Test and benchmark suites.

## Live Repository Metrics

Last refreshed: `2026-04-20`

### Snapshot

| Metric | Value |
|---|---:|
| Python files (repo only) | ~594 |
| Markdown files (repo only) | ~70 |
| Python LOC (repo only) | ~180,000 |
| Test files (repo only) | 38 |
| Test functions (`def test_*`, `async def test_*`) | ~466 |

### Python LOC by top-level folder (approximate)

| Folder | LOC |
|---|---:|
| `training` | 35,177 |
| `core` | 28,474 |
| `capibara` | 14,605 |
| `services` | 10,601 |
| `data` | 10,162 |
| `sub_models` | 9,616 |
| `agents` | 8,873 |
| `jax` | 8,788 |
| `utils` | 7,607 |
| `inference` | 6,208 |

```mermaid
xychart-beta
    title "Python LOC by Folder (Top 10)"
    x-axis [training, core, capibara, services, data, sub_models, agents, jax, utils, inference]
    y-axis "LOC" 0 --> 36000
    bar [35177, 28474, 14605, 10601, 10162, 9616, 8873, 8788, 7607, 6208]
```

## Current reality

This is an active research codebase, not a production-hardened product.
Some modules are fully functional, while others are still under active implementation.

Pending technical work lives in a single file:

- **[`BACKLOG.md`](./BACKLOG.md)** — concrete, verifiable pending items with scope and exit criteria.

The previous auto-generated `TODOs.md`, `TODOs_PRIORITIZED.md` and 20 per-folder `TODOs.md` files were removed on 2026-04-20 because they were low-signal regex scrapes that conflated code comments with real pending work.

## Requirements

- Python `>=3.9`
- `pip`

Optional acceleration backends:

- GPU: PyTorch + CUDA
- TPU: JAX + Flax

## Installation

```bash
git clone https://github.com/anachroni-co/capibaraGPT_v3.git
cd capibaraGPT_v3

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"
pip install -e ".[gpu]"
pip install -e ".[tpu]"
```

## Quick start

```python
from core.backends import get_backend, BackendType

backend = get_backend(BackendType.AUTO)
print(f"Using backend: {backend.name}")
```

## Run tests

```bash
pytest tests/ -v
```

Coverage:

```bash
pytest tests/ --cov=core --cov-report=term-missing
```

## Run benchmarks

```bash
python -m benchmarks run
```

## Repository layout

- `core/`: model/runtime components.
  - `core/think_anywhere/` — Think-Anywhere reasoning module (see below).
  - `core/special_tokens/` — structured meta-token framework (see below).
- `training/`: training systems and strategies.
  - `training/data_capture/` — training data capture pipeline (see below).
- `inference/`: inference engines and quantization paths.
- `data/`: datasets, processing, and loading.
- `capibara/`: specialized modules (VQ, SSM, routers, optimizations).
- `services/`: optional service-level integrations.
- `sub_models/`: specialized expert modules.
- `tests/`: unit/integration/security/benchmark tests.
- `docs/`: project documentation.

## Think-Anywhere reasoning

`core/think_anywhere/` implements the **Think-Anywhere** mechanism from
[_Think Anywhere in Code Generation_](https://arxiv.org/abs/2603.29957)
(Jiang et al., Peking University / Alibaba, 2026).

Instead of reasoning only before the output (upfront thinking), Think-Anywhere
lets the model insert `<thinkanywhere>` blocks at any token position during
generation — focusing computation where the code is hardest to write.

### Key components

| Class | File | Purpose |
|---|---|---|
| `ThinkAnywhereConfig` | `config.py` | All hyperparameters: token strings, reward weights (α = 0.1 / 0.9), GRPO settings, semantic-mix α = 0.5 |
| `ThinkAnywhereProcessor` | `token_processor.py` | Format prompts (Table 1), parse responses, validate structure, strip thinking blocks, initialize special-token embeddings (Eqs. 5–6) |
| `ThinkAnywhereReward` | `rewards.py` | Hierarchical reward R = 0.1·R\_struct + 0.9·R\_correct (Eq. 9), subprocess sandbox execution, GRPO group-normalized advantages (Eq. 7) |
| `ThinkAnywhereStreamFilter` | `streaming.py` | Real-time streaming filter: suppresses thinking blocks token-by-token without buffering the full response |

### Quick usage

```python
from core.think_anywhere import ThinkAnywhereProcessor, ThinkAnywhereReward

proc = ThinkAnywhereProcessor()

# Format a prompt for Think-Anywhere generation
prompt = proc.format_prompt("Write a function that returns the edit distance between two strings.")

# Parse a model response — extract clean code and thinking blocks
result = proc.parse(model_response)
print(result.clean_code)           # executable code, all <thinkanywhere> stripped
print(result.think_anywhere_blocks)  # list of inline reasoning fragments
print(result.is_valid)             # structural validation (R_struct)

# Compute GRPO reward for a batch of rollout responses
reward_fn = ThinkAnywhereReward()
results = reward_fn.batch(responses, test_cases=["assert f('horse','ros') == 3"])
advantages = reward_fn.group_normalized_advantages(results)
```

### Inference integration

Set `InferenceConfig.think_anywhere_mode = True` to automatically strip
thinking blocks from generated text before returning it to the caller:

```python
from inference.hybrid_inference_engine import InferenceConfig, InferenceBackend

config = InferenceConfig(
    backend=InferenceBackend.AUTO,
    think_anywhere_mode=True,   # strips <think> and <thinkanywhere> blocks
)
```

Streaming (`generate_stream`) uses `ThinkAnywhereStreamFilter` to suppress
thinking content in real time — the caller never sees reasoning tokens.

### Special-token variant (Think-Anywhere\*)

`ThinkAnywhereConfig(use_special_tokens=True)` switches to single-token
`<ta>` / `</ta>` delimiters. Call
`ThinkAnywhereProcessor.initialize_special_token_embedding()` to compute
the semantic-aware embeddings from Eqs. 5–6 before fine-tuning:

```python
e_open, e_close = proc.initialize_special_token_embedding(
    tokenizer.get_input_embeddings().weight.detach().numpy(),
    token_ids={"think": t1, "any": t2, "where": t3, "<im_start>": t4, "<im_end>": t5},
)
```

## Special-token framework

`core/special_tokens/` generalizes the Think-Anywhere pattern into a reusable
framework for any structured meta-token type. Each token has semantic-aware
embedding initialization, a real-time streaming filter, and a global registry.

### Built-in tokens

| Token | strip? | Purpose |
|---|---|---|
| `<verify>` / `</verify>` | ✅ | Self-verification: model checks its output before continuing |
| `<plan>` / `</plan>` | ✅ | Task decomposition: outline algorithm before writing code |
| `<uncertain>` / `</uncertain>` | ❌ kept | Low-confidence marker: preserved for caller post-processing |
| `<search>query</search>` | ✅ | On-demand local RAG trigger at the exact token position needed |
| `<web_search>query</web_search>` | ✅ | Real-time internet search (Brave/Serper/DuckDuckGo) |
| `<fact_check>claim</fact_check>` | ❌ kept | Contradiction/misinformation signal: surfaces to UI for verification |
| `<lang:XX>` / `</lang>` | ✅ | Inline language switch (gl/pt/es/en/…) |
| `<debug>` / `</debug>` | ✅ | Error diagnosis before writing a fix |

### Quick usage

```python
from core.special_tokens import (
    get_registry, SearchTokenHandler, LangTokenProcessor,
    WebSearchHandler, WebSearchRetriever, FactCheckHandler,
)

reg = get_registry()
print(reg.list_tokens())
# ['verify', 'plan', 'uncertain', 'search', 'lang', 'debug', 'fact_check', 'web_search']

# Strip all strippable tokens (keeps <uncertain> and <fact_check>)
clean = reg.strip_all(model_output)

# <web_search> with live internet retrieval + RAG indexing
handler = WebSearchHandler(
    retriever=WebSearchRetriever(engine="brave", api_key="…"),
    rag_store=my_rag_store,      # optional: index results for future queries
    data_logger=my_capture,      # optional: log for training data
)
output = handler.process(model_output)  # <web_search>q</web_search> → [Web: …]

# <fact_check> claim extraction and verification
fc = FactCheckHandler(verifier=my_verifier)
processed, verdicts = fc.verify(model_output)
```

### Inference integration

`InferenceConfig.strip_special_tokens=True` (default) wires the registry into
the inference engine — all strip tokens are removed from `generate()` output.
`<uncertain>` and `<fact_check>` are intentionally preserved.

```python
from inference.hybrid_inference_engine import InferenceConfig, InferenceBackend

config = InferenceConfig(
    backend=InferenceBackend.AUTO,
    think_anywhere_mode=True,     # strips <think> / <thinkanywhere>
    strip_special_tokens=True,    # strips verify/plan/search/web_search/lang/debug
)
```

### Adding a custom token

```python
from core.special_tokens import SpecialTokenConfig, register_token

register_token(SpecialTokenConfig(
    name="cite",
    open_tag="<cite>",
    close_tag="</cite>",
    seed_tokens=["source", "reference", "citation"],
    strip_from_output=False,  # keep citations in output
))
```

## Training data capture

`training/data_capture/` intercepts high-signal inference interactions and
converts them into training pairs automatically — building a self-improving
dataset as the model is used.

### Signal sources

| Source | Trigger | Pair type | File |
|---|---|---|---|
| `web_search` | `<web_search>` fired | SFT — grounded response | `web_search.jsonl` |
| `fact_check` | `<fact_check>` verified | DPO — corrected vs original | `fact_check.jsonl` |
| `api_routing` | `<uncertain>`/`<fact_check>` → external API | DPO — teacher vs student | `api_routing.jsonl` |
| `uncertain` | `<uncertain>` spans present | SFT — queued for review | `uncertain.jsonl` |

### Pipeline

```
User query
    ↓
ConfidenceRouter.generate()
    ├── local model response
    │       ↓
    │   <uncertain> or <fact_check> detected?
    │       ├── YES → route to external API (OpenRouter / Llama / etc.)
    │       │           ├── return api_response to user
    │       │           └── log (prompt, local, api) → api_routing.jsonl  [DPO pair]
    │       └── NO  → return local response
    │                   └── log uncertain spans   → uncertain.jsonl
    │
    └── <web_search> in response?
            ↓
        WebSearchHandler.process()
            ├── fetch live results (Brave / Serper / DDG)
            ├── inject [Web: snippet] inline
            ├── index into local RAG store
            └── log (query, result)              → web_search.jsonl  [SFT pair]
```

### Quick usage

```python
from training.data_capture import TrainingDataCapture, CaptureConfig, ConfidenceRouter, RouterConfig
from core.special_tokens import WebSearchHandler, WebSearchRetriever

capture = TrainingDataCapture(CaptureConfig(output_dir="data/captured"))

# Wrap your inference function
router = ConfidenceRouter(
    local_fn=my_model.generate,
    config=RouterConfig(
        sample_rate=0.1,                                    # 10% random exploration
        api_model="meta-llama/llama-3.1-8b-instruct:free", # via OpenRouter
        api_key="sk-or-…",
    ),
    capture=capture,
)

# Wire web search with RAG indexing + data logging
web_handler = WebSearchHandler(
    retriever=WebSearchRetriever(engine="brave", api_key="…"),
    rag_store=my_rag,
    data_logger=capture,
)

# Use normally — data is captured automatically
response = router.generate(user_prompt)
response = web_handler.process(response)

# Stats
print(capture.get_stats())
# {'web_search': 42, 'api_routing': 18, 'uncertain': 7, 'fact_check': 3}
```

## Limitations

- Several advanced paths still include placeholder/mock logic (see `BACKLOG.md`).
- Hardware-specific features depend on external stacks and environment.
- Performance numbers can vary significantly across machines.

## Reproduce metrics

```bash
# Python/Markdown files (excluding venv)
rg --files -g "*.py" -g "!**/.venv/**" -g "!**/venv/**" -g "!**/.git/**" | wc -l
rg --files -g "*.md" -g "!**/.venv/**" -g "!**/venv/**" -g "!**/.git/**" | wc -l

# Test functions
rg -n "^(async )?def test_" -g "*.py" -g "!**/venv/**" -g "!**/.git/**" | wc -l
```

## License

Dual licensing (open + commercial). See `LICENSE`.

## Contact

- GitHub: `https://github.com/anachroni-co/capibaraGPT_v3`
- Website: `https://www.anachroni.co`
- Email: `info@anachroni.co`
