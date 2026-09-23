# Backlog - Pending Technical Work

Last updated: `2026-09-23`

This is the single source of truth for pending technical work in CapibaraGPT v3.
The previous auto-generated files (`TODOs.md`, `TODOs_PRIORITIZED.md` and 20 per-folder `TODOs.md`) were removed because they were low-signal regex scrapes of code comments: the vast majority of their ~796 "pending" items were false positives (e.g. comments containing the word "Simulate", lines inside example code in READMEs, or meta-references to the TODO files themselves).

All items below have a concrete scope, a clearly stated problem, and verifiable exit criteria. New items should follow the same format.

---

## ISSUE-006 — `tests`: expand coverage for `core` and `training`

**Labels:** `tests`, `maintenance`, `medium-priority`

**Scope**

- `tests/unit/`, `tests/integration/`

**Problem**

Coverage ratio is roughly one test file per 3,800 LOC and ~0.85 test functions per production module. `core/backends`, `core/routers` and `core/cot` are still below the 70% target.

**Progress** (2026-07-01)

- CI coverage gates are live in `.github/workflows/python-app.yml`: `core/routers` + `core/cot` at 55% (`--cov-fail-under=55`) and `core/backends` at 40%.
- New `tests/unit/test_backends_utils.py` (hardware detection, device info, registry) — commit `04b3f3c`.
- Consensus integration tests added: `test_training_consensus_surface.py`, `test_meta_consensus_no_mocks.py`, `test_federated_consensus_smoke.py`. Since the 2026-07 refactor they target `training/research/consensus` (commit `3b04029`).
- `core/cot/enhanced_cot_module.py` now splits the CPU direct-call implementation from the Flax variant (`a60ba96`) and `core/routers/base.py` no longer inherits from `flax.linen.Module` (`bf4edde`). Both changes make the previously JAX-only lines reachable on CPU, so the numbers below need re-measuring.

**Progress** (2026-05-04)

Added 25 new unit tests across `test_routers_bto.py` and `test_cot_module.py` (total now 548 pass):

| Area | Before | After | Notes |
|---|---|---|---|
| `core/cot/__init__.py` | 67% | 100% | ChainOfThought, create_cot_handler |
| `core/cot/factory.py` | 33% | 100% | All three factory functions |
| `core/cot/module.py` | 85% | 87% | Pre-initialized call path |
| `core/cot/enhanced_cot_module.py` | 57% | 58% | Loop completion branch |
| `core/routers/base.py` | 35% | 69% | setup(), matmul, fallback, recovery, metrics |
| **core/cot (combined)** | ~60% | **68%** | 2% gap is JAX-specific paths (need hardware) |
| **core/routers (combined)** | ~65% | **76%** | ✓ above 70% |

**Remaining gaps**

- `core/cot` at 68% (not yet 70%) as of 2026-05-04; re-measure after the CPU/Flax split (`a60ba96`).
- `core/backends`: ~60% overall; `gpu_backend.py` (28%) and `tpu_backend.py` (27%) require GPU/TPU hardware.
- CI gates are below target (55% / 40% instead of 70%).

**Exit criteria**

- [ ] At least 70% coverage in `core/backends`, `core/routers`, `core/cot`.
- [x] Integration tests covering the main path of the consensus code (now `training/research/consensus`).
- [x] CI gate on a minimum coverage threshold.
- [ ] CI gates raised to 70% once the coverage above is reached.

---

## ISSUE-007 — `core/special_tokens`: validate TOON context parsing on 3B base model

**Labels:** `inference`, `toon`, `fine-tuning`, `quality`

**Scope**

- `core/special_tokens/search.py` — `SearchTokenHandler(use_toon=True)`
- `core/special_tokens/web_search.py` — `WebSearchHandler(use_toon=True)`
- Fine-tuning pipeline that prepares training data for the 3B model

**Problem**

TOON tabular format reduces RAG/web-search prompt tokens by ~30–40%, but
a 3B base model trained before TOON existed (pre-Nov 2025) has not seen
this format during pretraining. When context is dense (many results, long
snippets), the model may misparse tabular rows or ignore them, degrading
response quality. The `use_toon=False` fallback exists but has not been
empirically evaluated on the target model.

**Exit criteria**

- Benchmark: compare accuracy with `use_toon=True` vs `False` on a
  held-out RAG-dependent eval set (min 200 examples).
- If accuracy drops >2 points: add TOON-formatted examples to fine-tuning
  data via `training/data_capture` pipeline and re-evaluate after one
  fine-tuning run.
- If accuracy is equivalent or better: document as validated in this entry.
- Once validated, confirm `use_toon=True` as default or flip to `False`
  and defer TOON fine-tuning to a later milestone.

---

## ISSUE-008 — cleanup: leftovers after the 2026-07 pruning

**Labels:** `maintenance`, `cleanup`, `low-priority`

**Scope**

- `capibara/__init__.py`
- `services/automation/agent_executor.py`, `services/automation/n8n_service.py`
- `training/research/README.md`

**Problem**

The 2026-07 pruning (46 files/directories deleted across `core/`, `capibara/` and `training/`) left stale references behind:

- `capibara/__init__.py` still tries `from . import routers` / `from . import optimizations`; both packages were deleted (`e677e60`, `0c19f9c`), so these blocks always fall into the `except` branch.
- `agent_executor.py` still labels the standard path as `# Use standard n8n execution (simulated)` and the `n8n_service.py` module docstring still says "Some execution paths are simulated", although ISSUE-003 removed the simulation.
- `training/research/README.md` still lists "Mocks pendientes: BACKLOG ISSUE-001/002" for `consensus/`; both issues are resolved.

**Exit criteria**

- [ ] Remove the dead `routers` / `optimizations` import blocks (and their `*_AVAILABLE` flags, after checking no caller reads them).
- [ ] Update the outdated comments/docstrings in `services/automation`.
- [ ] Update the `consensus/` row in `training/research/README.md`.
- [ ] Full `pytest tests/` run passes after the pruning (no imports of deleted modules).

---

## Recent milestones (2026-04 → 2026-07)

Context for anyone picking up the backlog. These are delivered features, not pending work.

- **2026-05-02/03 — Special tokens framework** (`core/special_tokens/`): `<verify>`, `<plan>`, `<uncertain>`, `<search>`, `<lang:XX>`, `<debug>`, `<fact_check>`, `<web_search>` + registry, stream filter and tests; TOON serialization for search/web-search context (see ISSUE-007); training data capture pipeline (`training/data_capture/`).
- **2026-05-02 — Think-Anywhere** (`core/think_anywhere/`), checkpoint autoload and MoE load-balance loss.
- **2026-05-04/06 — CPU-only path**: `scripts/train_real_cpu.py`, `scripts/pipeline_cpu_test*.py`, L-MTP training on CPU and in Flax (`models/lmtp_flax.py`, `training/lmtp_flax_trainer.py`), CPU production pipeline (`inference/cpu_kv_cache.py`, `inference/int8_inference.py`, `serving/cpu_server.py`), GGUF export; mass replacement of stubs and silent `except: pass` handlers.
- **2026-05-08/10 — Legal corpus & Axion training**: legal corpus downloader (1.684B tokens, 166 shards), Axion ARM64 training (bf16, `--resume`, `--grad-checkpoint`, pmap, XLA cache, SVE2), model soup, distillation and LoRA fine-tuning, RAG + tools + MCP legal data layer, `RUNBOOK.md`, V2 design / V3 vision docs and papers.
- **2026-05-12/16 — Inference & specialties**: 3-level speculative decoding and 2-model cascade (`scripts/cascade_inference.py`, JIT bucketing), OpenAI-compatible server (`scripts/serve.py`), `scripts/sft_finetune.py`, RLM scaffold, gestoria / documentos LoRA data generators.
- **2026-07-01/17 — Codebase pruning**: CI fixes (CoT CPU/Flax split, routers as plain Python, TPU fixture), removal of dead `core/` modules, `capibara/routers`, `capibara/optimizations` and the VQ "intelligence" suite; `capibara.utils` / `capibara.prompts` shims; `spike_ssm` consolidated; experimental training code moved to `training/research/` (canonical path documented in `training/README.md`); `data_lineage`, `cython_kernels`, `unified_trainer` and orphan routers/bridges deleted; fix for the depthwise conv kernel in `FlaxMambaBlock`.

## Resolved

- **Sanitize per-folder TODO documentation** — removed all 20 per-folder `TODOs.md`, the two global aggregators (`TODOs.md`, `TODOs_PRIORITIZED.md`) and the generator script `scripts/clean_todos.py`. Pending work now lives only in this file.
- **Restore `capibara/` directory** — the `capibara/` tree (~14,600 LOC in 43 Python files covering VQ, SSM, `mvp_api`) was restored after being removed by mistake in commit `e164e01`.
- **ISSUE-001 — `training`: remove remaining TPU consensus mocks** (2026-04-22, PR #124, `6a77c3f`) — random mock embeddings replaced by deterministic SHA256-based embeddings; metrics derived from real execution; CPU fallback kept. Test: `tests/integration/test_tpu_v6_consensus_optimizer.py`. Since 2026-07 the optimizer imports from `training.research.consensus` (`285200f`).
- **ISSUE-002 — `training`: meta consensus still uses `mock_response`** (2026-04-23/29, PR #128, `4c2bac7`, `d01fc4f`) — `mock_response` / `mock_metrics` replaced by real downstream calls or explicit failure markers; hardcoded model paths moved to `MetaConsensusConfig`; bias and safety scores computed by heuristics instead of constants. Test: `tests/integration/test_meta_consensus_no_mocks.py`. The code now lives in `training/research/consensus/` (research, not canonical path).
- **ISSUE-003 — `services/automation`: simulated routes in executor** (2026-04-23, PR #129, `22b7146`) — real `set` / `webhook` / `httpRequest` handlers (aiohttp), unknown node types return `status="unsupported"`, n8n service posts to the real `/api/v1/workflows/{id}/execute`. Test: `tests/integration/test_automation_no_simulation.py` (8/8). Leftover comments tracked in ISSUE-008.
- **ISSUE-004 — `inference`: hybrid/quantized engines with simulated sections** (2026-04-23/29, PR #130, `df8ae03`, `6008114`) — real parameter loading (pickle/msgpack/orbax, `.npz`/`.safetensors`/`.pkl`) with no synthetic fallback; real multinomial sampling via `jax.random.categorical`; fake `asyncio.sleep` delays and hardcoded layer count removed. Test: `tests/integration/test_engines_no_mock_params.py` (8/8).
- **ISSUE-005 — `training/data_lineage`: split mock demo from real runtime** (2026-04-24, PR #131; `617b34b`) — demo isolated behind `CAPIBARA_DATA_LINEAGE_DEMO=1` and bare JAX import guarded. Superseded on 2026-07-17: the whole `training/data_lineage/` directory and its isolated test were deleted in the pruning (recoverable via git history).

## How to add a new item

1. Give it the next free `ISSUE-NNN` identifier.
2. Include: Labels, Scope (files/paths), Problem (one or two sentences), Exit criteria (checklist of verifiable outcomes).
3. Keep the wording short and concrete — if you cannot point to a file, it is not ready to be here yet.
