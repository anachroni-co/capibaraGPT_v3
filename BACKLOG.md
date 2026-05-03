# Backlog - Pending Technical Work

Last updated: `2026-04-20`

This is the single source of truth for pending technical work in CapibaraGPT v3.
The previous auto-generated files (`TODOs.md`, `TODOs_PRIORITIZED.md` and 20 per-folder `TODOs.md`) were removed because they were low-signal regex scrapes of code comments: the vast majority of their ~796 "pending" items were false positives (e.g. comments containing the word "Simulate", lines inside example code in READMEs, or meta-references to the TODO files themselves).

All items below have a concrete scope, a clearly stated problem, and verifiable exit criteria. New items should follow the same format.

---

## ISSUE-001 — `training`: remove remaining TPU consensus mocks

**Labels:** `training`, `consensus`, `tpu`, `high-priority`

**Scope**

- `training/tpu/tpu_v6_consensus_optimizer.py`

**Problem**

Mock embeddings and metrics are still present in the main path.

**Exit criteria**

- Real embeddings in the main flow.
- Performance metrics based on real execution.
- A documented minimal integration test.

---

## ISSUE-002 — `training`: meta consensus still uses `mock_response`

**Labels:** `training`, `consensus`, `high-priority`

**Scope**

- `training/consensus/meta_consensus_system.py`
- `training/consensus/advance_meta_consensus_integration.py`

**Problem**

Simulated responses/metrics still exist in consensus logic.

**Exit criteria**

- Replace simulated responses with real expert inference.
- Remove `mock_metrics` and use computed metrics.
- Add a non-regression test.

---

## ISSUE-003 — `services/automation`: simulated routes in executor

**Labels:** `services`, `automation`, `n8n`, `high-priority`

**Scope**

- `services/automation/agent_executor.py`
- `services/automation/n8n_service.py`

**Problem**

Simulated execution paths still exist at runtime.

**Exit criteria**

- Real execution for standard/fallback node.
- Stable input/output contracts.
- Basic flow smoke test.

---

## ISSUE-004 — `inference`: hybrid/quantized engines with simulated sections

**Labels:** `inference`, `quantization`, `high-priority`

**Scope**

- `inference/hybrid_inference_engine.py`
- `inference/engines/advanced_quantized_engine.py`

**Problem**

Parameter loading/generation still depends on simulation.

**Exit criteria**

- Real parameter loading from checkpoint/model hub.
- Remove simulated delays/sampling from the main path.

---

## ISSUE-005 — `training/data_lineage`: split mock demo from real runtime

**Labels:** `training`, `data-lineage`, `medium-priority`

**Scope**

- `training/data_lineage/demo_traceability_system.py`
- `training/data_lineage/inference_safe_parameter_controller.py`

**Problem**

Demo paths are mixed with potentially production runtime paths.

**Exit criteria**

- Explicit and isolated demo mode.
- Real runtime without mock dependencies.

---

## ISSUE-006 — `tests`: expand coverage for `core` and `training`

**Labels:** `tests`, `maintenance`, `medium-priority`

**Scope**

- `tests/unit/`, `tests/integration/`

**Problem**

Coverage ratio is roughly one test file per 3,800 LOC and ~0.85 test functions per production module. `core/backends`, `core/routers`, `core/cot` and `training/consensus` lack basic unit tests.

**Exit criteria**

- At least 70% coverage in `core/backends`, `core/routers`, `core/cot`.
- Integration tests covering the main path of `training/consensus`.
- CI gate on a minimum coverage threshold.

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

## Resolved

- **Sanitize per-folder TODO documentation** — removed all 20 per-folder `TODOs.md`, the two global aggregators (`TODOs.md`, `TODOs_PRIORITIZED.md`) and the generator script `scripts/clean_todos.py`. Pending work now lives only in this file.
- **Restore `capibara/` directory** — the `capibara/` tree (~14,600 LOC in 43 Python files covering VQ, SSM, `mvp_api`) was restored after being removed by mistake in commit `e164e01`.

## How to add a new item

1. Give it the next free `ISSUE-NNN` identifier.
2. Include: Labels, Scope (files/paths), Problem (one or two sentences), Exit criteria (checklist of verifiable outcomes).
3. Keep the wording short and concrete — if you cannot point to a file, it is not ready to be here yet.
