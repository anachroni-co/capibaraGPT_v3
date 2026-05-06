# Backlog - Pending Technical Work

Last updated: `2026-04-27`

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

## BACKLOG-011 — `core/kernels`: Pallas kernels for the MoE hot path

**Labels:** `core`, `model_factory`, `kernels`, `pallas`, `performance`

**Scope**

- `core/kernels/` (new package: `rmsnorm_pallas.py`, `rope_pallas.py`, `moe_dispatch_pallas.py`, `swiglu_pallas.py`).
- `core/model_factory/blocks.py` — RMSNorm, RoPE, `SparseMoEBlock`, `SwiGLUExpert` opt-in switch behind a feature flag (`ModelConfig.use_pallas_kernels: bool = False`).
- `tests/kernel_bench/` (consumed from BACKLOG-012) for acceptance gating.

**Problem**

The MoE hot path in `blocks.py` is implemented in pure Flax with `jnp.einsum` + per-token `w_gate[idx_k]` gathers (lines 283–355). On GPU this is 1.5–3× slower than a fused Pallas kernel that does grouped matmul with token-major routing. RMSNorm, RoPE and SwiGLU are simpler fused-kernel candidates.

The agentic generation pipeline from the GEAK paper (AMD, *2507.23194*) is **NOT** imported as a runtime dependency: kernels are committed statically under `core/kernels/`, generated offline in a separate `tools/kernel_agent/` scratch space (out of repo until it is mature). What we keep from GEAK here is the *acceptance contract* (correctness + speedup gate) and the modular layout that makes a future agent loop trivial to bolt on.

**Exit criteria**

- Each kernel has a Flax reference and a Pallas implementation under `core/kernels/<name>_pallas.py`.
- `ModelConfig.use_pallas_kernels=True` switches `blocks.py` to the Pallas path; the default stays `False` so CPU smoke runs are unaffected.
- Each kernel is accepted only if it passes BACKLOG-012's harness with: numerical equivalence (`rtol=1e-4`, `atol=1e-5`), and median latency ≥ `1.2×` of the reference on the target accelerator.
- A new section in the smoke recipe (`configs/smoke.toml`) exercises the Pallas path under CPU emulation when available; the test must skip gracefully if the backend is not Pallas-capable.

---

## BACKLOG-012 — `tests/kernel_bench`: numerical + latency harness for custom kernels

**Labels:** `tests`, `benchmark`, `kernels`, `tooling`

**Scope**

- `tests/kernel_bench/__init__.py`
- `tests/kernel_bench/bench_runner.py` — common harness (call/exec/speedup metrics, RNG seeding, warmup, `block_until_ready`, summary printer).
- `tests/kernel_bench/kernels/` — one file per candidate kernel: `rmsnorm.py`, `rope.py`, `moe_dispatch.py`, `swiglu.py`. Each file declares `reference_fn`, `candidate_fn`, `inputs_fn(seed)`, `tolerances`, `warmup`, `repeats`.
- `tests/kernel_bench/test_kernel_bench.py` — pytest entry point: parametrizes over the registry and asserts call accuracy + exec accuracy. Speedup is **measured and printed** but not asserted by default (gating happens in BACKLOG-011's acceptance, not in CI's smoke).
- `pyproject.toml` — new pytest marker `kernel_bench` so CI can run it on accelerator-equipped runners only.

**Problem**

Right now there is no neutral place to compare a candidate kernel against its Flax reference under the same JAX runtime. BACKLOG-011 cannot land safely without one: kernel rewrites are exactly the kind of change where a partially-correct implementation passes a smoke test and silently corrupts training (Anexo B of the GEAK paper makes this point explicit — "strong test suites are non-negotiable; EvalPlus drops false-pass by 28.9 pp").

**Exit criteria**

- Three metrics per registered kernel, all logged to stdout in a single table per pytest run:
  - **Call accuracy**: candidate compiles and runs on the configured backend without error.
  - **Exec accuracy**: candidate output matches the reference within the declared `(atol, rtol)`. Tested on at least 3 input shapes per kernel (small / medium / large).
  - **Speedup**: median of `repeats=11` runs (after `warmup=3`) of `reference_latency / candidate_latency`, computed only when exec accuracy passes.
- The harness is **kernel-agnostic**: adding a new kernel is a single new file in `tests/kernel_bench/kernels/` plus a registry import.
- `pytest tests/kernel_bench/` finishes in under 30 s on CPU (Flax-only path) and produces the table even when no Pallas kernel is registered yet (so the harness can be merged before BACKLOG-011 starts).
- Empty registry is a valid state and emits a single line `[kernel_bench] no kernels registered yet (BACKLOG-011)`.
- Sentinel: candidates that fail exec accuracy must NOT report a speedup number (avoid the "fast wrong code" trap from the paper's Anexo B).
- AST + smoke check in the commit script: AST parses, harness imports, empty-registry run is green.

---

## BACKLOG-013 — `core/trainer/recovery`: Reflexion-style recovery hook

**Labels:** `core`, `trainer`, `robustness`, `low-priority-hardening`

**Scope**

- `core/trainer/recovery.py` (new module).
- `core/trainer/__init__.py` — re-export `RecoveryHook`, `RecoveryAction`, `RecoveryConfig`.
- `scripts/train.py` — wire the hook as `on_step=` when `--enable-recovery` is passed.
- `tests/unit/test_trainer_recovery.py` — unit tests for each detector and action.

**Problem**

`Trainer.fit` already emits `StepMetrics(loss, ce_loss, aux_loss, grad_norm, lr, …)` on every step but does nothing if a metric goes pathological (NaN/Inf, loss explosion, aux_loss spike). On a long pretraining run this is the difference between losing a single batch and losing a 12-hour run.

GEAK's Reflexion module (paper section 4.2) reads the error trace, classifies the failure, and proposes a fix. We adapt it for training: the "error trace" is the `StepMetrics` stream; the "fix" is a `RecoveryAction`.

**Exit criteria**

- `RecoveryConfig(nan_inf_action='skip', explosion_factor=5.0, explosion_window=20, aux_band_sigma=4.0, lr_halving_after=3, max_consecutive_skips=10)`.
- Detectors implemented: NaN/Inf in `loss` or `grad_norm`; `loss > explosion_factor * median(last explosion_window)`; `aux_loss > mean + aux_band_sigma * std` of trailing window.
- Actions implemented: `skip` (return without applying gradients — requires a small refactor in `Trainer.fit` to consult the hook before commit), `halve_lr` (mutate the schedule scalar via a wrapper), `dump_last_ckpt`, `abort` (raise with a one-paragraph postmortem of the trailing window).
- Unit tests cover each detector with synthetic `StepMetrics` streams and assert the chosen action.
- Disabled by default; `--enable-recovery` in `scripts/train.py` opts in.

---

## BACKLOG-014 — `training/distill_tools`: strong-to-weak distillation through tool interfaces (exploratory)

**Labels:** `training`, `consensus`, `distillation`, `exploratory`, `low-priority`

**Status:** *Exploratory — no branch yet. Promote to a normal `BACKLOG-` item only after a design doc is written and reviewed.*

**Scope (sketch)**

- `training/distill_tools/` (new package, not yet created): versioned, reusable artefacts produced by a teacher LLM — data filters, validators, prompt formatters, label cleaners — that the distillation pipeline consumes when training the CapibaraMoEModel student.
- Hook into `training/consensus/` and `training/unified_trainer.py` (already documented as a "consensus-distillation seed" in BACKLOG-008's resolved doc commit).

**Problem / hypothesis**

The paper *El Agente Forjador* (arXiv 2604.14609, April 2026) shows that artefacts forged by a stronger model and reused by a weaker one act as a knowledge-transfer channel: weaker models gained up to **+16.5 pp** task accuracy by composing well-tested tools forged by Claude Opus 4.6. Translating the analogy to pretraining: a stronger LLM (teacher) can synthesize **versioned, deterministic tools** — not just labels — that the student's training loop consumes (filters that drop near-duplicate spans, prompt formatters for instruction-tuning slices, validators that reject malformed records before they hit the optimizer).

This is *not* token-level KD nor logit distillation. It is "interface distillation": the teacher's contribution is committed as inspectable Python code with tests, not as ephemeral logits.

**Open questions to resolve before promoting**

- Concrete first tool to forge: data dedup filter? prompt formatter for SFT? safety classifier?
- Versioning + test contract for forged tools (does each tool ship with a `golden.jsonl` regression set?).
- Cost/benefit vs. plain logit-KD for our scale.
- Whether `training/consensus/` is the right home or whether `training/distill_tools/` should live next to it.

**Exit criteria for promotion (NOT for implementation)**

- A `docs/design/BACKLOG-014.md` answering the four open questions above.
- One concrete forged tool prototyped end-to-end in a scratch branch.
- Measured delta (any metric: dataset quality, downstream eval, training stability) vs. the no-tool baseline.

Until those exist, this entry is here to remember the idea — not to schedule it.

**Adjacent idea (parked here, not promoted):** *task-aware MoE gating* from EEGMamba (arXiv 2407.20254, §2.4.2). The router conditions on a task-id token in addition to the input — only meaningful when we move from pure pretraining to multi-task SFT. Re-evaluate when the SFT/instruction phase lands; the implementation cost is low (one extra `task_emb` channel into the router) but the design only pays off if we have several disjoint task streams. Until then, see BACKLOG-015 for the *task-agnostic* slice we are taking from the same paper.

---

## BACKLOG-015 — `core/model_factory`: Universal Expert in `SparseMoEBlock`

**Labels:** `core`, `model_factory`, `moe`, `routing`, `stability`

**Scope**

- `core/model_factory/blocks.py` — `SparseMoEBlock` (lines 283–355). Add a single, always-on "universal" expert in parallel to the top-k specialists.
- `core/model_factory/config.py` — `ModelConfig.use_universal_expert: bool = False`, `ModelConfig.universal_expert_floor: float = 0.0` (lower bound on ω if we want to keep a permanent floor; 0.0 means pure `1 - max(gate)`).
- `tests/unit/test_blocks_universal_expert.py` — Layer-1 tests on the routing weight contract (no JAX-on-accelerator needed, just the math).

**Problem**

The TopK router in `SparseMoEBlock` is prone to *expert collapse* in the first ~5–10 k steps of pretraining: a handful of specialists hoard the routing mass while the rest never see meaningful gradient. The existing `aux_loss` mitigates this in the long run but does nothing for the first few thousand steps, exactly the window where rare-token training signal is most fragile.

EEGMamba (Yang et al., arXiv 2407.20254, §2.4.2) introduces a parallel **universal expert** that sees every token with weight `ω = 1 − max(gate)`. The intuition is that when the router is unconfident the universal expert dominates (ω → 1), which both stabilises the loss and gives every token a non-zero gradient through a shared parameter — independent of which specialist won the top-k. The paper is on EEG, but the routing component is task-agnostic and lifts cleanly into our autoregressive MoE block.

This entry tracks **only** the universal expert. The companion idea — task-aware gating — is parked under BACKLOG-014's "Adjacent idea" note because it only earns its keep in a multi-task SFT phase we have not started yet.

**Design (concise)**

In `SparseMoEBlock.__call__`, after computing `gate_softmax` and `topk_weights`:

    y_topk = sum_k topk_weights[..., k] * E_k(x)               # existing path
    if cfg.use_universal_expert:
        omega = 1.0 - jnp.max(gate_softmax, axis=-1, keepdims=True)
        omega = jnp.maximum(omega, cfg.universal_expert_floor)  # optional floor
        y = y_topk + omega * E_universal(x)
    else:
        y = y_topk

`E_universal` is one `SwiGLUExpert` instance (same shape as the specialists). Parameter count grows by `1 / num_experts` of the MoE block — for `num_experts=8` that is +12.5 % MoE FLOPs / params, ~+10–15 % end-to-end on a Mixtral-style stack.

**Exit criteria**

- Feature flag `use_universal_expert` defaults to `False`; existing smoke recipe is unaffected when the flag is off.
- With the flag on, smoke training (1 step, CPU) runs without NaN/Inf; `aux_loss` stays within the same band as the baseline run on the synthetic dataset.
- Unit tests assert: (1) `ω` is in `[universal_expert_floor, 1.0]` for every token; (2) when the gate is one-hot, `ω == universal_expert_floor`; (3) when the gate is uniform across N experts, `ω == 1 − 1/N` (modulo the floor); (4) the parameter tree gains exactly one `E_universal` subtree when the flag is on and zero when off.
- A short note in `commit_issue_backlog_015.sh` runs an AST sentinel: the routing line `omega = 1.0 - jnp.max(gate_softmax, ...)` must exist verbatim when the flag's branch is reached, so a future refactor cannot quietly turn `ω` into a constant.
- Eval ppl on the held-out smoke split is no worse than the baseline (ppl ratio ≤ 1.02 over the same step budget). The +10–15 % FLOPs cost is acceptable as long as ppl is at parity or better; if ppl regresses, the flag stays off and the entry is reopened.

---

## BACKLOG-016 — `tools/audit`: inventory of `layers/` and `sub_models/`

**Labels:** `tooling`, `documentation`, `hygiene`, `low-priority`

**Scope**

- `tools/audit/inventory_layers_submodels.py` (new script) — walks `layers/**/*.py` and `sub_models/**/*.py`, classifies each file as `alive` / `referenced` / `dead`, and writes a JSON manifest.
- `docs/sub_models_inventory.json` — machine-readable manifest produced by the script.
- `docs/sub_models_inventory.md` — human-readable rendering of the manifest, sorted by status, with a "Notes" column flagging broken / duplicate / silent-fallback / misleading-name entries.
- `tests/unit/test_inventory_consistency.py` — drift detector: re-runs the audit and asserts the manifest is up to date (so a future PR cannot quietly add a new dead module).

**Problem**

`layers/` (≈ 20 modules, including the subpackages `abstract_reasoning/`, `pasive/` [sic], `sparsity/`) and `sub_models/` (≈ 30 modules across `capibaras/`, `experimental/`, `hybrid/`, `mamba/`, `semiotic/`, `vision/`) accumulated as the project evolved. A manual review on 2026-04-26 surfaced four classes of problem:

1. **Broken code committed as production** — `sub_models/mamba/mamba_module.py` calls `delta.unsqueeze(-1)` (PyTorch syntax) inside `_selective_scan`; runtime would fail on first call.
2. **Misleading names** — `layers/sparsity/mixture_of_rookies.py` is a vanilla top-k MoE; it does **not** implement Pinto/Arnau/González (arXiv 2202.04990).
3. **Duplicates** — `SSM_TPU` exists triplicated (`capibara/ssm/ssm_tpu.py`, `sub_models/SSM_TPU.py`, partial restatement in `layers/ssm_hybrid_layers.py` via dead imports). `spike_ssm.py` exists in both `capibara/ssm/` and `sub_models/experimental/`.
4. **Silent fallbacks** — `sub_models/__init__.py` wraps every submodule import in `try / except Exception → flag = False`, exactly the pattern banned by `CONTRIBUTING.md` §1.

Without an inventory we cannot make any of the BACKLOG-017 cleanup decisions safely. This entry is the cheap prerequisite that turns "should we delete this?" into "delete files X, Y, Z; rename W".

**Exit criteria**

- `python tools/audit/inventory_layers_submodels.py` runs to completion in under 30 s on the full repo from a clean checkout and writes both the JSON and the markdown.
- Each entry in the manifest has: `path`, `defined_symbols`, `external_importers` (importers from outside its own package), `classification` (`alive` / `referenced` / `dead`), `notes` (free-text flags).
- Classification rules are explicit in `docs/sub_models_inventory.md`:
  - `alive` — imported from `core/`, `scripts/`, or `training/` (productive paths).
  - `referenced` — imported only from `tests/`, or only from other `layers/` / `sub_models/` modules (internal coupling).
  - `dead` — no importer outside its own package `__init__.py`.
- The audit on current `main` produces a manifest with every `.py` file under `layers/` and `sub_models/` classified — no `unknown` entries.
- The `Notes` column flags each of the four problem classes above with a stable token (`broken`, `misleading-name`, `duplicate-of:<path>`, `silent-fallback`).
- `tests/unit/test_inventory_consistency.py` re-runs the script in-process and asserts the on-disk manifest is byte-identical (CI gate against drift).

---

## BACKLOG-017 — Cleanup of dead code, duplicates, and misleading names in `layers/` + `sub_models/`

**Labels:** `cleanup`, `hygiene`, `breaking-change`

**Depends on:** BACKLOG-016 (the manifest is the input to this work).

**Scope**

Concrete actions, each driven by the BACKLOG-016 manifest. The branch ships them as one PR so that the manifest goes from "lots of red" to "all clean" in a single commit:

- `layers/sparsity/mixture_of_rookies.py` — if `dead`, delete; if `referenced`, rename to `topk_moe_legacy.py` and update importers. The file does **not** implement Mixture-of-Rookies (paper); leaving the misleading name is a documentation bug.
- SSM_TPU triplicate — keep `capibara/ssm/ssm_tpu.py` (most documented, best docstring) as the canonical S4 implementation. Delete `sub_models/SSM_TPU.py`. Remove the stale `S4Block` / `MambaBlock` import attempt at the top of `layers/ssm_hybrid_layers.py` (the source module no longer exists since BACKLOG-007).
- `spike_ssm.py` duplicate — keep `capibara/ssm/spike_ssm.py`, delete `sub_models/experimental/spike_ssm.py` (or vice versa, decided by which has more recent meaningful changes per `git log --follow`).
- `sub_models/mamba/mamba_module.py` — **do not delete** (it has correct algorithm intent), but move to `sub_models/_quarantine/mamba_module.py` with a `README.md` next to it that says: "Broken: uses PyTorch `.unsqueeze` inside `_selective_scan`; not Flax-compatible. See BACKLOG-018 (not yet promoted) for the rewrite." This makes the broken status visible in the file tree, not just in a manifest.
- `sub_models/__init__.py` — replace every `try / import / except Exception → ModuleName = None; FLAG = False` block with explicit `from .X import Y`. If a submodule is too heavyweight to import eagerly, gate it behind a single explicit `if os.environ.get("CAPIBARA_SUBMODELS") == "1"` flag, not behind silent exception swallowing.
- For each entry the BACKLOG-016 manifest classifies as `dead`: delete the file. Update any `__init__.py` re-exports.

**Problem**

The four problems enumerated under BACKLOG-016 are not just a documentation issue — they erode trust in module names ("if `mixture_of_rookies.py` is not Mixture-of-Rookies, what else is wrong?") and they make every future audit pay the same cost again. Fixing them once is cheap; postponing them compounds.

**Exit criteria**

- Re-running `tools/audit/inventory_layers_submodels.py` post-cleanup produces a manifest with **zero** `dead` entries and **zero** `Notes` flags of type `broken` / `misleading-name` / `duplicate-of` / `silent-fallback` — except for the explicitly quarantined `sub_models/_quarantine/mamba_module.py`, which remains flagged `broken` by design.
- `pytest -q` is green on the post-cleanup tree (no `referenced` file deleted by mistake).
- `commit_issue_backlog_017.sh` (worktree script, same template as 008/009) runs: AST parse on every touched file, the inventory script in `--check` mode, and the relevant pytest selection. Pushes only if all gates pass.
- Sentinel test: a single grep-style assertion that fails CI if `mixture_of_rookies` appears anywhere as a filename or class name without referencing arXiv 2202.04990. Same idea for any future "named after a paper but isn't" trap.
- Commit message records the pre/post manifest summary (counts per status) so the diff is auditable in `git log` without re-running the script.

---

## Resolved

- **Sanitize per-folder TODO documentation** — removed all 20 per-folder `TODOs.md`, the two global aggregators (`TODOs.md`, `TODOs_PRIORITIZED.md`) and the generator script `scripts/clean_todos.py`. Pending work now lives only in this file.
- **Restore `capibara/` directory** — the `capibara/` tree (~14,600 LOC in 43 Python files covering VQ, SSM, `mvp_api`) was restored after being removed by mistake in commit `e164e01`.
- **BACKLOG-016 — `tools/audit`: inventory of `layers/` and `sub_models/`.** Shipped `tools/audit/inventory_layers_submodels.py` (AST-based importer scan with re-export resolution + `availability-shim` vs `silent-fallback` distinction), `docs/sub_models_inventory.{json,md}` manifests, `tests/unit/test_inventory_consistency.py` drift detector (4 tests, `--check` mode gate). `CONTRIBUTING.md` adds the "no silent failures" policy that BACKLOG-016's `silent-fallback` flag enforces. Pre-cleanup audit on `main` reported 56 files / 1 alive / 25 referenced / 30 dead / 7 silent-fallback / 1 broken / 1 misleading-name / 4 duplicate-of.
- **BACKLOG-017 — Cleanup of dead code, duplicates, and misleading names in `layers/` + `sub_models/`.** Shipped together with BACKLOG-016 in `commit_issue_backlog_017.sh`. Deleted 47 files (entire `layers/abstract_reasoning/`, `layers/pasive/`, several leaf `layers/*.py`, `sub_models/Byte_TPU.py`, `sub_models/SSM_TPU.py`, `sub_models/aleph_Tilde.py`, `sub_models/capibaras/`, `sub_models/csa_expert_tpu_optimized.py`, `sub_models/experimental/`, `sub_models/mamba/` directory, `sub_models/semiotic/`, `sub_models/ultra_*`, `sub_models/vision/`, plus `layers/sparsity/{affine_quantizer,bitnet,mixture_of_rookies}.py`). Quarantined `sub_models/mamba/mamba_module.py` to `sub_models/_quarantine/mamba_module.py` with a README explaining the broken `.unsqueeze` PyTorch syntax (BACKLOG-018, not yet promoted, will rewrite it). Rewrote `layers/__init__.py`, `layers/sparsity/__init__.py`, `sub_models/__init__.py` with explicit `from .X import Y` re-exports — zero silent-fallback. Sentinel test `tests/unit/test_naming_sentinels.py` blocks future `mixture_of_rookies` reintroductions without an arXiv 2202.04990 citation. Post-cleanup manifest: 16 files / 1 alive / 15 referenced / **0 dead** / **0 silent-fallback** / **0 misleading-name** / **0 duplicate-of** / 6 availability-shim (acceptable, not banned) / 1 broken (the explicitly-quarantined mamba module, by design).

## How to add a new item

1. Give it the next free `ISSUE-NNN` identifier.
2. Include: Labels, Scope (files/paths), Problem (one or two sentences), Exit criteria (checklist of verifiable outcomes).
3. Keep the wording short and concrete — if you cannot point to a file, it is not ready to be here yet.
