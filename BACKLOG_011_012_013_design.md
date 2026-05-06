# BACKLOG-011 / 012 / 013 — design notes

Drafts inspired by *GEAK: Introducing Triton Kernel AI Agent & Evaluation
Benchmarks* (AMD, arXiv 2507.23194). The three items are deliberately
independent so they can land on separate branches.

Recommended landing order: **012 → 011 → 013**. The harness in 012 is the
acceptance gate for 011; 013 is orthogonal and can ship any time.

---

## BACKLOG-012 — kernel benchmark harness (DETAILED)

### Why this lands first

BACKLOG-011 rewrites four numerical hot-paths. Without 012 there is no
neutral place to compare a candidate kernel against its Flax reference under
the same JAX runtime. The Anexo B of the GEAK paper is explicit: "strong
test suites are non-negotiable; EvalPlus dropped false-pass rates by 28.9pp
just by augmenting test coverage." We are not building an agent here — we
are building the contract the agent will need.

### Layout

    tests/kernel_bench/
        __init__.py
        bench_runner.py           # the harness
        registry.py               # imports each kernel module to register it
        kernels/
            __init__.py
            rmsnorm.py            # one file per candidate kernel
            rope.py
            moe_dispatch.py
            swiglu.py
        test_kernel_bench.py      # pytest entry point

### Per-kernel contract

Each file in `tests/kernel_bench/kernels/` exposes a `KERNEL_SPEC` dict:

    KERNEL_SPEC = {
        "name":         "rmsnorm",
        "reference_fn": rmsnorm_flax_reference,    # Flax callable
        "candidate_fn": rmsnorm_pallas_candidate,  # may be None until BACKLOG-011
        "inputs_fn":    rmsnorm_inputs,            # def inputs_fn(seed, size) -> tuple
        "sizes":        ["small", "medium", "large"],
        "tolerances":   {"atol": 1e-5, "rtol": 1e-4},
        "warmup":       3,
        "repeats":      11,                        # odd -> well-defined median
    }

`candidate_fn=None` is a valid state: the harness reports the kernel as
"NOT REGISTERED" and skips it. This lets us merge BACKLOG-012 with an empty
registry, then BACKLOG-011 fills in the candidates one by one.

### Three metrics, in order

The harness mirrors the metric names from TritonBench-revised so the
labelling is portable:

1. **Call accuracy** — `candidate_fn(*inputs)` returns without raising.
2. **Exec accuracy** — `jnp.allclose(out_candidate, out_reference, **tol)`.
   On failure, log the max abs diff and the index where it occurs.
3. **Speedup** — only computed when (1) and (2) pass. Median of `repeats`
   timed runs of `candidate_fn / reference_fn`, with `block_until_ready`
   on each output and `warmup` runs discarded.

A failed exec-accuracy run **must not** report a speedup number. This is
the "fast wrong code" trap from the paper's Anexo B; the sentinel test
asserts the harness cannot bypass it.

### bench_runner.py — pseudocode

    def run_kernel(spec, size):
        rng = jax.random.PRNGKey(0)
        inputs = spec["inputs_fn"](rng, size)
        ref_out = spec["reference_fn"](*inputs)

        if spec["candidate_fn"] is None:
            return Result(name=spec["name"], size=size, status="UNREGISTERED")

        # call accuracy
        try:
            cand_out = spec["candidate_fn"](*inputs)
        except Exception as e:
            return Result(..., status="CALL_FAIL", err=str(e))

        # exec accuracy
        ok = jnp.allclose(cand_out, ref_out, **spec["tolerances"])
        if not bool(ok):
            return Result(..., status="EXEC_FAIL",
                          max_abs_diff=float(jnp.max(jnp.abs(cand_out - ref_out))))

        # speedup
        for _ in range(spec["warmup"]):
            spec["candidate_fn"](*inputs)[...].block_until_ready()
            spec["reference_fn"](*inputs)[...].block_until_ready()
        ref_t = _time_median(spec["reference_fn"], inputs, spec["repeats"])
        cand_t = _time_median(spec["candidate_fn"], inputs, spec["repeats"])
        return Result(..., status="OK", speedup=ref_t / cand_t)

### test_kernel_bench.py — pytest contract

    def test_kernel_bench_table():
        results = [run_kernel(spec, size) for spec in REGISTRY for size in spec["sizes"]]
        _print_table(results)
        # Hard assertions only on correctness:
        for r in results:
            if r.status == "EXEC_FAIL":
                pytest.fail(f"{r.name}/{r.size}: exec accuracy failed (max_abs_diff={r.max_abs_diff})")
            if r.status == "CALL_FAIL":
                pytest.fail(f"{r.name}/{r.size}: call failed: {r.err}")
        # Speedup is logged but not gated here. BACKLOG-011 gates it
        # in its own CI job (acceptance pipeline).

When the registry is empty (the BACKLOG-012-only state), the table is
just the header and one informational line:

    [kernel_bench] no kernels registered yet (BACKLOG-011)

This must still exit 0.

### Test list (criterios de aceptación verificables)

In `tests/kernel_bench/test_harness_self.py` we test the harness itself:

1. `test_empty_registry_returns_zero` — no kernels registered, pytest exits 0.
2. `test_unregistered_candidate_does_not_run_speedup` — candidate=None
   never reaches the timing block.
3. `test_exec_fail_blocks_speedup_report` — when allclose returns False,
   the result has `speedup=None`.
4. `test_call_fail_classified_correctly` — a kernel that raises is
   classified `CALL_FAIL` and not `EXEC_FAIL`.
5. `test_warmup_runs_are_excluded_from_median` — patch the timer, assert
   only `repeats` samples reach the median.
6. `test_block_until_ready_is_called` — patch a sentinel, assert the
   harness calls `.block_until_ready()` on every timed run.
7. `test_table_printer_handles_all_statuses` — formatting test for the
   four statuses (`OK`, `EXEC_FAIL`, `CALL_FAIL`, `UNREGISTERED`).

### Sentinel checks for the commit script

The commit script (`commit_issue_backlog_012.sh`) runs three guards before
calling `git commit`:

1. **AST parses** for every file in the new package.
2. **Empty-registry pytest is green** — `pytest tests/kernel_bench/test_kernel_bench.py -q`
   in the worktree must exit 0 even before any kernel is registered.
3. **Sentinel: speedup-without-correctness is impossible** — grep the
   harness AST for the path "EXEC_FAIL → return Result(speedup=None)" and
   fail the script if anyone removes it.

### Pyproject marker

    [tool.pytest.ini_options]
    markers = [
        "kernel_bench: heavyweight kernel benchmark (run on accelerator runners)",
    ]

CI default: `pytest -m "not kernel_bench"`. The kernel-bench job runs
`pytest -m kernel_bench` on a GPU/TPU runner.

---

## BACKLOG-011 — Pallas kernels for the MoE hot path (sketch)

### Hot-path inventory in `core/model_factory/blocks.py`

| Class               | Lines     | Why it's a candidate                                                  |
|---------------------|-----------|----------------------------------------------------------------------|
| `RMSNorm`           | 31–55     | trivially fusable: scale + rsqrt + mul. ~10% of step on small models. |
| `apply_rotary` / RoPE | 58–83   | in-place rotate halves; XLA emits a copy.                            |
| `SwiGLUExpert`      | 174–215   | three matmuls + silu; fuses into one kernel cleanly.                 |
| `SparseMoEBlock`    | 283–355   | per-token gather of `w_gate[idx_k]` + grouped matmul; 1.5–3× ceiling. |

`SparseMoEBlock` is the highest-value rewrite and the hardest. The other
three are warmups that prove the harness and the feature flag work.

### Feature flag

In `core/model_factory/config.py`:

    @dataclass(frozen=True)
    class ModelConfig:
        ...
        use_pallas_kernels: bool = False

`blocks.py` branches at the top of each `__call__` on
`self.use_pallas_kernels` (threaded through from `ModelConfig`). Default
is False so the existing CPU smoke run is unaffected.

### Acceptance per kernel (gated by BACKLOG-012)

For every Pallas candidate to land:

- `tests/kernel_bench/kernels/<name>.py` registered with `candidate_fn`
  pointing at the Pallas implementation.
- `pytest tests/kernel_bench/` reports `OK` for that kernel on all sizes.
- Median speedup ≥ `1.2×` on the target accelerator (logged, not in CI's
  smoke gate).
- Numerical equivalence: `atol=1e-5, rtol=1e-4` against the Flax reference.

### Out of scope here (deliberately)

- The agent loop from GEAK (Generator/Reflector/Evaluator/Optimizer) is
  **not** a runtime dependency. If we ever build it, it lives in
  `tools/kernel_agent/` and emits committed `.py` files into
  `core/kernels/`. It does not run during training.

---

## BACKLOG-013 — RecoveryHook (sketch)

### Public surface

    @dataclass(frozen=True)
    class RecoveryConfig:
        nan_inf_action: Literal["skip", "abort"] = "skip"
        explosion_factor: float = 5.0
        explosion_window: int = 20
        aux_band_sigma: float = 4.0
        lr_halving_after: int = 3       # consecutive skips before halving LR
        max_consecutive_skips: int = 10  # then abort

    class RecoveryHook:
        def __init__(self, cfg: RecoveryConfig, ckpt_mgr=None): ...
        def __call__(self, m: StepMetrics) -> Optional[RecoveryAction]: ...
        # returns None on healthy steps; an action otherwise.

### Detectors

1. NaN/Inf in `loss` or `grad_norm` → `nan_inf_action`.
2. `loss > explosion_factor * median(last explosion_window losses)` → `skip`,
   then `halve_lr` after `lr_halving_after` consecutive skips, then
   `abort` after `max_consecutive_skips`.
3. `aux_loss > mean(window) + aux_band_sigma * std(window)` → `skip`
   (aux spikes usually mean a router collapse — the model recovers a few
   steps later if you skip the bad batch).

### Wiring

`Trainer.fit` already calls `on_step(metrics)` at the end of each step.
For `skip` to be useful we need to consult the hook **before** committing
the gradient — small refactor: emit `pre_metrics` (loss/aux/grad_norm,
without the params yet) and let the hook veto.

`scripts/train.py --enable-recovery` constructs a default `RecoveryHook`
and passes it as `on_step`. Default off; tests run with explicit configs.

### Test list

`tests/unit/test_trainer_recovery.py`:

- `test_nan_loss_triggers_skip`
- `test_nan_grad_norm_triggers_skip`
- `test_explosion_after_window_triggers_skip`
- `test_explosion_does_not_fire_inside_warmup`
- `test_consecutive_skips_halve_lr`
- `test_too_many_skips_aborts`
- `test_aux_band_spike_triggers_skip`
- `test_healthy_metrics_return_none`
- `test_recoveryhook_is_pure_no_jax_dependency`  # only needs StepMetrics + numpy

All Layer-1 tests (no JAX needed) so they run in <1 s on CI.

---

## Recommended commit/branch plan

| Branch                                       | Lands              | Depends on |
|----------------------------------------------|--------------------|------------|
| `feat/issue-backlog-012-kernel-bench`        | BACKLOG-012        | none       |
| `feat/issue-backlog-013-recovery-hook`       | BACKLOG-013        | none       |
| `feat/issue-backlog-011-pallas-rmsnorm`      | BACKLOG-011 step 1 | 012 merged |
| `feat/issue-backlog-011-pallas-rope`         | BACKLOG-011 step 2 | 012 merged |
| `feat/issue-backlog-011-pallas-swiglu`       | BACKLOG-011 step 3 | 012 merged |
| `feat/issue-backlog-011-pallas-moe-dispatch` | BACKLOG-011 step 4 | 012 merged |

Each branch ships its own `commit_issue_backlog_NNN_<step>.sh` worktree
script following the same template as 008 / 009.
