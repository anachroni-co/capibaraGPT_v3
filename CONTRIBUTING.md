# Contributing to CapibaraGPT v3

Thanks for taking the time to contribute. This document records the engineering
rules that we already apply de facto across the BACKLOG-002 to BACKLOG-006
work, and which we want to keep enforcing as the project grows.

It is intentionally short. If a rule below conflicts with a specific situation,
explain why in the PR description rather than silently bypassing it.

---

## 1. No silent failures

Scripts, runtime tools, and library functions in this repository **must** fail
loudly. Concretely:

- **Raise explicit exceptions** when an error condition is hit. Do not return
  `None`, `{}`, `0.0`, an empty array, or a "reasonable default" to paper over
  a real failure.
- **Do not suppress warnings or stack traces** in a `try/except: pass` (or
  `except Exception: logger.warning(...)`) just to keep a long run alive. If
  the caller wants tolerant behaviour, they pass an explicit flag (for example
  the `--enable-recovery` flag in `scripts/train.py` once BACKLOG-013 lands).
- **Do not embed `mock_*` fallbacks in the runtime path.** Mocks belong in
  tests. The `mock_response`, `mock_metrics`, `mock_lineage` and
  `np.random.randn`-as-params patterns we removed in BACKLOG-002 to BACKLOG-005
  are exactly the anti-pattern this rule targets.
- **If a feature is not yet wired**, exit non-zero with a clear message that
  points at the BACKLOG item that will wire it. `scripts/train.py` does this
  today: without `--synthetic-data` it returns exit code 3 and points at
  BACKLOG-010 (real data loader).

The reasoning is borrowed from *El Agente Forjador* (arXiv 2604.14609, §3.3):
silent failures mask bugs and prevent the self-healing behaviour central to a
debuggable system. Loud failures give the next reader — human or LLM — enough
information to actually fix the problem.

## 2. Validated public interfaces

Configuration, recipe, and any cross-module data contract uses one of:

- `@dataclass(frozen=True)` with a `__post_init__` that asserts invariants
  (the pattern used in `core.trainer.config_loader.TrainingConfig`,
  `core.model_factory.config.ModelConfig`).
- `pydantic.BaseModel` for interfaces where runtime type coercion is
  desirable (none in the repo today; reserved for future agent-tool
  contracts in `tools/kernel_agent/` if BACKLOG-011 grows that subproject).

Either choice makes the contract visible in code, validated at construction
time, and not reliant on convention.

## 3. Tests are not optional

Every BACKLOG item that touches runtime code lands with at least one of:

- Layer-1 tests that run on CPU without JAX/Flax/Optax (config, parsing,
  CLI argument handling).
- Layer-2 tests that depend on JAX, gated so they skip gracefully when the
  dependency is missing (e.g. the smoke trainer tests in BACKLOG-009).
- An AST-level sentinel check in the commit script when the rule being
  enforced is too subtle for a normal unit test (e.g. "no live
  `load_balance_weight` reference in `core/trainer/trainer.py`" — see
  `commit_issue_backlog_009.sh`).

Speedup numbers, when claimed, are accompanied by a numerical-equivalence
test (see BACKLOG-012). Reporting a speedup for code that fails
exec-accuracy is a hard error in the harness, not a warning.

## 4. Branch and commit hygiene

- One BACKLOG item per branch. Branch name: `fix/issue-backlog-NNN-<short-slug>`
  or `feat/issue-backlog-NNN-<short-slug>`.
- Each branch ships its own `commit_issue_backlog_NNN.sh` worktree script
  following the template established by BACKLOG-008 / 009: the script works
  on a clean `git worktree` rooted at `origin/main`, runs AST + sentinel
  checks, runs the relevant pytest selection, and only commits + pushes if
  every gate passes. This way the contributor's working tree never has to
  be in a clean state for the commit to be safe.
- Commit messages spell out the contract: what is added, why the design
  choice was made, and which sentinel checks guard against regression.
  See the `feat(BACKLOG-009)` message inside `commit_issue_backlog_009.sh`
  as the reference shape.

## 5. Adding a BACKLOG item

See `BACKLOG.md` — the rules at the bottom of that file are authoritative.
Briefly: an item is ready to land in the backlog only when it has a
concrete Scope (file paths), a one-paragraph Problem statement, and
verifiable Exit criteria. Vague items go in the *exploratory* tier
(see BACKLOG-014 for the template) and are promoted only after a design
doc resolves the open questions.
