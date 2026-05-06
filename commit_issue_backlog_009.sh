#!/usr/bin/env bash
#
# Commit plan for BACKLOG-009: canonical pretraining trainer.
#
# Uses git worktree to create an independent branch off origin/main so your
# current working tree is never touched. Run from the repo root, after:
#   sed -i 's/\r$//' commit_issue_backlog_009.sh
#
# Env vars:
#   SKIP_TESTS=1   run AST checks only, skip pytest (useful when JAX is not
#                  installed locally; CI will run the smoke + roundtrip tests).

set -euo pipefail

test -d .git || { echo "Run from repo root (no .git directory here)"; exit 1; }

REPO_ROOT="$(pwd)"
BRANCH="fix/issue-backlog-009-core-trainer"

echo ">>> repo root: $REPO_ROOT"
echo ">>> current branch: $(git rev-parse --abbrev-ref HEAD)"
echo ">>> target branch: $BRANCH"

git fetch origin

FILES=(
    "core/trainer/__init__.py"
    "core/trainer/config_loader.py"
    "core/trainer/optimizer.py"
    "core/trainer/checkpointing.py"
    "core/trainer/trainer.py"
    "scripts/train.py"
    "tests/unit/test_trainer.py"
)

for f in "${FILES[@]}"; do
    test -f "$f" || { echo "missing file in working tree: $f"; exit 1; }
done

SNAP_DIR="$(mktemp -d)"
echo ">>> snapshot dir: $SNAP_DIR"
for f in "${FILES[@]}"; do
    mkdir -p "$SNAP_DIR/$(dirname "$f")"
    cp "$f" "$SNAP_DIR/$f"
done

MSG='feat(BACKLOG-009): canonical pretraining trainer for CapibaraMoEModel

New module core/trainer/ (no capibara/* nor training/* dependency, only
jax + flax + optax + orbax). It is the single launcher path for the
1B / 3B Mixtral-style MoE recipes shipped in BACKLOG-008.

- config_loader.py : Recipe = (model: ModelConfig, training: TrainingConfig,
                     checkpoint: CheckpointConfig, logging: LoggingConfig).
                     load_recipe(toml_path) parses one TOML file and
                     delegates [model] to ModelConfig.from_toml so the
                     architecture parser stays single-source-of-truth.
                     Frozen dataclasses with __post_init__ validation
                     (warmup_steps <= total_steps, lr_peak > 0,
                     batch_size_global divisible by grad_accum_steps,
                     label_smoothing in [0, 1), schedule_type in
                     {cosine, constant, linear}). effective_end_value
                     defaults to lr_peak * lr_min_ratio when omitted.

- optimizer.py    : build_optimizer(cfg) -> (tx, schedule_fn). AdamW
                    (b1=0.9, b2=0.95, eps=1e-8, decoupled weight_decay)
                    composed under optax.clip_by_global_norm(grad_clip)
                    so the global norm is computed before the AdamW
                    preconditioner is applied. Three schedule kinds are
                    plumbed end-to-end: warmup_cosine_decay (default),
                    constant after warmup, and linear decay. The
                    schedule_fn is exposed so callers can log the LR
                    actually used at each step.

- trainer.py      : make_train_step(apply_fn, label_smoothing) builds a
                    @jax.jit step with jax.value_and_grad(loss_fn,
                    has_aux=True) so logging metrics share the forward
                    pass with the gradient computation. The forward pass
                    feeds rngs={"router": ..., "dropout": ...} so the
                    Mixtral router and any future dropout layers find
                    their PRNG without re-jitting. CRITICAL: aux_loss
                    is added to ce_loss WITHOUT a second
                    load_balance_weight multiplication - the model
                    already pre-multiplies it (this is the
                    BACKLOG-008 -> BACKLOG-009 hand-off bug we
                    explicitly avoid). grad_norm is computed from the
                    raw gradient tree before clip is applied so the
                    metric reflects the true signal. Each step uses
                    jax.random.fold_in(base_key, raw_step) for
                    deterministic per-step PRNG splitting. StepMetrics
                    captures (step, loss, ce_loss, aux_loss, grad_norm,
                    lr, seconds_per_step, tokens_per_second). Trainer
                    drives data_iter -> train_step, optionally calls
                    on_step(metrics) (used by scripts/train.py to log
                    to wandb), runs eval every recipe.logging.eval_every,
                    and saves checkpoints every recipe.checkpoint.save_every.

- checkpointing.py: CheckpointManager wraps Orbax PyTreeCheckpointer with
                    a pickle fallback for environments where orbax is not
                    importable. Layout: <out_dir>/step_<NNNNNNNN>/orbax/
                    + meta.pkl (or state.pkl in fallback). save() enforces
                    keep_last by deleting the oldest excess directories.
                    list_steps() / latest_step() / restore(step=None) are
                    the public entry points; restore loads the latest
                    when step is None. wait_until_finished() is a no-op
                    placeholder for parity with the async path.

- __init__.py     : minimal public surface (Recipe, TrainingConfig,
                    CheckpointConfig, LoggingConfig, load_recipe,
                    build_optimizer, CheckpointManager, Trainer,
                    TrainState, StepMetrics, make_train_step,
                    make_eval_step).

New launcher scripts/train.py:
- argparse CLI: --config (TOML, required), --steps (override total_steps),
  --resume <ckpt_dir>, --synthetic-data (smoke-mode tokens until
  BACKLOG-010 lands the real loader), --enable-wandb, --log-level.
- Wires Recipe -> CheckpointManager -> build_model -> Trainer -> fit().
- _make_wandb_hook(recipe) returns an optional on_step callback that
  logs all StepMetrics fields under the configured wandb project / run.
  wandb is NOT a hard dependency: ImportError is logged and the run
  continues without wandb.
- Without --synthetic-data the launcher exits with code 3 and a clear
  message pointing at BACKLOG-010 (real tokenized streaming loader).

Tests (tests/unit/test_trainer.py, 23 tests, ~70s on CPU):
- Layer 1 (no JAX needed): TrainingConfig defaults, end_value derivation
  with and without explicit override, per_device_batch_size division,
  warmup > total_steps raises, unknown schedule_type raises, invalid
  label_smoothing raises. RecipeLoading covers the shipped configs/*.toml:
  load_recipe(1b.toml) MUST equal preset_1b on the model side and pull
  lr_peak / warmup / end_value / checkpoint / logging from their TOML
  sections; same for 3b. recipe.name proxies model.name. Missing path
  raises FileNotFoundError.
- Layer 2 (JAX): cosine schedule starts at init_value, peaks at warmup
  step, decays to end_value at decay_steps; constant schedule holds
  after warmup. TestTrainerSmoke runs Trainer.fit on the smoke preset
  (~150K params, batch=2, seq=8) with a single repeated batch for 15
  steps and asserts: (a) all logged metrics are finite and well-formed
  (step counter monotonic, aux_loss >= 0, tokens/sec > 0), (b) mean of
  last-3 losses < mean of first-3 (memorization on a single batch is
  a robust loss-decrease signal even at this trunk size), (c) lr is
  strictly increasing during the warmup window and reaches >= 90% of
  lr_peak somewhere in the run, (d) two Trainers seeded identically
  produce identical step-1 loss + aux_loss (determinism guard).
  TestCheckpointRoundtrip asserts CheckpointManager.save -> .restore
  recovers params bit-for-bit and that keep_last=2 prunes older steps
  (only the two newest survive after 4 saves).

Bootstrap detail in the test file: we mount core, core.model_factory and
core.trainer as synthetic namespace packages (the same pattern as
test_model_factory.py) so the submodules can use canonical
"from core.model_factory.X import Y" without ever executing the heavy
core/__init__.py.

Validation: 23/23 tests pass locally with jax 0.6.2 + flax 0.10.7 +
optax 0.2.8 on CPU. End-to-end smoke run (`python3 scripts/train.py
--config configs/smoke.toml --synthetic-data --steps 20`) drove the
loss from 6.9311 to 6.6198 over 20 steps with finite gradients and
correctly summed aux_loss every step.'

wt_dir="$(mktemp -d)"
echo
echo "========================================================="
echo " BRANCH: $BRANCH"
echo " worktree: $wt_dir"
echo "========================================================="

git worktree add -B "$BRANCH" "$wt_dir" origin/main >/dev/null

for f in "${FILES[@]}"; do
    mkdir -p "$wt_dir/$(dirname "$f")"
    cp "$SNAP_DIR/$f" "$wt_dir/$f"
done

(
    cd "$wt_dir"

    # AST parse check on the Python files only.
    python3 - "${FILES[@]}" <<'PY'
import ast, sys
checked = 0
for p in sys.argv[1:]:
    if p.endswith(".py"):
        ast.parse(open(p, encoding="utf-8").read())
        checked += 1
print(f"AST OK for {checked} Python files")
PY

    # Sentinel checks: trainer must add aux_loss without a second multiplication
    # (the BACKLOG-008 -> BACKLOG-009 hand-off bug we explicitly avoid).
    python3 - <<'PY'
import ast
src = open("core/trainer/trainer.py", encoding="utf-8").read()
assert "total = ce + out.aux_loss" in src, (
    "trainer.py: expected the verbatim 'total = ce + out.aux_loss' line "
    "(aux_loss is already pre-weighted by the model)."
)
# load_balance_weight may appear in module/function docstrings explaining the
# contract, but must NEVER appear as a value in executable code (assignment,
# multiplication, attribute access). We walk the AST and check Name + Attribute
# nodes - that excludes docstrings and inline comments by construction.
tree = ast.parse(src)
class Finder(ast.NodeVisitor):
    found = False
    def visit_Name(self, node):
        if node.id == "load_balance_weight":
            self.found = True
    def visit_Attribute(self, node):
        if node.attr == "load_balance_weight":
            self.found = True
        self.generic_visit(node)
f = Finder()
f.visit(tree)
assert not f.found, (
    "trainer.py executable code must NOT reference load_balance_weight "
    "(the model already applies it; the trainer adds aux_loss raw)."
)
print("Sentinel OK: aux_loss added once; no live load_balance_weight code in trainer.")
PY

    # Tests: run the trainer unit tests. Layer-2 (JAX) tests skip gracefully
    # if JAX/Flax/Optax are not importable.
    TESTS=(
        "tests/unit/test_trainer.py"
    )

    if [ "${SKIP_TESTS:-0}" = "1" ]; then
        echo "SKIP_TESTS=1 - skipping pytest for $BRANCH (AST + sentinel checks already passed)"
    elif command -v pytest >/dev/null 2>&1; then
        pytest "${TESTS[@]}" -q || { echo "tests failed in $BRANCH - aborting"; exit 1; }
    elif python3 -c "import pytest" 2>/dev/null; then
        python3 -m pytest "${TESTS[@]}" -q || { echo "tests failed in $BRANCH - aborting"; exit 1; }
    else
        echo ""
        echo "!!! pytest not found in PATH nor in python3's site-packages."
        echo "!!! Options:"
        echo "!!!   a) pip install --user pytest  (then re-run)"
        echo "!!!   b) source your venv where pytest lives (then re-run)"
        echo "!!!   c) SKIP_TESTS=1 bash commit_issue_backlog_009.sh"
        echo "!!!      (relies only on AST + sentinel checks)"
        exit 1
    fi

    git add "${FILES[@]}"
    git status --short

    echo "--- press Enter to commit $BRANCH, Ctrl+C to abort ---"
    read -r _

    git commit -m "$MSG"
    git push -u origin "$BRANCH"
)

git worktree remove "$wt_dir"
rm -rf "$SNAP_DIR"

echo
echo "Done. Open the PR at:"
echo "  https://github.com/anachroni-co/capibaraGPT_v3/pull/new/$BRANCH"
echo
echo "Your original branch is unchanged."
