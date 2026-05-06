#!/usr/bin/env bash
#
# Commit plan: document training/unified_trainer.py as a non-canonical,
# consensus-distillation seed (companion to BACKLOG-009 plan-B trainer).
#
# Uses git worktree to create an independent branch off origin/main so your
# current working tree is never touched. Run from the repo root, after:
#   sed -i 's/\r$//' commit_doc_unified_trainer.sh
#
# This change is documentation-only: the file's executable code is untouched
# (still has the three broken imports). The point is to leave a clear paper
# trail so the next contributor knows exactly what would be needed to wire
# UnifiedTrainer into the future consensus-distillation track.

set -euo pipefail

test -d .git || { echo "Run from repo root (no .git directory here)"; exit 1; }

REPO_ROOT="$(pwd)"
BRANCH="docs/unified-trainer-integration-plan"

echo ">>> repo root: $REPO_ROOT"
echo ">>> current branch: $(git rev-parse --abbrev-ref HEAD)"
echo ">>> target branch: $BRANCH"

git fetch origin

FILES=(
    "training/unified_trainer.py"
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

MSG='docs(training): mark unified_trainer.py as non-canonical, document integration plan

UnifiedTrainer is preserved on disk as the seed for a future consensus-
distillation track, but it is NOT the canonical pretraining loop for the
new core/model_factory.CapibaraMoEModel introduced in BACKLOG-008.

This commit is documentation-only. The executable code is untouched -
the three relative imports still raise ModuleNotFoundError, exactly as
before, by design.

What changes:
- Replace the v2 docstring with an explicit integration plan that
  enumerates the seven things blocking revival:
    1. Three relative imports point to modules that have moved to
       config/training_config.py, training/consensus/unified_consensus.py,
       and training/optimizations/tpu_optimizations.py.
    2. Forward-pass signature uses training= where CapibaraMoEModel
       expects deterministic= and rngs={"router", "dropout"}.
    3. loss_fn drops the MoE load-balance aux_loss (would collapse the
       experts within the first epoch).
    4. Hyper-parameters and shapes are hardcoded (lr=3e-4, seq_len=2048).
    5. Checkpointing is a stub ("Restoration logic would go here").
    6. wandb.init runs unconditionally; CI cannot afford it.
    7. _consensus_train_step uses student-as-its-own-teacher placeholder.
- Add an INTEGRATION-TODO comment block right above the broken imports
  so any reader who skips the docstring still sees the warning.
- Reference the canonical pretraining trainer (core/trainer/, planned
  in BACKLOG-009) and the future migration phase that will absorb the
  metrics scaffold + consensus init into core/trainer/distillation.py.

Validation: ast.parse on the modified file passes. No behavior change
(the file was not importable before this commit, and is not importable
after either - by design).'

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

    # AST parse check.
    python3 - "${FILES[@]}" <<'PY'
import ast, sys
for p in sys.argv[1:]:
    if p.endswith(".py"):
        ast.parse(open(p, encoding="utf-8").read())
print(f"AST OK for {len(sys.argv) - 1} file(s)")
PY

    # Verify the docstring marker is present and the executable code below
    # the docstring was NOT accidentally modified (defensive check).
    python3 - <<'PY'
src = open("training/unified_trainer.py", encoding="utf-8").read()
assert "STATUS: NOT THE CANONICAL TRAINER" in src, "docstring marker missing"
assert "INTEGRATION-TODO" in src, "INTEGRATION-TODO comment missing"
# These three broken imports MUST still be present verbatim - that is the
# whole point. If a future linter "fixes" them, this guard catches it.
for needed in (
    "from .training_config import ModelScale",
    "from .consensus_strategies import",
    "from .tpu_optimizations import setup_tpu_environment",
):
    assert needed in src, f"executable line missing: {needed!r}"
print("doc markers + broken-import sentinels OK")
PY

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
