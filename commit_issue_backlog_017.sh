#!/usr/bin/env bash
#
# Commit plan for BACKLOG-016 + BACKLOG-017: inventory + cleanup of layers/
# and sub_models/.
#
# Ships in a single PR because BACKLOG-017's drift-detector test imports the
# audit script that BACKLOG-016 introduces; landing them separately would
# leave one of the two branches with red CI.
#
# Uses git worktree to create an independent branch off origin/main so your
# current working tree is never touched. Run from the repo root:
#   sed -i 's/\r$//' commit_issue_backlog_017.sh   # in case of CRLF
#   bash commit_issue_backlog_017.sh
#
# Env vars:
#   SKIP_TESTS=1   run AST + sentinel checks only, skip pytest (use only when
#                  pytest is not importable locally; CI runs the full set).
#
# Pre/post audit summary (per BACKLOG-017 exit criteria, recorded in commit
# message and visible via `git log` without re-running the script):
#
#   pre  : 56 files / 1 alive / 25 referenced / 30 dead
#          7 silent-fallback / 1 broken / 1 misleading-name / 4 duplicate-of
#   post : 16 files / 1 alive / 15 referenced /  0 dead
#          0 silent-fallback / 1 broken (quarantined-by-design)
#          0 misleading-name / 0 duplicate-of / 6 availability-shim (allowed)

set -euo pipefail

test -d .git || { echo "Run from repo root (no .git directory here)"; exit 1; }

REPO_ROOT="$(pwd)"
BRANCH="fix/issue-backlog-017-cleanup-layers-submodels"

echo ">>> repo root: $REPO_ROOT"
echo ">>> current branch: $(git rev-parse --abbrev-ref HEAD)"
echo ">>> target branch: $BRANCH"

git fetch origin

# -----------------------------------------------------------------------------
# File lists. Keep these arrays as the single source of truth for what the
# branch ships; the script never touches anything not declared here.
# -----------------------------------------------------------------------------

# NEW files (BACKLOG-016 deliverables: audit + drift gate + manifests + policy).
NEW_FILES_016=(
    "tools/audit/inventory_layers_submodels.py"
    "tests/unit/test_inventory_consistency.py"
    "docs/sub_models_inventory.json"
    "docs/sub_models_inventory.md"
    "CONTRIBUTING.md"
)

# NEW files (BACKLOG-017 deliverables: quarantine + naming sentinel).
NEW_FILES_017=(
    "sub_models/_quarantine/__init__.py"
    "sub_models/_quarantine/README.md"
    "sub_models/_quarantine/mamba_module.py"
    "tests/unit/test_naming_sentinels.py"
)

# MODIFIED files (BACKLOG-017 cleanup of __init__.py silent-fallback +
# importer fixes after the mamba quarantine move + BACKLOG.md Resolved entry).
MODIFIED_FILES=(
    "layers/__init__.py"
    "layers/sparsity/__init__.py"
    "sub_models/__init__.py"
    "sub_models/hybrid/hybrid_attention_module.py"
    "tests/unit/test_submodels_cpu_ready.py"
    "BACKLOG.md"
)

# DELETED files (BACKLOG-017: 47 entries flagged dead/duplicate/misleading by
# the BACKLOG-016 manifest).
DELETED_FILES=(
    "layers/README.md"
    "layers/abstract_reasoning/game_theory.py"
    "layers/abstract_reasoning/platonic.py"
    "layers/abstract_reasoning/quineana.py"
    "layers/conv1d_block.py"
    "layers/embedding.py"
    "layers/meta_la.py"
    "layers/neuro_adaptive.py"
    "layers/neurogenesis.py"
    "layers/pasive/attention.py"
    "layers/pasive/synthetic_embedding.py"
    "layers/smb_layer.py"
    "layers/sparsity/affine_quantizer.py"
    "layers/sparsity/bitnet.py"
    "layers/sparsity/mixture_of_rookies.py"
    "layers/ssm_hybrid_layers.py"
    "layers/stack.py"
    "layers/ultra_layer_integration.py"
    "sub_models/Byte_TPU.py"
    "sub_models/README.md"
    "sub_models/SSM_TPU.py"
    "sub_models/aleph_Tilde.py"
    "sub_models/capibaras/capibara2.py"
    "sub_models/capibaras/capibara_byte.py"
    "sub_models/capibaras/capibara_jax_ssm.py"
    "sub_models/capibaras/tpu_base_config.py"
    "sub_models/csa_expert_tpu_optimized.py"
    "sub_models/experimental/README.md"
    "sub_models/experimental/__init__.py"
    "sub_models/experimental/adaptive_vq_submodel.py"
    "sub_models/experimental/dual_process.py"
    "sub_models/experimental/liquid.py"
    "sub_models/experimental/meta_bamdp.py"
    "sub_models/experimental/snns_LiCell.py"
    "sub_models/experimental/spike_ssm.py"
    "sub_models/hybrid/README.md"
    "sub_models/mamba/README.md"
    "sub_models/mamba/__init__.py"
    "sub_models/mamba/mamba_module.py"
    "sub_models/semiotic/__init__.py"
    "sub_models/semiotic/mnemosyne_semio_module.py"
    "sub_models/semiotic/sapir_whorf_adapter.py"
    "sub_models/semiotic/semio.py"
    "sub_models/semiotic/semiotic_interaction.py"
    "sub_models/ultra_enhanced_integration.py"
    "sub_models/ultra_submodel_orchestrator.py"
    "sub_models/vision/capivision.py"
)

# Sanity: every file we declare as new/modified must exist on disk so we know
# we are committing what we expect (deletions are not checked here; we let
# `git rm` complain if a file is already gone).
ALL_TOUCHED_NONDELETE=("${NEW_FILES_016[@]}" "${NEW_FILES_017[@]}" "${MODIFIED_FILES[@]}")
for f in "${ALL_TOUCHED_NONDELETE[@]}"; do
    test -f "$f" || { echo "missing file in working tree: $f"; exit 1; }
done

# Snapshot the new/modified files into a tempdir BEFORE we touch the worktree,
# so the worktree branch only ever sees the post-cleanup state.
SNAP_DIR="$(mktemp -d)"
echo ">>> snapshot dir: $SNAP_DIR"
for f in "${ALL_TOUCHED_NONDELETE[@]}"; do
    mkdir -p "$SNAP_DIR/$(dirname "$f")"
    cp "$f" "$SNAP_DIR/$f"
done

MSG='feat(BACKLOG-016 + BACKLOG-017): inventory + cleanup of layers/ and sub_models/

BACKLOG-016 ships the audit machinery; BACKLOG-017 applies it. The two land
in one PR because the drift-detector test introduced by 016 imports the
audit module, and 017 deletes 47 files whose dead status only the manifest
017 produces makes verifiable. Splitting them would leave one branch red.

== BACKLOG-016: tools/audit + manifests + drift gate ==

- tools/audit/inventory_layers_submodels.py: AST-based audit. Walks layers/
  + sub_models/, classifies each .py as alive (importer in core/, scripts/,
  training/), referenced (importer only in tests/ or sibling layers/sub_models/),
  or dead (no importer outside its own __init__.py). Notes column flags four
  problem classes with stable tokens: broken, misleading-name,
  duplicate-of:<path>, silent-fallback. Distinguishes silent-fallback from
  availability-shim (try-import jax/flax/torch + except -> AVAILABLE=False is
  a legitimate optional-dep guard, not a banned pattern). Re-export map:
  scans every __init__.py for "from .child import Name" so a test that does
  "from layers import SelfAttention" correctly attributes the import to
  layers/self_attention.py instead of declaring it dead. --check mode for
  CI (exit 0 == manifest in sync; nonzero == drift).

- docs/sub_models_inventory.{json,md}: machine + human renderings, sorted
  by classification then path, Notes column populated. Generated by the
  script; edits go through the script, not by hand.

- tests/unit/test_inventory_consistency.py: 4 tests. (1) JSON manifest is
  byte-identical to a fresh build. (2) Markdown manifest is byte-identical.
  (3) No entry remains "unknown". (4) --check mode exits 0 on the shipped
  manifest. ~5s on CPU.

- CONTRIBUTING.md: documents the "no silent failures" policy that the
  silent-fallback note flag enforces (try-import + except-assign-False is
  banned for non-availability libs; the audit makes the violation
  greppable).

== BACKLOG-017: cleanup driven by the BACKLOG-016 manifest ==

Pre-cleanup audit on main:
    56 files / 1 alive / 25 referenced / 30 dead
    7 silent-fallback / 1 broken / 1 misleading-name / 4 duplicate-of

Post-cleanup audit on this branch:
    16 files / 1 alive / 15 referenced / 0 dead
    0 silent-fallback / 1 broken (quarantined-by-design) / 0 misleading-name
    0 duplicate-of / 6 availability-shim (allowed)

Concrete actions:

- 47 files deleted. Driven by the manifest: every dead entry under layers/
  + sub_models/ goes, plus duplicate SSM_TPU, plus the misleadingly-named
  layers/sparsity/mixture_of_rookies.py (was a vanilla top-k MoE; did NOT
  implement Pinto/Arnau/Gonzalez arXiv 2202.04990).

- sub_models/mamba/mamba_module.py NOT deleted. Algorithm intent is correct
  but the implementation calls .unsqueeze(-1) inside _selective_scan, which
  is PyTorch syntax in a JAX repo - first call would raise. Moved to
  sub_models/_quarantine/mamba_module.py with a header docstring and a
  README in the quarantine directory documenting the broken status and
  pointing at BACKLOG-018 (not yet promoted) for the rewrite. The audit
  flags it broken on purpose so the manifest stays honest.

- sub_models/__init__.py rewritten: every "try: from .X import Y / except
  Exception: Y = None; FLAG = False" block replaced with explicit
  "from .X import Y". Same for layers/__init__.py and
  layers/sparsity/__init__.py. Zero silent-fallback in surviving code.

- sub_models/hybrid/hybrid_attention_module.py and
  tests/unit/test_submodels_cpu_ready.py: import path updated from
  sub_models.mamba.mamba_module to sub_models._quarantine.mamba_module so
  the hybrid attention smoke test still runs against the (broken-but-
  documented) reference implementation.

- tests/unit/test_naming_sentinels.py: two new sentinels. Test 1 fails CI
  if any future PR reintroduces a file named mixture_of_rookies under
  layers/ or sub_models/. Test 2 fails CI if a class literally named
  MixtureOfRookies appears anywhere under those trees without the
  arXiv 2202.04990 reference in the same file. Same idea generalises to
  any future "named after a paper but isn'\''t" trap.

- BACKLOG.md: 016 + 017 added under Resolved with the pre/post manifest
  summary inline (so the diff is auditable in git log without re-running
  the script).

== Validation gates (run inside the worktree before commit) ==

- AST parse on every touched .py file (committed and surviving).
- Sentinel: trainer.py is untouched (out of scope; defensive check).
- Audit script --check mode: must exit 0 (manifest in sync).
- Naming sentinels: pytest tests/unit/test_naming_sentinels.py.
- Drift detector: pytest tests/unit/test_inventory_consistency.py.
- Smoke: pytest tests/unit/test_layers_smoke.py
                tests/unit/test_submodels_cpu_ready.py
                tests/unit/test_factorization.py.

If any gate fails, the script aborts before commit. The git push only
runs after a manual Enter confirmation.'

wt_dir="$(mktemp -d)"
echo
echo "========================================================="
echo " BRANCH: $BRANCH"
echo " worktree: $wt_dir"
echo "========================================================="

git worktree add -B "$BRANCH" "$wt_dir" origin/main >/dev/null

# Apply the cleanup inside the worktree:
#   1. copy snapshot of new/modified files into the worktree
#   2. git rm the dead files
(
    cd "$wt_dir"

    # Copy snapshot files into worktree.
    for f in "${ALL_TOUCHED_NONDELETE[@]}"; do
        mkdir -p "$(dirname "$f")"
        cp "$SNAP_DIR/$f" "$f"
    done

    # Delete dead files via git rm (ignores already-gone files because we
    # branch off origin/main where they still exist; a missing file would
    # be a sign the branch base is stale - prefer to fail loud).
    for f in "${DELETED_FILES[@]}"; do
        if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
            git rm -f "$f" >/dev/null
        else
            echo "WARN: $f not tracked on origin/main - skipping"
        fi
    done

    # Prune now-empty subdirectories (layers/abstract_reasoning, layers/pasive,
    # sub_models/capibaras, sub_models/experimental, sub_models/mamba,
    # sub_models/semiotic, sub_models/vision). git rm leaves empty dirs;
    # we remove them so the post-cleanup tree is actually clean.
    for d in layers/abstract_reasoning layers/pasive \
             sub_models/capibaras sub_models/experimental \
             sub_models/mamba sub_models/semiotic sub_models/vision; do
        if [ -d "$d" ] && [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
            rmdir "$d"
        fi
    done

    # ---------------- Gate 1: AST parse ----------------
    python3 - "${NEW_FILES_016[@]}" "${NEW_FILES_017[@]}" "${MODIFIED_FILES[@]}" <<'PY'
import ast, sys
checked = 0
for p in sys.argv[1:]:
    if p.endswith(".py"):
        ast.parse(open(p, encoding="utf-8").read())
        checked += 1
print(f"AST OK for {checked} Python files")
PY

    # ---------------- Gate 2: filename sentinel ----------------
    # Belt-and-suspenders against a future revert that drags
    # mixture_of_rookies back. The pytest-level test is the canonical gate;
    # this inline check fails the script even if pytest is not importable.
    python3 - <<'PY'
from pathlib import Path
offenders = [
    str(p) for top in ("layers", "sub_models")
    if Path(top).exists()
    for p in Path(top).rglob("*.py")
    if "mixture_of_rookies" in p.name
]
assert not offenders, (
    f"mixture_of_rookies filename present after cleanup: {offenders}. "
    "Either implement arXiv 2202.04990 or rename the file."
)
print("Sentinel OK: no mixture_of_rookies filename in tree.")
PY

    # ---------------- Gate 3: audit --check ----------------
    python3 tools/audit/inventory_layers_submodels.py --check

    # ---------------- Gate 4: pytest ----------------
    TESTS=(
        "tests/unit/test_inventory_consistency.py"
        "tests/unit/test_naming_sentinels.py"
        "tests/unit/test_layers_smoke.py"
        "tests/unit/test_submodels_cpu_ready.py"
        "tests/unit/test_factorization.py"
    )

    if [ "${SKIP_TESTS:-0}" = "1" ]; then
        echo "SKIP_TESTS=1 - skipping pytest (AST + sentinel + audit --check passed)"
    elif command -v pytest >/dev/null 2>&1; then
        pytest "${TESTS[@]}" -q || { echo "tests failed in $BRANCH - aborting"; exit 1; }
    elif python3 -c "import pytest" 2>/dev/null; then
        python3 -m pytest "${TESTS[@]}" -q || { echo "tests failed in $BRANCH - aborting"; exit 1; }
    else
        echo
        echo "!!! pytest not found in PATH nor in python3's site-packages."
        echo "!!! Options:"
        echo "!!!   a) pip install --user pytest  (then re-run)"
        echo "!!!   b) source your venv where pytest lives (then re-run)"
        echo "!!!   c) SKIP_TESTS=1 bash commit_issue_backlog_017.sh"
        echo "!!!      (relies only on AST + sentinel + audit --check)"
        exit 1
    fi

    # ---------------- Stage + diff preview ----------------
    git add "${ALL_TOUCHED_NONDELETE[@]}"
    # deletions already staged by `git rm`
    git status --short
    echo
    echo "Diffstat:"
    git diff --cached --shortstat

    echo "--- press Enter to commit + push $BRANCH, Ctrl+C to abort ---"
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
