#!/usr/bin/env bash
#
# Commit plan for ISSUE-001 and ISSUE-002.
#
# Prerequisites:
#   - Working tree currently sits on `wip/pre-issue-002-backup` with:
#       * ISSUE-001 changes (training/tpu/... + tests/integration/...)
#       * ISSUE-002 changes (jax/ -> capibara/jax_ext/, 79 imports migrated,
#         capibara/jax/__init__.py tombstone)
#   - Run this from the repo root on YOUR machine (Windows/PowerShell or WSL).
#   - Branches already created: fix/issue-001-tpu-consensus-mocks, and we will
#     create fix/issue-002-jax-shim-collision fresh on top of main.
#
# Safety: this script stashes everything before switching branches, so no
# work-in-progress is lost. Read before running.

set -euo pipefail

REPO_ROOT="$(pwd)"

echo ">>> current branch: $(git rev-parse --abbrev-ref HEAD)"
git status --short | head -20
echo "... (truncated)"

# ------------------------------------------------------------------
# 0. Safety stash
# ------------------------------------------------------------------
git stash push --include-untracked -m "pre-issue-001-002-split"

# ------------------------------------------------------------------
# 1. ISSUE-001  -> fix/issue-001-tpu-consensus-mocks
# ------------------------------------------------------------------
git checkout fix/issue-001-tpu-consensus-mocks

# Bring ONLY the ISSUE-001 files from the stash
git checkout stash@{0} -- \
    training/tpu/tpu_v6_consensus_optimizer.py \
    tests/integration/test_tpu_v6_consensus_optimizer.py

git add \
    training/tpu/tpu_v6_consensus_optimizer.py \
    tests/integration/test_tpu_v6_consensus_optimizer.py

git commit -m "fix(ISSUE-001): replace TPU v6 consensus mocks with deterministic implementations

- Replace random.normal mock embeddings with SHA256-based deterministic embeddings.
- Replace simulated TPU metrics with values derived from actual execution.
- Add integration tests covering embedding determinism, query embedding,
  expert preparation, metrics snapshot, and end-to-end CPU fallback path.
- Keeps full fallback compatibility when JAX/TPU are not available."

git push -u origin fix/issue-001-tpu-consensus-mocks

# ------------------------------------------------------------------
# 2. ISSUE-002  -> fix/issue-002-jax-shim-collision (fresh from main)
# ------------------------------------------------------------------
git checkout main
git pull --ff-only origin main || true
git checkout -B fix/issue-002-jax-shim-collision

# Restore the full working tree from the stash.
git stash pop

# Remove the ISSUE-001-only files from this branch (already committed in #1).
git checkout HEAD -- \
    training/tpu/tpu_v6_consensus_optimizer.py || true
# (The test file is untracked on main, so just discard it here.)
rm -f tests/integration/test_tpu_v6_consensus_optimizer.py

# Explicit delete of the old jax/ tree (mv left 69 "D" entries).
git rm -r --ignore-unmatch jax/ || true

# Stage everything for the migration commit.
git add -A

git status --short | head -30
echo "--- review above and press Enter to commit, Ctrl+C to abort ---"
read -r _

git commit -m "fix(ISSUE-002): resolve capibara.jax shim collision with real JAX

Root cause: the local \`jax/\` directory shadowed the real PyPI \`jax\`
package on sys.path. This caused circular imports
(\`partially initialized module 'jax.tree_util' has no attribute
'tree_flatten'\`) and unpredictable symbol shadowing across the codebase.

Changes:

- Move \`jax/\` -> \`capibara/jax_ext/\`. Only \`tpu_v4/\` (custom TPU
  kernels, profilers, backend setup) has real repo-local content; that
  subpackage stays in-repo under the new namespace.
- Migrate 79 files: \`capibara.jax.*\` -> \`jax.*\`, with
  \`capibara.jax.tpu_v4.*\` -> \`capibara.jax_ext.tpu_v4.*\`.
- Replace \`capibara/jax/__init__.py\` with a tombstone that raises
  \`ImportError\` so any lingering consumer fails loudly instead of
  re-shadowing the real \`jax\`.

Validation: 0 syntax errors across the 86 touched files; \`jax\` real,
\`jax.numpy\`, and \`capibara.jax_ext.tpu_v4\` import cleanly; ISSUE-001
integration assertions still pass on the migrated tree."

git push -u origin fix/issue-002-jax-shim-collision

echo
echo "Both branches pushed. Open PRs at:"
echo "  https://github.com/anachroni-co/capibaraGPT_v3/pull/new/fix/issue-001-tpu-consensus-mocks"
echo "  https://github.com/anachroni-co/capibaraGPT_v3/pull/new/fix/issue-002-jax-shim-collision"
