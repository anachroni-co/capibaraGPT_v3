#!/usr/bin/env bash
#
# Commit plan for BACKLOG ISSUE-002 (meta_consensus mocks).
#
# Pre-req: your working tree currently holds the three edited files from
# the ongoing session:
#   - training/consensus/meta_consensus_system.py
#   - training/consensus/advance_meta_consensus_integration.py
#   - tests/integration/test_meta_consensus_no_mocks.py  (new)
#
# This script stashes them, switches to a fresh branch based on origin/main,
# re-applies the stash, and commits + pushes.
#
# Run from the repo root on your machine.

set -euo pipefail

echo ">>> current branch: $(git rev-parse --abbrev-ref HEAD)"
git status --short | head -10

# ------------------------------------------------------------------
# 0. Safety stash
# ------------------------------------------------------------------
git stash push --include-untracked -m "pre-backlog-002-meta-consensus" -- \
    training/consensus/meta_consensus_system.py \
    training/consensus/advance_meta_consensus_integration.py \
    tests/integration/test_meta_consensus_no_mocks.py

# ------------------------------------------------------------------
# 1. Fresh branch from origin/main
# ------------------------------------------------------------------
git fetch origin
git checkout -B fix/issue-backlog-002-meta-consensus-mocks origin/main

# ------------------------------------------------------------------
# 2. Re-apply the three edited files (tracked + new test)
# ------------------------------------------------------------------
git stash pop

# ------------------------------------------------------------------
# 3. Sanity check
# ------------------------------------------------------------------
python - <<'PY'
import ast, sys
paths = [
    "training/consensus/meta_consensus_system.py",
    "training/consensus/advance_meta_consensus_integration.py",
    "tests/integration/test_meta_consensus_no_mocks.py",
]
for p in paths:
    ast.parse(open(p, encoding="utf-8").read())
print("All 3 files parse OK")
PY

# Run the non-regression tests
python -m pytest tests/integration/test_meta_consensus_no_mocks.py -q || {
    echo "Tests failed - aborting"; exit 1;
}

# ------------------------------------------------------------------
# 4. Commit + push
# ------------------------------------------------------------------
git add \
    training/consensus/meta_consensus_system.py \
    training/consensus/advance_meta_consensus_integration.py \
    tests/integration/test_meta_consensus_no_mocks.py

git status --short
echo "--- review above and press Enter to commit, Ctrl+C to abort ---"
read -r _

git commit -m "fix(BACKLOG-002): remove mock_response/mock_metrics from meta consensus

ISSUE-002 (BACKLOG.md): training/consensus still shipped three hard-coded
placeholders on the main path:

- meta_consensus_system._execute_hybrid_routing synthesized a fake
  'Based on the analysis from N expert models, here is the consensus'
  string after the router finished.
- meta_consensus_system._execute_unified_consensus fed a hard-coded
  {loss: 0.5, accuracy: 0.85, perplexity: 2.1} into unified_consensus
  and returned a fixed 'fallback response' string.
- advance_meta_consensus_integration._apply_federated_consensus
  proposed consensus with a literal [{'response': 'mock_response', ...}]
  dict regardless of the real upstream result.

Changes:

- _execute_hybrid_routing now delegates to enhanced_consensus when it
  is available (real HF serverless executor) and returns an explicit
  low-confidence result with consensus_method='hybrid_routing_no_executor'
  when it is not. No fabricated response text.
- _execute_unified_consensus derives live metrics from query_history
  (avg confidence, quality, cost) and reuses the last successful
  consensus result; if none exists it returns a truthful empty-history
  marker instead of a fake string.
- _apply_federated_consensus builds the proposal from the real
  result.expert_responses (or from the aggregated result itself) so the
  federated layer votes on actual content.

Validation:

- 0 syntax errors across the three touched files.
- New tests/integration/test_meta_consensus_no_mocks.py (7 tests, all
  passing) asserts, via AST + source inspection, that the forbidden
  identifiers are gone and the new branches exist."

git push -u origin fix/issue-backlog-002-meta-consensus-mocks

echo
echo "Done. Open PR at:"
echo "  https://github.com/anachroni-co/capibaraGPT_v3/pull/new/fix/issue-backlog-002-meta-consensus-mocks"
