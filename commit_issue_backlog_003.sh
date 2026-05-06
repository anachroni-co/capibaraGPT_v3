#!/usr/bin/env bash
#
# Commit plan for BACKLOG ISSUE-003 (services/automation real execution).
#
# Pre-req: your working tree currently holds the three edited files from
# the ongoing session:
#   - services/automation/agent_executor.py
#   - services/automation/n8n_service.py
#   - tests/integration/test_automation_no_simulation.py  (new)
#
# This script stashes them, switches to a fresh branch based on origin/main,
# re-applies the stash, and commits + pushes.
#
# Run from the repo root on your machine, after:
#   sed -i 's/\r$//' commit_issue_backlog_003.sh

set -euo pipefail

echo ">>> current branch: $(git rev-parse --abbrev-ref HEAD)"
git status --short | head -10

# ------------------------------------------------------------------
# 0. Safety stash
# ------------------------------------------------------------------
git stash push --include-untracked -m "pre-backlog-003-automation" -- \
    services/automation/agent_executor.py \
    services/automation/n8n_service.py \
    tests/integration/test_automation_no_simulation.py

# ------------------------------------------------------------------
# 1. Fresh branch from origin/main
# ------------------------------------------------------------------
git fetch origin
git checkout -B fix/issue-backlog-003-automation-real-execution origin/main

# ------------------------------------------------------------------
# 2. Re-apply the three edited files (tracked + new test)
# ------------------------------------------------------------------
git stash pop

# ------------------------------------------------------------------
# 3. Sanity check
# ------------------------------------------------------------------
python - <<'PY'
import ast
paths = [
    "services/automation/agent_executor.py",
    "services/automation/n8n_service.py",
    "tests/integration/test_automation_no_simulation.py",
]
for p in paths:
    ast.parse(open(p, encoding="utf-8").read())
print("All 3 files parse OK")
PY

# Run the non-regression tests
python -m pytest tests/integration/test_automation_no_simulation.py -q || {
    echo "Tests failed - aborting"; exit 1;
}

# ------------------------------------------------------------------
# 4. Commit + push
# ------------------------------------------------------------------
git add \
    services/automation/agent_executor.py \
    services/automation/n8n_service.py \
    tests/integration/test_automation_no_simulation.py

git status --short
echo "--- review above and press Enter to commit, Ctrl+C to abort ---"
read -r _

git commit -m "fix(BACKLOG-003): remove simulated execution from services/automation

ISSUE-003 (BACKLOG.md): services/automation still shipped two fabricated
execution paths on the main code path:

- AgentExecutor._execute_node_standard returned a hard-coded
  {\"status\": 200, \"data\": \"simulated response\"} payload for every
  httpRequest node and 'Simulated output for {node.type}' for any other
  type, making the \"standard\" execution mode a pure no-op.
- CapibaraN8nAutomationService._execute_standard_n8n awaited
  asyncio.sleep(0.1) and returned {'message': 'Executed via n8n', ...}
  regardless of whether an n8n server was reachable, so the
  workflows_executed / successful_executions counters were meaningless.

Changes:

- _execute_node_standard now dispatches: 'set' nodes pass through their
  parameters, 'webhook' nodes expose input_data, 'httpRequest' nodes go
  through the new _execute_http_request_node helper that performs a real
  aiohttp request (with normalised n8n header encoding and timeout
  handling). Unknown node types are tagged status='unsupported' with a
  descriptive error instead of a fake success string.
- _execute_standard_n8n now creates the workflow in n8n via the existing
  _create_n8n_workflow helper and POSTs to
  /api/v1/workflows/{id}/execute through self._http_session. If the
  HTTP session is missing (aiohttp not installed or startup() was not
  awaited) it returns status='failed' with an explicit
  n8n_api_not_configured marker; registration failures surface as
  workflow_registration_failed. HTTP status codes drive the final
  ExecutionResult.status field.
- Added an AIOHTTP_AVAILABLE flag to services/automation/agent_executor
  so callers can detect whether real HTTP is viable.

Validation:

- 0 syntax errors across the three touched files.
- New tests/integration/test_automation_no_simulation.py (8 tests, all
  passing) asserts, via AST + source inspection, that the forbidden
  strings are gone, the helper exists and uses aiohttp, and the new
  failure markers are present."

git push -u origin fix/issue-backlog-003-automation-real-execution

echo
echo "Done. Open PR at:"
echo "  https://github.com/anachroni-co/capibaraGPT_v3/pull/new/fix/issue-backlog-003-automation-real-execution"
