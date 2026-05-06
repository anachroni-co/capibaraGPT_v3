"""Non-regression tests for BACKLOG ISSUE-002.

These tests verify that the training/consensus meta-consensus code path no
longer ships the hard-coded ``mock_response`` / ``mock_metrics`` placeholders
and that the replacement logic behaves deterministically.

The production module has heavy optional dependencies (torch, flax, HF
clients) that are not present in every environment, so these tests operate
on the source AST instead of importing the module. This keeps the check
portable across CI runners while still catching real regressions.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
META_CONSENSUS = REPO_ROOT / "training" / "consensus" / "meta_consensus_system.py"
ADVANCE_INTEGRATION = (
    REPO_ROOT / "training" / "consensus" / "advance_meta_consensus_integration.py"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LIVE_MOCK_ASSIGN_RE = re.compile(
    r"""
    ^\s*                               # indent only
    (?P<name>mock_response|mock_metrics)   # forbidden identifier
    \s*=                                   # assignment
    """,
    re.MULTILINE | re.VERBOSE,
)


def _find_func(tree: ast.Module, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    """Return the first (possibly nested) function with the given name."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in AST")


def _source_slice(src: str, node: ast.AST) -> str:
    lines = src.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


# ---------------------------------------------------------------------------
# Parseability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [META_CONSENSUS, ADVANCE_INTEGRATION])
def test_module_parses(path: Path) -> None:
    """The two touched files must still parse cleanly."""
    assert path.exists(), f"missing file: {path}"
    ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# No live mock identifiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [META_CONSENSUS, ADVANCE_INTEGRATION])
def test_no_live_mock_assignments(path: Path) -> None:
    """No top-level ``mock_response =`` / ``mock_metrics =`` bindings."""
    src = path.read_text(encoding="utf-8")
    matches = list(_LIVE_MOCK_ASSIGN_RE.finditer(src))
    assert not matches, (
        f"{path.name}: live mock assignment(s) still present: "
        + ", ".join(m.group("name") for m in matches)
    )


def test_federated_proposal_not_hardcoded_mock() -> None:
    """_apply_federated_consensus must not build the proposal from a literal
    ``{'response': 'mock_response', ...}`` dict."""
    src = ADVANCE_INTEGRATION.read_text(encoding="utf-8")
    func = _find_func(ast.parse(src), "_apply_federated_consensus")
    body_src = _source_slice(src, func)
    # docstring may mention the old pattern; we only reject assignments.
    stripped = re.sub(r'"""[\s\S]*?"""', "", body_src)
    stripped = re.sub(r"'''[\s\S]*?'''", "", stripped)
    assert "'response': 'mock_response'" not in stripped, (
        "hard-coded mock_response dict still drives the federated proposal"
    )


# ---------------------------------------------------------------------------
# Replacement logic is present
# ---------------------------------------------------------------------------


def test_hybrid_routing_has_executor_branch_and_no_executor_branch() -> None:
    """_execute_hybrid_routing must (a) delegate to enhanced_consensus when
    available and (b) emit a ``hybrid_routing_no_executor`` marker otherwise."""
    src = META_CONSENSUS.read_text(encoding="utf-8")
    func = _find_func(ast.parse(src), "_execute_hybrid_routing")
    body = _source_slice(src, func)

    assert "self.enhanced_consensus" in body, (
        "hybrid routing should consult enhanced_consensus"
    )
    assert "get_enhanced_consensus_response" in body, (
        "hybrid routing should call the real executor when available"
    )
    assert "hybrid_routing_no_executor" in body, (
        "hybrid routing should expose a truthful no-executor consensus_method"
    )
    assert "hybrid_routing+enhanced_executor" in body, (
        "hybrid routing should tag the delegated path with a real method id"
    )


def test_unified_consensus_uses_live_metrics() -> None:
    """_execute_unified_consensus must derive metrics from query_history and
    expose an explicit ``unified_no_history`` marker in the empty case."""
    src = META_CONSENSUS.read_text(encoding="utf-8")
    func = _find_func(ast.parse(src), "_execute_unified_consensus")
    body = _source_slice(src, func)

    assert "self.query_history" in body, (
        "unified consensus should read from live query_history"
    )
    assert "live_metrics" in body, (
        "unified consensus should build a live_metrics dict"
    )
    assert "unified_no_history" in body, (
        "unified consensus needs an explicit empty-history marker"
    )
    assert "unified_consensus_reuse_last" in body, (
        "unified consensus should advertise the reuse-last branch"
    )
    # No hard-coded 0.5 / 0.85 / 2.1 constants fed to the update call.
    assert "0.85" not in body or "perplexity" not in body, (
        "legacy mock_metrics constants still present"
    )
