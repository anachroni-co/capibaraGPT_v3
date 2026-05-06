"""Non-regression tests for BACKLOG ISSUE-003.

These tests verify that ``services/automation`` no longer returns fabricated
success results from its "standard" execution paths. Specifically:

- ``AgentExecutor._execute_node_standard`` must not return the hard-coded
  ``"simulated response"`` / ``f"Simulated output for {node.type}"`` strings,
  and must delegate real HTTP requests to a dedicated helper.
- ``CapibaraN8nAutomationService._execute_standard_n8n`` must not fabricate
  success via ``await asyncio.sleep(0.1)`` + ``"Executed via n8n"``. When the
  HTTP session is missing it must surface an explicit
  ``n8n_api_not_configured`` marker; otherwise it must call the real n8n
  ``/api/v1/workflows/{id}/execute`` endpoint.

These checks run on the source AST instead of importing the module so they
stay portable across CI runners that do not have FastAPI / aiohttp / torch
installed.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_EXECUTOR = REPO_ROOT / "services" / "automation" / "agent_executor.py"
N8N_SERVICE = REPO_ROOT / "services" / "automation" / "n8n_service.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_func(tree: ast.Module, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    """Return the first (possibly nested) function with the given name."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in AST")


def _source_slice(src: str, node: ast.AST) -> str:
    lines = src.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def _strip_docstrings(src: str) -> str:
    stripped = re.sub(r'"""[\s\S]*?"""', "", src)
    stripped = re.sub(r"'''[\s\S]*?'''", "", stripped)
    return stripped


# ---------------------------------------------------------------------------
# Parseability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [AGENT_EXECUTOR, N8N_SERVICE])
def test_module_parses(path: Path) -> None:
    """Both touched modules must still parse cleanly."""
    assert path.exists(), f"missing file: {path}"
    ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# No forbidden simulation strings in executable code
# ---------------------------------------------------------------------------


_FORBIDDEN = (
    "simulated response",
    "Simulated output for",
    "Executed via n8n",
)


@pytest.mark.parametrize("path", [AGENT_EXECUTOR, N8N_SERVICE])
def test_no_live_simulation_strings(path: Path) -> None:
    """Forbidden placeholder strings must not appear outside docstrings."""
    src = path.read_text(encoding="utf-8")
    live = _strip_docstrings(src)
    offenders = [needle for needle in _FORBIDDEN if needle in live]
    assert not offenders, (
        f"{path.name}: live simulation string(s) still present: {offenders}"
    )


# ---------------------------------------------------------------------------
# AgentExecutor._execute_node_standard replacement logic
# ---------------------------------------------------------------------------


def test_execute_node_standard_has_real_dispatch() -> None:
    """``_execute_node_standard`` must dispatch to real handlers or emit an
    explicit ``unsupported`` marker - never a fake ``Simulated output``."""
    src = AGENT_EXECUTOR.read_text(encoding="utf-8")
    tree = ast.parse(src)
    func = _find_func(tree, "_execute_node_standard")
    body = _source_slice(src, func)

    # Delegates real HTTP to a named helper so the test also catches accidental
    # revert to an inline simulation.
    assert "_execute_http_request_node" in body, (
        "standard-mode executor should delegate httpRequest nodes to "
        "_execute_http_request_node"
    )
    assert '"n8n-nodes-base.set"' in body, "set node branch lost"
    assert '"n8n-nodes-base.webhook"' in body, "webhook node branch lost"
    assert '"unsupported"' in body, (
        "unknown node types must be tagged status='unsupported' instead of "
        "returning a fake 'Simulated output' string"
    )

    # No live reference to the old fake payload.
    stripped = _strip_docstrings(body)
    assert '"simulated response"' not in stripped
    assert 'f"Simulated output for' not in stripped


def test_http_request_helper_exists_and_uses_aiohttp() -> None:
    """The HTTP-request helper must be defined and actually use aiohttp."""
    src = AGENT_EXECUTOR.read_text(encoding="utf-8")
    tree = ast.parse(src)
    func = _find_func(tree, "_execute_http_request_node")
    body = _source_slice(src, func)

    assert "aiohttp.ClientSession" in body, (
        "_execute_http_request_node should open a real aiohttp ClientSession"
    )
    assert "session.request(" in body, (
        "_execute_http_request_node should issue a real HTTP request"
    )
    assert "AIOHTTP_AVAILABLE" in body, (
        "_execute_http_request_node should guard the aiohttp-missing case"
    )

    # Module must expose the feature flag (it is set inside a try/except at
    # import time, so the line is indented by four spaces).
    assert re.search(r"^\s*AIOHTTP_AVAILABLE\s*=\s*True", src, re.MULTILINE), (
        "agent_executor.py should set AIOHTTP_AVAILABLE = True when aiohttp "
        "imports cleanly so callers can decide whether real HTTP is viable"
    )


# ---------------------------------------------------------------------------
# CapibaraN8nAutomationService._execute_standard_n8n replacement logic
# ---------------------------------------------------------------------------


def test_execute_standard_n8n_no_fake_sleep_success() -> None:
    """``_execute_standard_n8n`` must no longer return a fake success built
    around ``asyncio.sleep(0.1)`` + ``Executed via n8n``."""
    src = N8N_SERVICE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    func = _find_func(tree, "_execute_standard_n8n")
    body = _source_slice(src, func)
    # Strip docstrings so we only inspect executable code - the new
    # implementation explains what it replaced and that prose may legitimately
    # reference the old identifiers.
    live = _strip_docstrings(body)

    assert "asyncio.sleep(0.1)" not in live, (
        "fake sleep-based success path still present in _execute_standard_n8n"
    )
    assert "Executed via n8n" not in live, (
        "hard-coded 'Executed via n8n' message still present"
    )


def test_execute_standard_n8n_hits_real_api() -> None:
    """``_execute_standard_n8n`` must call the real n8n execute endpoint and
    expose the ``n8n_api_not_configured`` marker when it cannot."""
    src = N8N_SERVICE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    func = _find_func(tree, "_execute_standard_n8n")
    body = _source_slice(src, func)

    # Real API path: must build an /execute URL and use the aiohttp session.
    assert "/execute" in body, (
        "_execute_standard_n8n should target the n8n /execute endpoint"
    )
    assert "self._http_session.post" in body, (
        "_execute_standard_n8n should POST via the aiohttp session"
    )

    # Fail-fast markers.
    assert '"n8n_api_not_configured"' in body, (
        "missing explicit n8n_api_not_configured marker when HTTP session is "
        "unavailable"
    )
    assert '"workflow_registration_failed"' in body, (
        "missing workflow_registration_failed marker when create step fails"
    )

    # Status must be derived from the real HTTP response, not hard-coded.
    assert "response.status" in body, (
        "status should be derived from response.status, not constants"
    )
