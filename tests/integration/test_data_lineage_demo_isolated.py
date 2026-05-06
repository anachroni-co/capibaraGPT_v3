"""Non-regression tests for BACKLOG ISSUE-005.

Verifies that ``training/data_lineage`` no longer ships runtime mocks nor
leaks module-level logging configuration:

1. ``InferenceSafeParameterController.create_dataset_mask_safe`` must NOT
   fabricate a lineage of 1/3 of all parameters when a dataset has no
   registered lineage. It must log a warning and return an empty
   ``dataset_params`` list.
2. ``training/data_lineage/demo_traceability_system.py`` must be demo-only:
   - ``logging.basicConfig`` must not be invoked at module scope (it leaks
     into every process that imports the package).
   - The demo CLI must be gated by the ``CAPIBARA_DATA_LINEAGE_DEMO=1``
     environment variable.
   - The demo mock class must be private (``_DemoMockModel``) so
     nothing outside the demo imports it by accident.

These checks run on the source AST and do not import the modules, so they
remain portable to CI runners lacking jax / aiohttp / torch.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ISPC = REPO_ROOT / "training" / "data_lineage" / "inference_safe_parameter_controller.py"
DTS = REPO_ROOT / "training" / "data_lineage" / "demo_traceability_system.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError("function {0!r} not found".format(name))


def _source_slice(src, node):
    lines = src.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def _strip_docstrings(src):
    stripped = re.sub(r'"""[\s\S]*?"""', "", src)
    stripped = re.sub(r"'''[\s\S]*?'''", "", stripped)
    return stripped


# ---------------------------------------------------------------------------
# Parseability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [ISPC, DTS])
def test_module_parses(path):
    assert path.exists(), "missing file: {0}".format(path)
    ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. inference_safe_parameter_controller: no mock lineage fallback
# ---------------------------------------------------------------------------


def test_create_dataset_mask_safe_has_no_mock_lineage():
    src = ISPC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    func = _find_func(tree, "create_dataset_mask_safe")
    body = _source_slice(src, func)
    live = _strip_docstrings(body)

    # Old forbidden comment and code.
    assert "Create mock lineage for testing" not in live, (
        "old 'Create mock lineage for testing' comment still present"
    )
    assert "all_params[:len(all_params)//3]" not in live, (
        "fabricated 1/3-of-parameters lineage still present"
    )

    # New behaviour: warn + empty list + BACKLOG-005 marker.
    assert "BACKLOG-005" in body, (
        "create_dataset_mask_safe should reference BACKLOG-005 in the "
        "replacement comment"
    )
    assert "logger.warning" in live, (
        "missing dataset should emit a logger.warning"
    )
    assert "dataset_params = []" in live, (
        "missing dataset should yield an empty dataset_params list"
    )


# ---------------------------------------------------------------------------
# 2. demo_traceability_system: demo-only isolation
# ---------------------------------------------------------------------------


def test_demo_no_module_level_basicConfig():
    """logging.basicConfig must not run at import time."""
    src = DTS.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Walk top-level statements and reject any direct call to
    # logging.basicConfig at module scope.
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "logging"
                and call.func.attr == "basicConfig"
            ):
                raise AssertionError(
                    "logging.basicConfig must not be called at module scope"
                )


def test_demo_has_configure_helper():
    src = DTS.read_text(encoding="utf-8")
    tree = ast.parse(src)
    func = _find_func(tree, "_configure_demo_logging")
    body = _source_slice(src, func)
    assert "logging.basicConfig" in body, (
        "_configure_demo_logging should call logging.basicConfig"
    )


def test_demo_cli_is_env_gated():
    src = DTS.read_text(encoding="utf-8")
    # Grab everything after the __main__ guard.
    marker = 'if __name__ == "__main__":'
    assert marker in src, "demo missing __main__ guard"
    tail = src.split(marker, 1)[1]

    assert "CAPIBARA_DATA_LINEAGE_DEMO" in tail, (
        "demo CLI must be gated by CAPIBARA_DATA_LINEAGE_DEMO env var"
    )
    assert "_configure_demo_logging()" in tail, (
        "demo CLI must configure logging before running"
    )
    assert "sys.exit" in tail, (
        "demo CLI must exit when opt-in env var is missing"
    )


def test_demo_mock_model_is_private():
    src = DTS.read_text(encoding="utf-8")
    tree = ast.parse(src)

    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }
    assert "_DemoMockModel" in names, (
        "demo should expose a private _DemoMockModel class (BACKLOG-005)"
    )
    assert "MockModel" not in names, (
        "public 'MockModel' class should be renamed to _DemoMockModel"
    )

    # No stray references to the old public name (word-boundary check so
    # _DemoMockModel( is not a false positive).
    live = _strip_docstrings(src)
    assert not re.search(r"(?<![A-Za-z0-9_])MockModel\(", live), (
        "live code still instantiates the old public MockModel"
    )
