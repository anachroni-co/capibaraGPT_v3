"""Integration surface tests for training/consensus (BACKLOG-006).

training/consensus is ~13k LOC with heavy optional dependencies (jax,
ray, etc.). Rather than importing every module (which would require the
full ML stack), these tests validate the **public API surface** via AST
inspection: class names, method names, and async vs sync kind. They are
enough to catch accidental renames/deletions that would silently break
downstream consumers.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONSENSUS_DIR = REPO_ROOT / "training" / "consensus"


def _module_tree(relpath: str) -> ast.Module:
    return ast.parse((CONSENSUS_DIR / relpath).read_text(encoding="utf-8"))


def _top_level_classes(tree: ast.Module) -> dict:
    return {
        node.name: node
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.ClassDef)
    }


def _methods(cls: ast.ClassDef) -> dict:
    out = {}
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
    return out


# ---------------------------------------------------------------------------
# Parseability of every consensus module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(CONSENSUS_DIR.glob("*.py")))
def test_consensus_module_parses(path: Path) -> None:
    """Every training/consensus module must still parse cleanly."""
    ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# meta_consensus_system.py - main entrypoint
# ---------------------------------------------------------------------------


def test_meta_consensus_system_exposes_main_classes():
    tree = _module_tree("meta_consensus_system.py")
    classes = _top_level_classes(tree)

    expected = {
        "ConsensusMode",
        "SystemState",
        "MetaConsensusConfig",
        "QueryContext",
        "ConsensusResult",
        "SystemMetrics",
        "MetaConsensusSystem",
    }
    missing = expected - classes.keys()
    assert not missing, f"missing classes in meta_consensus_system: {missing}"


def test_meta_consensus_system_core_methods():
    tree = _module_tree("meta_consensus_system.py")
    mcs = _top_level_classes(tree)["MetaConsensusSystem"]
    methods = _methods(mcs)

    # Core public entrypoints that other modules depend on.
    for name in ("__init__", "initialize", "process_query", "get_system_status"):
        assert name in methods, f"MetaConsensusSystem.{name} missing"

    # initialize + process_query must remain async (they are async in the
    # existing code and callers await them).
    assert isinstance(methods["initialize"], ast.AsyncFunctionDef), (
        "initialize must remain an async method"
    )
    assert isinstance(methods["process_query"], ast.AsyncFunctionDef), (
        "process_query must remain an async method"
    )


def test_meta_consensus_system_has_strategy_dispatchers():
    tree = _module_tree("meta_consensus_system.py")
    mcs = _top_level_classes(tree)["MetaConsensusSystem"]
    methods = _methods(mcs)
    for name in (
        "_execute_enhanced_consensus",
        "_execute_hybrid_routing",
        "_execute_unified_consensus",
    ):
        assert name in methods, f"strategy dispatcher {name} missing"
        assert isinstance(methods[name], ast.AsyncFunctionDef)


def test_meta_consensus_system_has_factory_functions():
    tree = _module_tree("meta_consensus_system.py")
    funcs = {
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "create_meta_consensus_system" in funcs
    assert "create_default_config" in funcs


# ---------------------------------------------------------------------------
# advance_meta_consensus_integration.py
# ---------------------------------------------------------------------------


def test_advance_integration_has_federated_consensus_entry():
    tree = _module_tree("advance_meta_consensus_integration.py")
    classes = _top_level_classes(tree)
    assert classes, "advance_meta_consensus_integration has no classes"

    # At least one class should expose _apply_federated_consensus (the
    # BACKLOG-002 fix lives there).
    found = any(
        "_apply_federated_consensus" in _methods(cls) for cls in classes.values()
    )
    assert found, "_apply_federated_consensus method missing"


# ---------------------------------------------------------------------------
# __init__.py re-exports
# ---------------------------------------------------------------------------


def test_consensus_package_init_parses():
    init = CONSENSUS_DIR / "__init__.py"
    assert init.exists()
    ast.parse(init.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# BACKLOG-002 regression guard (tie-in with earlier fix)
# ---------------------------------------------------------------------------


def test_no_mock_response_in_meta_consensus_hot_path():
    """The BACKLOG-002 fix must not silently regress."""
    src = (CONSENSUS_DIR / "meta_consensus_system.py").read_text(encoding="utf-8")
    # Strip docstrings so explanatory prose doesn't trigger false positives.
    import re
    live = re.sub(r'"""[\s\S]*?"""', "", src)
    live = re.sub(r"'''[\s\S]*?'''", "", live)
    assert '"mock_response"' not in live, (
        "mock_response literal resurfaced - BACKLOG-002 regression"
    )
