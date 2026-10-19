"""Non-regression tests for BACKLOG ISSUE-004.

Verifies that the two production inference engines no longer ship the
fabricated parameter / forward-pass placeholders documented in ``BACKLOG.md``:

- ``inference/hybrid_inference_engine.py``:
    * ``_load_model_params`` used to fall back to
      ``{"dummy": jnp.array([1.0])}`` when no checkpoint file was found.
    * ``_compile_generation_function`` used to return logits made of
      ``jnp.ones((batch, seq, 32000))``.
- ``inference/engines/advanced_quantized_engine.py``:
    * ``_load_model_params`` built a full 12-layer transformer out of
      ``np.random.randn(...)`` arrays regardless of ``model_path``.
    * ``_get_memory_usage`` returned the hard-coded constant ``512.0`` MB.

All checks run on the source AST + regex inspection so the tests remain
usable on CI environments without torch / flax / psutil installed.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HYBRID = REPO_ROOT / "inference" / "hybrid_inference_engine.py"
ADVANCED = REPO_ROOT / "inference" / "engines" / "advanced_quantized_engine.py"


def _find_func(tree: ast.Module, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
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


@pytest.mark.parametrize("path", [HYBRID, ADVANCED])
def test_module_parses(path: Path) -> None:
    assert path.exists(), f"missing file: {path}"
    ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# hybrid_inference_engine.py
# ---------------------------------------------------------------------------


def test_hybrid_load_model_params_raises_without_checkpoint() -> None:
    src = HYBRID.read_text(encoding="utf-8")
    func = _find_func(ast.parse(src), "_load_model_params")
    body = _source_slice(src, func)
    live = _strip_docstrings(body)

    # Mock fallback must be gone in executable code.
    assert 'jnp.array([1.0])' not in live, (
        "dummy jnp.array([1.0]) fallback still present in _load_model_params"
    )
    assert '"dummy"' not in live, "'dummy' key fabrication still present"
    assert "Using mock parameters" not in live, (
        "mock-parameter log message still present"
    )
    # Must raise FileNotFoundError when nothing loads.
    assert "raise FileNotFoundError" in live, (
        "_load_model_params must raise FileNotFoundError when no checkpoint "
        "is available instead of returning synthetic params"
    )
    # Must try the real formats.
    assert "checkpoint.pkl" in body, "pickle path probe missing"
    assert "msgpack" in body.lower(), "flax msgpack probe missing"
    assert "orbax" in body.lower(), "orbax probe missing"


def test_hybrid_compile_generation_uses_real_module() -> None:
    src = HYBRID.read_text(encoding="utf-8")
    func = _find_func(ast.parse(src), "_compile_generation_function")
    body = _source_slice(src, func)
    live = _strip_docstrings(body)

    # The fake ones-logits tensor must be gone from executable code.
    assert "jnp.ones(" not in live, (
        "fake jnp.ones logits still produced by _compile_generation_function"
    )
    # Must consult a real attached module.
    assert 'getattr(self, "model_module"' in body, (
        "compile step should consult a real model_module attribute"
    )
    assert "module.apply(" in body, (
        "compile step should delegate the forward pass to module.apply(...)"
    )
    # When no module is attached the engine must refuse to generate.
    assert "self.compiled_generate = None" in body, (
        "compile step should leave compiled_generate = None when no real "
        "module is wired"
    )


def test_hybrid_generate_fails_fast_without_compiled_fn() -> None:
    src = HYBRID.read_text(encoding="utf-8")
    func = _find_func(ast.parse(src), "generate")
    body = _source_slice(src, func)

    assert "compiled_generate is None" in body, (
        "generate() must fail fast when no compiled function is available "
        "instead of crashing or producing fake tokens"
    )
    assert "BACKLOG-004" in body, (
        "generate() guard should reference BACKLOG-004 so future readers "
        "understand the removal"
    )


# ---------------------------------------------------------------------------
# advanced_quantized_engine.py
# ---------------------------------------------------------------------------


def test_advanced_load_model_params_no_random_weights() -> None:
    src = ADVANCED.read_text(encoding="utf-8")
    func = _find_func(ast.parse(src), "_load_model_params")
    body = _source_slice(src, func)
    live = _strip_docstrings(body)

    # Forbidden synthetic-weights identifiers.
    forbidden = [
        "np.random.randn(50000",
        "np.random.randn(768",
        "transformer_layer_",  # only inside the loop that fabricates params
    ]
    # transformer_layer_ may legitimately appear elsewhere; we only check the
    # _load_model_params body.
    assert "np.random.randn" not in live, (
        "_load_model_params still fabricates weights via np.random.randn"
    )
    assert "transformer_layer_" not in live, (
        "_load_model_params still hard-codes a 12-layer random transformer"
    )
    # Must raise when checkpoint is absent.
    assert "raise FileNotFoundError" in live
    # Must accept at least .npz / .safetensors / .pkl branches.
    assert '".npz"' in body
    assert '".safetensors"' in body
    assert '".pkl"' in body


def test_advanced_has_unflatten_helper() -> None:
    src = ADVANCED.read_text(encoding="utf-8")
    tree = ast.parse(src)
    # module-level helper
    assert any(
        isinstance(n, ast.FunctionDef) and n.name == "_unflatten_param_tree"
        for n in tree.body
    ), "_unflatten_param_tree module-level helper missing"


def test_advanced_get_memory_usage_uses_psutil() -> None:
    src = ADVANCED.read_text(encoding="utf-8")
    func = _find_func(ast.parse(src), "_get_memory_usage")
    body = _source_slice(src, func)
    live = _strip_docstrings(body)

    assert "return 512.0" not in live, (
        "_get_memory_usage still returns the hard-coded 512.0 MB placeholder"
    )
    assert "psutil" in body, (
        "_get_memory_usage should query real memory usage via psutil"
    )
    assert "memory_info().rss" in body, (
        "_get_memory_usage should read .memory_info().rss from psutil"
    )
    # Explicit sentinel when psutil is unavailable.
    assert "-1.0" in body, (
        "_get_memory_usage should return -1.0 when psutil is missing so "
        "callers can distinguish real measurements from placeholder values"
    )
