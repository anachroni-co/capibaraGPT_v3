"""Centralized lazy-import helpers for backend modules.

Each helper ensures that the required library is imported exactly once
(thread-safe via the GIL + global guard) and raises a clear error if the
library is missing.
"""

from __future__ import annotations

import threading
import subprocess
import sys
from typing import Any, Dict

_lock = threading.Lock()
_cache: Dict[str, Any] = {}


def _lazy_load(key: str, imports: Dict[str, str], install_hint: str) -> Dict[str, Any]:
    """Import *imports* lazily, caching the result under *key*.

    Parameters
    ----------
    key:
        Cache key (e.g. ``"torch"``, ``"jax"``).
    imports:
        Mapping ``{local_name: module_path}`` – each value is passed to
        ``importlib.import_module``.
    install_hint:
        Human-readable install command shown on failure.

    Returns
    -------
    dict mapping *local_name* → imported module.
    """
    if key in _cache:
        return _cache[key]
    with _lock:
        if key in _cache:  # double-check
            return _cache[key]
        import importlib
        if key == "torch" and not _torch_import_works():
            raise ImportError(f"{key} is not importable on this system.")
        result: Dict[str, Any] = {}
        try:
            for name, path in imports.items():
                result[name] = importlib.import_module(path)
        except Exception as e:
            raise ImportError(
                f"{key} is not installed. Install with: {install_hint}"
            ) from e
        _cache[key] = result
        return result


def _torch_import_works() -> bool:
    """Check torch import in a subprocess to avoid hard crashes."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


# ── public helpers ──────────────────────────────────────────────

def ensure_torch():
    """Lazily import PyTorch modules. Returns (torch, nn, F)."""
    mods = _lazy_load("torch", {
        "torch": "torch",
        "nn": "torch.nn",
        "F": "torch.nn.functional",
    }, "pip install torch")
    return mods["torch"], mods["nn"], mods["F"]


def ensure_jax():
    """Lazily import JAX modules. Returns (jax, jnp, flax, optax)."""
    mods = _lazy_load("jax", {
        "jax": "jax",
        "jnp": "jax.numpy",
        "flax": "flax",
        "optax": "optax",
    }, "pip install jax[tpu]")
    return mods["jax"], mods["jnp"], mods["flax"], mods["optax"]
