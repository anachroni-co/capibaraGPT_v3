"""
JAX compat module (portable stubs).

Proporciona utilidades mínimas requeridas por capibara.jax.nn
sin depender de JAX real.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

# Intentar usar numpy real; si no está, usar el submódulo interno capibara.jax.numpy
try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - environment without numpy
    try:
        from . import numpy as _np  # type: ignore
    except Exception:
        # Minimal fallback without numpy
        class _MiniNP:  # type: ignore
            def array(self, obj, dtype=None):
                return obj
            def asarray(self, obj, dtype=None):
                return obj
            def zeros(self, shape):
                return [0 for _ in range(shape if isinstance(shape, int) else shape[0])]
            def ones(self, shape):
                return [1 for _ in range(shape if isinstance(shape, int) else shape[0])]
            def sum(self, a, axis=None):
                return sum(a) if isinstance(a, (list, tuple)) else a
            def mean(self, a, axis=None):
                return (sum(a) / len(a)) if isinstance(a, (list, tuple)) and a else a
            def tanh(self, x):
                return x
        _np = _MiniNP()  # type: ignore


def get_jax() -> Any:
    """Returns a jax stub with minimal API usada por nn (random.bernoulli)."""
    def _bernoulli(rng, p, shape):  # rng ignorado en stub
        try:
            import random as _random
            return [[1 if _random.random() < p else 0 for _ in range(shape[-1])]]
        except Exception:
            return 1

    return SimpleNamespace(random=SimpleNamespace(bernoulli=_bernoulli))


def get_numpy() -> Any:
    """Returns real numpy or internal submodule capibara.jax.numpy como sustituto de jnp."""
    return _np
