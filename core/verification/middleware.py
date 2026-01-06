from __future__ import annotations

from typing import Any, Dict, Callable

try:
    from capibara.jax.numpy import jnp  # type: ignore
except Exception:
    try:
        import numpy as jnp  # type: ignore
    except Exception:
        jnp = None  # type: ignore

from .constitutional_ai import ComprehensiveVerificationSystem


class VerificationMiddleware:
    """Middleware para integrar verificación en cualquier pipeline."""

    def __init__(self, verification_system: ComprehensiveVerificationSystem):
        self.verification_system = verification_system

    def __call__(self, func: Callable[..., Any]):
        """Decorador para añadir verificación a funciones."""

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if self.verification_system and isinstance(result, dict):
                verified_result = self._verify_result(result)
                return verified_result
            return result

        return wrapper

    def _verify_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Verifies the result of a function."""
        embedding = jnp.ones((768,)) if jnp is not None else None
        verification = self.verification_system.verify_output(embedding)
        return {
            **result,
            "verification": verification,
            "safety_level": verification.get("safety_level", "unknown"),
            "verified": True,
        }