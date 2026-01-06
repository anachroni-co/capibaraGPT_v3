from __future__ import annotations

from typing import Any, Dict

try:
    from capibara.jax.numpy import jnp  # type: ignore
except Exception:
    try:
        import numpy as jnp  # type: ignore
    except Exception:
        jnp = None  # type: ignore

from capibara.core.modular_model import ModularCapibaraModel
from .constitutional_ai import ComprehensiveVerificationSystem


class VerificationAPI:
    """API unificada para verificación en Capibara-6."""

    def __init__(self, modular_model: ModularCapibaraModel):
        self.model = modular_model
        self.verification_system: ComprehensiveVerificationSystem = modular_model.verification_system  # type: ignore[attr-defined]

    def verify_text(self, text: str) -> Dict[str, Any]:
        embedding = jnp.ones((768,)) if jnp is not None else None
        return self.verification_system.verify_output(embedding, text)

    def verify_and_correct(self, text: str) -> Dict[str, Any]:
        embedding = jnp.ones((768,)) if jnp is not None else None
        verification = self.verification_system.verify_output(embedding, text)
        if verification.get("requires_correction", False):
            correction = self.verification_system.apply_corrections(embedding, verification)
            return {
                "original_text": text,
                "corrected_text": "Texto corregido",
                "verification": correction["final_verification"],
                "correction_applied": True,
            }
        return {
            "original_text": text,
            "corrected_text": text,
            "verification": verification,
            "correction_applied": False,
        }

    def get_safety_report(self) -> Dict[str, Any]:
        return self.verification_system.get_alignment_report()