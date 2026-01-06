from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import time

try:
    # Prefer the project's JAX shim
    from capibara.jax.numpy import jnp  # type: ignore
    JAX_AVAILABLE = True
except Exception:
    try:
        import numpy as jnp  # type: ignore
        JAX_AVAILABLE = False
    except Exception:
        jnp = None  # type: ignore
        JAX_AVAILABLE = False


@dataclass
class AlignmentConfig:
    enable_constitutional_ai: bool = True
    enable_real_time_verification: bool = True
    enable_bias_detection: bool = True
    enable_harm_prevention: bool = True

    verification_threshold: float = 0.85
    safety_threshold: float = 0.90
    bias_threshold: float = 0.50
    harm_threshold: float = 0.30

    max_correction_attempts: int = 3
    enable_self_correction: bool = True
    log_all_interventions: bool = True

    helpfulness_weight: float = 1.0
    harmlessness_weight: float = 1.5
    honesty_weight: float = 1.2
    respect_weight: float = 1.3

    use_middleware: bool = True
    verify_inputs: bool = True
    verify_outputs: bool = True
    enable_api: bool = True


class ComprehensiveVerificationSystem:
    """Lightweight Constitutional-AI inspired verifier.

    This is a deterministic, dependency-light implementation that scores text/embeddings
    and returns a safety verdict suitable for integration tests and basic safeguards.
    """

    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config: AlignmentConfig = config or AlignmentConfig()
        self._stats: Dict[str, Any] = {
            "total_verifications": 0,
            "corrections_applied": 0,
            "last_verification": None,
            "start_time": time.time(),
        }

    def _to_array(self, embedding_like: Any):
        if jnp is None:
            return None
        try:
            return jnp.ones((768,)) if embedding_like is None else jnp.asarray(embedding_like)
        except Exception:
            # Fallback shape
            try:
                return jnp.ones((768,))
            except Exception:
                return None

    def _analyze_text(self, text: Optional[str]) -> Dict[str, float]:
        if not text:
            return {"bias": 0.05, "harm": 0.05}
        lowered = text.lower()
        # Very simple keyword-based heuristic signals
        bias_keywords = ["stereotype", "bias", "racist", "sexist", "prejudice", "discrim"]
        harm_keywords = [
            "kill", "attack", "weapon", "bomb", "harm", "suicide", "self-harm",
            "abuse", "violence", "hate", "terror", "bully", "harass", "threat"
        ]
        bias_score = 0.05 + 0.15 * sum(1 for k in bias_keywords if k in lowered)
        harm_score = 0.05 + 0.20 * sum(1 for k in harm_keywords if k in lowered)
        # Clamp 0..1
        bias_score = max(0.0, min(1.0, bias_score))
        harm_score = max(0.0, min(1.0, harm_score))
        return {"bias": bias_score, "harm": harm_score}

    def _compute_overall(self, bias: float, harm: float) -> float:
        # Weighted aggregation favoring harmlessness
        harmless_component = max(0.0, 1.0 - harm)
        unbiased_component = max(0.0, 1.0 - bias)
        weighted = (
            self.config.helpfulness_weight * 0.3
            + self.config.honesty_weight * 0.2
            + self.config.respect_weight * 0.2
            + self.config.harmlessness_weight * (0.15 * harmless_component + 0.15 * unbiased_component)
        )
        # Normalize to 0..1 proxy
        max_weight = (
            self.config.helpfulness_weight * 0.3
            + self.config.honesty_weight * 0.2
            + self.config.respect_weight * 0.2
            + self.config.harmlessness_weight * 0.3
        )
        score = weighted / max_weight if max_weight > 0 else 1.0
        # Penalize directly by detected harms/bias
        score = score * (0.9 + 0.1 * harmless_component) * (0.9 + 0.1 * unbiased_component)
        return max(0.0, min(1.0, score))

    def _safety_level(self, overall: float, bias: float, harm: float) -> str:
        if harm >= 0.80:
            return "blocked"
        if harm >= 0.50 or bias >= 0.70:
            return "warning"
        if overall < self.config.verification_threshold or (harm >= self.config.harm_threshold):
            return "caution"
        return "safe"

    def verify_output(self, embedding_like: Any, text: Optional[str] = None) -> Dict[str, Any]:
        if not (self.config.enable_constitutional_ai and self.config.verify_outputs):
            return {
                "overall_score": 1.0,
                "safety_level": "safe",
                "requires_correction": False,
                "bias_score": 0.0,
                "harm_score": 0.0,
                "enabled": False,
            }

        _ = self._to_array(embedding_like)
        text_signals = self._analyze_text(text)
        bias = text_signals["bias"] if self.config.enable_bias_detection else 0.0
        harm = text_signals["harm"] if self.config.enable_harm_prevention else 0.0
        overall = self._compute_overall(bias, harm)
        level = self._safety_level(overall, bias, harm)

        requires_corr = overall < self.config.safety_threshold or harm >= self.config.harm_threshold

        result = {
            "overall_score": overall,
            "safety_level": level,
            "requires_correction": requires_corr and self.config.enable_self_correction,
            "bias_score": bias,
            "harm_score": harm,
            "principles": {
                "helpfulness": self.config.helpfulness_weight,
                "harmlessness": self.config.harmlessness_weight,
                "honesty": self.config.honesty_weight,
                "respect": self.config.respect_weight,
            },
            "enabled": True,
        }

        self._stats["total_verifications"] += 1
        self._stats["last_verification"] = result
        return result

    def apply_corrections(self, embedding_like: Any, verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a correction loop and return a final verification snapshot."""
        attempts = 0
        current = dict(verification_result)
        while attempts < self.config.max_correction_attempts and current.get("requires_correction", False):
            # Simulated improvement: nudge scores to safer region deterministically
            improved_overall = min(1.0, current.get("overall_score", 0.0) + 0.07)
            reduced_harm = max(0.0, current.get("harm_score", 0.0) - 0.05)
            reduced_bias = max(0.0, current.get("bias_score", 0.0) - 0.03)
            level = self._safety_level(improved_overall, reduced_bias, reduced_harm)
            current.update({
                "overall_score": improved_overall,
                "harm_score": reduced_harm,
                "bias_score": reduced_bias,
                "safety_level": level,
                "requires_correction": improved_overall < self.config.safety_threshold or reduced_harm >= self.config.harm_threshold,
            })
            attempts += 1

        self._stats["corrections_applied"] += 1 if attempts > 0 else 0
        return {
            "final_verification": current,
            "attempts": attempts,
        }

    def get_alignment_report(self) -> Dict[str, Any]:
        uptime = time.time() - self._stats["start_time"]
        return {
            "uptime_sec": uptime,
            "stats": {
                "total_verifications": self._stats["total_verifications"],
                "corrections_applied": self._stats["corrections_applied"],
            },
            "last_verification": self._stats["last_verification"],
            "thresholds": {
                "verification_threshold": self.config.verification_threshold,
                "safety_threshold": self.config.safety_threshold,
                "bias_threshold": self.config.bias_threshold,
                "harm_threshold": self.config.harm_threshold,
            },
            "features": {
                "constitutional_ai": self.config.enable_constitutional_ai,
                "real_time": self.config.enable_real_time_verification,
                "bias_detection": self.config.enable_bias_detection,
                "harm_prevention": self.config.enable_harm_prevention,
            },
        }


def create_verification_system(config: Optional[AlignmentConfig] = None) -> ComprehensiveVerificationSystem:
    return ComprehensiveVerificationSystem(config or AlignmentConfig())