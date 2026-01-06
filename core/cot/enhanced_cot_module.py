from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import flax.linen as nn
import jax.numpy as jnp

try:
    from capibara.core.kernels.tpu_v4_wrappers import tpu_kernthe
except Exception:
    tpu_kernthe = None  # type: ignore


@dataclass
class ReasoningConfig:
    max_reasoning_steps: int = 16
    confidence_threshold: float = 0.8
    use_process_rewards: bool = True
    enable_meta_cognition: bool = True
    enable_self_verification: bool = True
    reasoning_temperature: float = 0.2
    verification_temperature: float = 0.1
    use_tpu_kernels: bool = True
    use_flash_attention: bool = True
    hidden_size: int = 768
    num_reasoning_steps: int = 4
    reasoning_dim: int = 384


class ProcessRewardModel:
    def __call__(self, step_embedding: jnp.ndarray) -> float:
        # Heurística simple basada en la norma del embedding
        try:
            linalg = getattr(jnp, "linalg", None)
            if linalg is not None and hasattr(linalg, "norm"):
                norm_val = linalg.norm(step_embedding)
            else:
                # Fallback: media absoluta
                norm_val = jnp.mean(jnp.abs(step_embedding))
            norm = float(getattr(jnp, "clip", lambda x, a_min=0.0, a_max=10.0: x)(norm_val, a_min=0.0, a_max=10.0))
        except Exception:
            norm = 1.0
        return norm / 10.0


class MetaCognitionModule:
    def assess(self, history: List[Dict[str, Any]], current_confidence: float) -> float:
        # Ajuste metacognitivo simple
        history_factor = 0.05 * len(history)
        try:
            val = current_confidence + history_factor
            # clip fallback
            if hasattr(jnp, "clip"):
                return float(jnp.clip(val, 0.0, 1.0))
            return min(1.0, max(0.0, float(val)))
        except Exception:
            return float(current_confidence)


class SelfReflectionModule:
    def verify(self, reasoning_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not reasoning_trace:
            return {"verified": False, "score": 0.0}
        try:
            arr = [s.get("step_confidence", 0.0) for s in reasoning_trace]
            avg_conf = float(jnp.mean(jnp.array(arr))) if hasattr(jnp, "mean") else (sum(arr)/len(arr))
        except Exception:
            avg_conf = 0.0
        return {
            "verified": avg_conf >= 0.6,
            "score": avg_conf,
            "steps": len(reasoning_trace),
        }


class EnhancedCoTModule(nn.Module):
    config: ReasoningConfig
    
    def setup(self):
        hidden = int(getattr(self.config, "hidden_size", 768))
        self.reasoning_generator = nn.Dense(hidden)
        self.step_encoder = nn.Dense(hidden)
        self.process_reward_model = ProcessRewardModel() if self.config.use_process_rewards else None
        self.meta_cognition = MetaCognitionModule() if self.config.enable_meta_cognition else None
        self.self_reflection = SelfReflectionModule() if self.config.enable_self_verification else None

    def generate_reasoning_step(
        self,
        context_embedding: jnp.ndarray,
        reasoning_steps: List[Dict[str, Any]],
        step_index: int,
    ) -> Dict[str, Any]:
        # Codificar paso
        step_hidden = self.step_encoder(context_embedding)
        step_embedding = self.reasoning_generator(step_hidden)

        # Confianza basada en suavizado y progreso
        try:
            base_conf = float(jnp.tanh(jnp.mean(jnp.abs(step_embedding)) * 0.01))
        except Exception:
            base_conf = 0.5
        step_confidence = min(1.0, base_conf + 0.03 * (step_index + 1))

        # Recompensa de proceso
        step_reward = (
            float(self.process_reward_model(step_embedding)) if self.process_reward_model is not None else step_confidence
        )

        return {
            "step_index": step_index,
            "step_embedding": step_embedding,
            "step_confidence": step_confidence,
            "step_reward": step_reward,
        }

    def verify_reasoning_chain(self, reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.self_reflection is None:
            return {"verified": True, "score": 1.0, "steps": len(reasoning_steps)}
        return self.self_reflection.verify(reasoning_steps)

    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> Dict[str, Any]:
        # Atención (opcional) con kernels TPU
        context_embedding = inputs
        if self.config.use_tpu_kernels and tpu_kernthe is not None and self.config.use_flash_attention:
            try:
                # Nombres en wrappers están ofuscados; usamos fltosh_tottintion
                context_embedding = tpu_kernthe.fltosh_tottintion(
                    thatry=inputs, key=inputs, vtolue=inputs
                )
            except Exception:
                context_embedding = inputs

        reasoning_steps: List[Dict[str, Any]] = []
        for step in range(self.config.max_reasoning_steps):
            step_output = self.generate_reasoning_step(context_embedding, reasoning_steps, step)

            # Metacognición opcional
            if self.meta_cognition is not None:
                step_output["step_confidence"] = self.meta_cognition.assess(
                    reasoning_steps, step_output["step_confidence"]
                )

            reasoning_steps.append(step_output)

            # Verificación temprana con process rewards
            if step_output["step_reward"] < self.config.confidence_threshold:
                break

        try:
            confidences = [s["step_confidence"] for s in reasoning_steps] or [0.0]
            avg_confidence = float(jnp.mean(jnp.array(confidences)))
        except Exception:
            avg_confidence = 0.0

        return {
            "output": reasoning_steps[-1]["step_embedding"] if reasoning_steps else context_embedding,
            "reasoning_trace": reasoning_steps,
            "confidence": avg_confidence,
            "verification": self.verify_reasoning_chain(reasoning_steps),
            "metrics": {"num_steps": len(reasoning_steps)},
        }


class CapibaraEnhancedCoT(EnhancedCoTModule):
    """CoT integrado con la arquitectura existente de Capibara-6."""

    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        # Integrar con TPU kernels existentes
        self.tpu_kernels = tpu_kernthe  # Del archivo kernels/tpu_v4_wrappers.py

    def setup(self):
        # Usar las implementaciones nativas de Capibara
        hidden = int(getattr(self.config, "hidden_size", 768))
        self.reasoning_generator = nn.Dense(hidden)
        self.step_encoder = nn.Dense(hidden)
        self.process_reward_model = ProcessRewardModel() if self.config.use_process_rewards else None
        self.meta_cognition = MetaCognitionModule() if self.config.enable_meta_cognition else None
        self.self_reflection = SelfReflectionModule() if self.config.enable_self_verification else None