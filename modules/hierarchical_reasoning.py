from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalReasoningConfig:
    h_dim: int = 512
    l_dim: int = 256
    max_iterations: int = 100
    convergence_threshold: float = 0.01
    tpu_cores_assigned: int = 3
    use_with_consensus: bool = True
    weight_in_consensus: float = 3.0
    override_threshold: float = 0.8

    keywords_math: Tuple[str, ...] = (
        "calcular", "resolver", "ecuación", "integral", "derivada", "límite",
        "matriz", "suma", "producto", "probabilidad", "combinatoria",
    )
    keywords_reasoning: Tuple[str, ...] = (
        "deducir", "inferir", "lógica", "premisa", "conclusión", "argumento",
        "razonar", "hipótesis", "demostrar", "prueba",
    )
    keywords_planning: Tuple[str, ...] = (
        "planificar", "estrategia", "pasos", "optimizar", "cronograma",
        "objetivos", "tareas", "secuencia",
    )


class HierarchicalReasoningModule:
    """Simple hierarchical reasoning scaffold with abstract (H) and detailed (L) phases.

    This is a light, deterministic scaffold designed to be integrated quickly
    and replaced by a learned model later. It provides:
      - task detection
      - iterative H/L reasoning loop with convergence heuristic
      - confidence and explanation
    """

    def __init__(self, config: Optional[HierarchicalReasoningConfig] = None):
        self.config = config or HierarchicalReasoningConfig()

    def is_reasoning_task(self, prompt: str) -> bool:
        text = prompt.lower()
        return any(k in text for k in self.config.keywords_math) \
            or any(k in text for k in self.config.keywords_reasoning) \
            or any(k in text for k in self.config.keywords_planning)

    def process(self, prompt: str) -> Dict[str, Any]:
        """Run a simple hierarchical reasoning loop.

        Returns a dict with fields: {steps, plan, converged, iterations, confidence, response}
        """
        plan = self._phase_h_abstract_plan(prompt)
        steps: List[str] = []
        score = 0.0
        converged = False

        for iteration in range(1, self.config.max_iterations + 1):
            step_text, delta = self._phase_l_detailed_step(prompt, plan, iteration)
            steps.append(step_text)
            score += max(0.0, delta)

            if delta < self.config.convergence_threshold:
                converged = True
                break

        confidence = max(0.0, min(1.0, 0.5 + 0.5 * (score / max(1.0, float(self.config.max_iterations)))))
        response = self._compose_response(plan, steps)
        return {
            "plan": plan,
            "steps": steps,
            "converged": converged,
            "iterations": iteration,
            "confidence": confidence,
            "response": response,
        }

    def _phase_h_abstract_plan(self, prompt: str) -> List[str]:
        text = prompt.lower()
        plan: List[str] = []
        if any(k in text for k in self.config.keywords_math):
            plan = [
                "Identificar variables y lo que se pide",
                "Descomponer el problema en subpasos matemáticos",
                "Aplicar reglas y teoremas relevantes",
                "Simplificar y verificar coherencia",
            ]
        elif any(k in text for k in self.config.keywords_planning):
            plan = [
                "Definir objetivo y restricciones",
                "Listar recursos y riesgos",
                "Secuenciar pasos con dependencias",
                "Evaluar métricas de éxito y ajustar",
            ]
        else:
            plan = [
                "Clarificar la premisa",
                "Enumerar hipótesis y alternativas",
                "Evaluar implicaciones lógicas",
                "Llegar a una conclusión justificada",
            ]
        return plan

    def _phase_l_detailed_step(self, prompt: str, plan: List[str], iteration: int) -> Tuple[str, float]:
        # Simple deterministic refinement signal
        base = 1.0 / (iteration + 2.0)
        step_idx = min(iteration - 1, len(plan) - 1) if plan else 0
        step_text = f"Paso {iteration}: {plan[step_idx] if plan else 'Analizar el problema'}"
        return step_text, base

    def _compose_response(self, plan: List[str], steps: List[str]) -> str:
        lines = ["Plan propuesto:"] + [f"- {p}" for p in plan] + ["\nEjecución:"] + steps
        return "\n".join(lines)