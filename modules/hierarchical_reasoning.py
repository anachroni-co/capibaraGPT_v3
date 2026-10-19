"""
Hierarchical Reasoning Module - Multi-phase reasoning scaffold.

This module provides a hierarchical reasoning system that processes prompts
through abstract (H) and detailed (L) phases. It is designed as a lightweight,
deterministic scaffold that can be integrated quickly and replaced by learned
models later.

Key Components:
    - HierarchicalReasoningConfig: Configuration dataclass for reasoning parameters
    - HierarchicalReasoningModule: Main reasoning class with H/L phase processing
    - HierarchicalReasoning: Alias for backwards compatibility

Example:
    >>> from modules.hierarchical_reasoning import HierarchicalReasoningModule
    >>> config = HierarchicalReasoningConfig(max_iterations=50)
    >>> reasoner = HierarchicalReasoningModule(config)
    >>> result = reasoner.process("Calculate the sum of 5 and 3")
    >>> print(result["confidence"])

Author: Skydesk International Dev Team.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalReasoningConfig:
    """Configuration for HierarchicalReasoningModule.
    
    Attributes:
        h_dim: Dimension for abstract (H) phase representations.
        l_dim: Dimension for detailed (L) phase representations.
        max_iterations: Maximum number of reasoning iterations.
        convergence_threshold: Threshold for detecting convergence.
        tpu_cores_assigned: Number of TPU cores for computation.
        use_with_consensus: Whether to integrate with consensus system.
        weight_in_consensus: Weight factor when used in consensus.
        override_threshold: Threshold for overriding consensus decisions.
        keywords_math: Keywords for detecting mathematical tasks.
        keywords_reasoning: Keywords for detecting reasoning tasks.
        keywords_planning: Keywords for detecting planning tasks.
    """
    h_dim: int = 512
    l_dim: int = 256
    max_iterations: int = 100
    convergence_threshold: float = 0.01
    tpu_cores_assigned: int = 3
    use_with_consensus: bool = True
    weight_in_consensus: float = 3.0
    override_threshold: float = 0.8

    keywords_math: Tuple[str, ...] = (
        "calculate", "solve", "equation", "integral", "derivative", "limit",
        "matrix", "sum", "product", "probability", "combinatorics",
    )
    keywords_reasoning: Tuple[str, ...] = (
        "deduce", "infer", "logic", "premise", "conclusion", "argument",
        "reason", "hypothesis", "demonstrate", "proof",
    )
    keywords_planning: Tuple[str, ...] = (
        "plan", "strategy", "steps", "optimize", "schedule",
        "objectives", "tasks", "sequence",
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
        """Initialize the HierarchicalReasoningModule.
        
        Args:
            config: Optional configuration object. If not provided, defaults are used.
        """
        self.config = config or HierarchicalReasoningConfig()

    def is_reasoning_task(self, prompt: str) -> bool:
        """Check if a prompt requires reasoning based on keyword detection.
        
        Args:
            prompt: The input text to analyze.
            
        Returns:
            True if the prompt contains math, reasoning, or planning keywords.
        """
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
                "Identify variables and what is being asked",
                "Break down the problem into mathematical substeps",
                "Apply relevant rules and theorems",
                "Simplify and verify coherence",
            ]
        elif any(k in text for k in self.config.keywords_planning):
            plan = [
                "Define objective and constraints",
                "List resources and risks",
                "Sequence steps with dependencies",
                "Evaluate success metrics and adjust",
            ]
        else:
            plan = [
                "Clarify the premise",
                "Enumerate hypotheses and alternatives",
                "Evaluate logical implications",
                "Reach a justified conclusion",
            ]
        return plan

    def _phase_l_detailed_step(self, prompt: str, plan: List[str], iteration: int) -> Tuple[str, float]:
        # Simple deterministic refinement signal
        base = 1.0 / (iteration + 2.0)
        step_idx = min(iteration - 1, len(plan) - 1) if plan else 0
        step_text = f"Step {iteration}: {plan[step_idx] if plan else 'Analyze the problem'}"
        return step_text, base

    def _compose_response(self, plan: List[str], steps: List[str]) -> str:
        lines = ["Proposed plan:"] + [f"- {p}" for p in plan] + ["\nExecution:"] + steps
        return "\n".join(lines)


# Backwards-compatible alias used across docs/imports.
HierarchicalReasoning = HierarchicalReasoningModule
