#!/usr/bin/env python3
"""Chain of Thought (CoT) Reasoning Module for CapibaraGPT.

This module implements step-by-step reasoning capabilities that allow the model
to break down complex problems into logical steps, similar to how humans solve
problems through explicit reasoning chains.

The Chain of Thought approach improves model performance on:
- Mathematical reasoning
- Logical deduction
- Multi-step problem solving
- Complex question answering

Key Components:
    - ChainOfThought: Main class for managing reasoning steps
    - create_cot_handler: Factory function for creating CoT handlers

Example:
    >>> from capibara.core.cot import ChainOfThought
    >>> cot = ChainOfThought()
    >>> cot.add_step("Analyze", "Break down the problem")
    >>> cot.add_step("Calculate", "Perform the calculation")
    >>> result = cot.solve_problem("What is 2 + 2?")

References:
    - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
      (Wei et al., 2022) https://arxiv.org/abs/2201.11903
    - "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
      (Wang et al., 2023) https://arxiv.org/abs/2203.11171
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ChainOfThought:
    """Chain of Thought reasoning implementation for multi-step problem solving.

    This class manages a sequence of reasoning steps that form a logical chain
    from problem statement to solution. Each step includes both the action taken
    and the reasoning behind it, enabling transparent and explainable problem solving.

    Attributes:
        steps (List[Dict[str, Any]]): List of reasoning steps, each containing:
            - step: Description of the step action
            - reasoning: Explanation of why this step is taken
            - step_number: Sequential number of the step
        current_step (int): Index of the current step being processed

    Example:
        >>> cot = ChainOfThought()
        >>> cot.add_step("Identify variables", "x and y are the unknowns")
        >>> cot.add_step("Set up equations", "Create system of equations")
        >>> chain = cot.get_reasoning_chain()
        >>> print(f"Total steps: {len(chain)}")
        Total steps: 2

    Note:
        The reasoning chain can be inspected at any point to understand the
        model's decision-making process, which is valuable for debugging and
        improving model performance.
    """

    def __init__(self):
        """Initialize an empty Chain of Thought reasoning system.

        Creates a new reasoning chain with no steps. Steps should be added
        sequentially as the problem is analyzed and solved.

        Example:
            >>> cot = ChainOfThought()
            >>> print(len(cot.steps))
            0
        """
        self.steps = []
        self.current_step = 0

    def add_step(self, step: str, reasoning: str):
        """Add a reasoning step to the chain.

        Appends a new step to the reasoning chain with its associated reasoning.
        Each step is automatically numbered sequentially.

        Args:
            step (str): Brief description of the action or step taken.
            reasoning (str): Detailed explanation of why this step is necessary
                and what it accomplishes.

        Example:
            >>> cot = ChainOfThought()
            >>> cot.add_step(
            ...     step="Parse the equation",
            ...     reasoning="Need to identify coefficients and variables"
            ... )
            >>> cot.add_step(
            ...     step="Apply quadratic formula",
            ...     reasoning="Standard method for solving quadratic equations"
            ... )

        Note:
            Each step is logged for debugging purposes. The step number is
            automatically assigned based on the current length of the chain.
        """
        self.steps.append({
            "step": step,
            "reasoning": reasoning,
            "step_number": len(self.steps) + 1
        })
        logger.info(f"Added CoT step {len(self.steps)}: {step}")

    def get_reasoning_chain(self) -> List[Dict[str, Any]]:
        """Retrieve the complete reasoning chain.

        Returns all steps that have been added to the reasoning chain,
        preserving their sequential order.

        Returns:
            List[Dict[str, Any]]: List of reasoning steps, each containing:
                - step: The action taken
                - reasoning: Explanation of the step
                - step_number: Sequential position in the chain

        Example:
            >>> cot = ChainOfThought()
            >>> cot.add_step("Step 1", "Reason 1")
            >>> cot.add_step("Step 2", "Reason 2")
            >>> chain = cot.get_reasoning_chain()
            >>> for item in chain:
            ...     print(f"Step {item['step_number']}: {item['step']}")
            Step 1: Step 1
            Step 2: Step 2
        """
        return self.steps

    def clear(self):
        """Clear the reasoning chain and reset to initial state.

        Removes all steps from the reasoning chain and resets the step counter.
        Useful when starting a new problem or reusing the same CoT instance.

        Example:
            >>> cot = ChainOfThought()
            >>> cot.add_step("Step 1", "Reason 1")
            >>> print(len(cot.steps))
            1
            >>> cot.clear()
            >>> print(len(cot.steps))
            0

        Note:
            This operation is logged for debugging purposes. All previous steps
            are permanently removed and cannot be recovered.
        """
        self.steps = []
        self.current_step = 0
        logger.info("Cleared reasoning chain")

    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """Solve a problem using chain of thought reasoning.

        Applies systematic reasoning steps to solve the given problem. This is a
        demonstration implementation that shows the general structure of CoT
        problem solving. In practice, this would be integrated with the model's
        generation capabilities.

        Args:
            problem (str): The problem statement to solve using CoT reasoning.

        Returns:
            Dict[str, Any]: Solution dictionary containing:
                - problem: The original problem statement
                - solution: The generated solution
                - steps: Complete reasoning chain used
                - confidence: Confidence score for the solution (0.0 to 1.0)

        Example:
            >>> cot = ChainOfThought()
            >>> result = cot.solve_problem("What is 2 + 2?")
            >>> print(result['problem'])
            What is 2 + 2?
            >>> print(f"Steps taken: {len(result['steps'])}")
            Steps taken: 4
            >>> print(f"Confidence: {result['confidence']}")
            Confidence: 0.95

        Note:
            This is a template implementation. In production, this method would
            integrate with the actual language model to generate step-by-step
            reasoning and solutions. The current implementation adds generic
            reasoning steps as a demonstration.

        See Also:
            - add_step: For manually adding custom reasoning steps
            - get_reasoning_chain: For inspecting the reasoning process
        """
        logger.info(f"Solving problem: {problem}")

        # Example reasoning steps - in production, these would be generated by the model
        self.add_step("Understand the problem", "First, I need to understand what is being asked")
        self.add_step("Break down into parts", "Divide the problem into smaller, manageable parts")
        self.add_step("Apply reasoning", "Use logical reasoning to solve each part")
        self.add_step("Verify solution", "Check if the solution makes sense")

        return {
            "problem": problem,
            "solution": "Problem solved using CoT reasoning",
            "steps": self.get_reasoning_chain(),
            "confidence": 0.95
        }

def create_cot_handler() -> ChainOfThought:
    """Factory function to create a Chain of Thought handler.

    Convenience function for creating a new ChainOfThought instance.
    Useful for dependency injection and cleaner code structure.

    Returns:
        ChainOfThought: A new, empty Chain of Thought handler instance.

    Example:
        >>> cot = create_cot_handler()
        >>> isinstance(cot, ChainOfThought)
        True
        >>> cot.add_step("First step", "Initial reasoning")

    Note:
        This is equivalent to directly calling ChainOfThought(), but provides
        a more explicit and discoverable API for users of this module.
    """
    return ChainOfThought()
