"""
Capibara Auto Agent - Self-improving agent with iterative execution.

This module provides an autonomous agent that can iteratively improve
its responses by generating new agent specifications and executing them
until a satisfactory result is achieved.

Key Components:
    - CapibaraAutoAgent: Main class for autonomous agent execution

Example:
    >>> from agents.capibara_auto_agent import CapibaraAutoAgent
    >>> agent = CapibaraAutoAgent(base_model)
    >>> history = agent.run("Find information about quantum computing")

Author: Skydesk International Dev Team.
"""

import time
from .capibara_agent_factory import create_agent_from_spec
from .capibara_prompt_to_spec import CapibaraPromptToAgentSpec

import logging
logger = logging.getLogger(__name__)

class CapibaraAutoAgent:
    """Autonomous agent with iterative self-improvement.
    
    This agent generates its own specifications based on user goals
    and iteratively improves responses until satisfactory results
    are achieved.
    
    Attributes:
        base_model: The base model for specification generation.
        spec_generator: The specification generator instance.
    """
    
    def __init__(self, base_model):
        """Initialize the auto agent.
        
        Args:
            base_model: The base model to use for spec generation.
        """
        self.base_model = base_model
        self.spec_generator = CapibaraPromptToAgentSpec(base_model)

    def run(self, user_goal: str, max_iterations: int = 3):
        """Run the auto agent with iterative improvement.
        
        Args:
            user_goal: The user's goal or instruction.
            max_iterations: Maximum number of improvement iterations.
            
        Returns:
            List of tuples containing (spec, response) for each iteration.
        """
        history = []
        current_instruction = user_goal

        for i in range(max_iterations):
            logger.info(f"\n Iteración {i+1}")
            spec = self.spec_generator.generate_spec(current_instruction)
            agent = create_agent_from_spec(spec)

            logger.info(f" Ejecutando agente '{spec['name']}'...")
            response = agent.ask(current_instruction)

            logger.info(f" Respuesta del agente:\n{response}")
            history.append((spec, response))

            # Improvement/end logic (very basic here, you can use any heuristic):
            if "no se encontró información" in response.lower():
                logger.info("️ Respuesta vacía o débil, generando nuevo agente...")
                current_instruction += " Mejora la precisión."
                time.sleep(1)
            else:
                logger.info(" Respuesta aceptable, finalizando.")
                break

        return history
