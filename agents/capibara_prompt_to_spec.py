"""
Capibara Prompt to Spec - Agent specification generator from prompts.

This module converts natural language instructions into structured agent
specifications that can be used to instantiate configured agents.

Key Components:
    - CapibaraPromptToAgentSpec: Class for generating agent specs from prompts

Example:
    >>> from agents.capibara_prompt_to_spec import CapibaraPromptToAgentSpec
    >>> generator = CapibaraPromptToAgentSpec(model)
    >>> spec = generator.generate_spec("Create a math assistant")

Author: Skydesk International Dev Team.
"""

class CapibaraPromptToAgentSpec:
    """Generates agent specifications from natural language instructions.
    
    This class uses an LLM to convert user instructions into structured
    JSON specifications that define agent configuration including LLM
    type, model, tools, and vector database settings.
    
    Attributes:
        model: The LLM model used for specification generation.
    """
    
    def __init__(self, model):
        """Initialize the specification generator.
        
        Args:
            model: The LLM model to use for generating specs (CapibaraModel
                or any compatible wrapper).
        """
        self.model = model  # Can be CapibaraModel or any open source wrapper

    def generate_spec(self, instruction: str) -> dict:
        """Generate an agent specification from a user instruction.
        
        Args:
            instruction: The natural language instruction describing
                what the agent should do.
                
        Returns:
            A dictionary containing the agent specification with keys:
                - name: Agent name
                - llm: LLM configuration (type, model)
                - tools: List of tool names
                - vectordb: Optional vector database configuration
                
        Raises:
            ValueError: If the model output cannot be parsed as JSON.
        """
        prompt = f"""
You are an agent generator for an AI system called CapibaraGPT.

Your task is to create a JSON specification with this structure:
{{
  "name": "...",           <- agent name
  "llm": {{
    "type": "ollama",      <- only use open source LLMs
    "model": "llama3"      <- local model
  }},
  "tools": ["..."],        <- list of tools to use (e.g.: "add", "web_search", etc.)
  "vectordb": {{
    "type": "qdrant"       <- if document context is needed, include this
  }}
}}

User instruction:
{instruction}

Return only the JSON with the specification.
        """

        raw = self.model.generate(prompt)
        try:
            import json
            return json.loads(raw)
        except Exception:
            raise ValueError(f"Error interpreting model output: {raw}")
