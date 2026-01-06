# capibara/agents/capibara_prompt_to_spec.py

class CapibaraPromptToAgentSpec:
    def __init__(self, model):
        self.model = model  # Puede be CapibaraModel or any wrapper open source

    def generate_spec(self, instruction: str) -> dict:
        prompt = f"""
Eres un generador de agentes para un system de IA llamado CapibaraGPT. 

Tu tarea es crear una especificación JSON con esta estructura:
{{
  "name": "...",           ← nombre del agente
  "llm": {{
    "type": "ollama",      ← solo usar LLMs open source
    "model": "llama3"      ← modelo local
  }},
  "tools": ["..."],        ← lista de herramientas a usar (por ejemplo: "sumar", "web_search", etc.)
  "vectordb": {{
    "type": "qdrant"       ← si se necesita contexto documental, incluir esto
  }}
}}

Instrucción del usuario:
{instruction}

Devuelve solo el JSON con la especificación.
        """

        raw = self.model.generate(prompt)
        try:
            import json
            return json.loads(raw)
        except Exception:
            raise ValueError(f"Error interpretando la salida del modelo: {raw}")
