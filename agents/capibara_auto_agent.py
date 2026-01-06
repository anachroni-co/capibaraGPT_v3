# capibara/agents/capibara_auto_agent.py

import time
from .capibara_agent_factory import create_agent_from_spec
from .capibara_prompt_to_spec import CapibaraPromptToAgentSpec

class CapibaraAutoAgent:
    def __init__(self, base_model):
        self.base_model = base_model
        self.spec_generator = CapibaraPromptToAgentSpec(base_model)

    def run(self, user_goal: str, max_iterations: int = 3):
        history = []
        current_instruction = user_goal

        for i in range(max_iterations):
            print(f"\n🔁 Iteración {i+1}")
            spec = self.spec_generator.generate_spec(current_instruction)
            agent = create_agent_from_spec(spec)

            print(f"🤖 Ejecutando agente '{spec['name']}'...")
            response = agent.ask(current_instruction)

            print(f"🧠 Respuesta del agente:\n{response}")
            history.append((spec, response))

            # Improvement/end logic (very basic here, you can use any heuristic):
            if "no se encontró información" in response.lower():
                print("⚠️ Respuesta vacía o débil, generando nuevo agente...")
                current_instruction += " Mejora la precisión."
                time.sleep(1)
            else:
                print("✅ Respuesta aceptable, finalizando.")
                break

        return history
