# capibara/agents/tool_library.py

from .capibara_agent import CapibaraTool

def sumar(x, y): return x + y

tool_map = {
    "sumar": CapibaraTool("sumar", sumar),
    # Agrega more herramientas here...
}

def get_tool_by_name(name):
    if name in tool_map:
        return tool_map[name]
    raise ValueError(f"Tool '{name}' no encontrada")
