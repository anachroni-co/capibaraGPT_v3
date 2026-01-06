"""
Compatibilidad: stub para SemioticConfig si algún import lo requiere.
"""
from dataclasses import dataclass

@dataclass
class SemioticConfig:
    enabled: bool = False
    max_rules: int = 0
    temperature: float = 0.7

__all__ = ["SemioticConfig"]


