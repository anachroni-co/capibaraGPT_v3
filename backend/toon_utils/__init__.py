"""
TOON Utils - Token-Oriented Object Notation
Formato compacto para reducir uso de tokens en LLMs
"""

from .parser import ToonParser
from .encoder import ToonEncoder
from .format_manager import FormatManager

# Versiones optimizadas
from .parser_optimized import ToonParserOptimized
from .encoder_optimized import ToonEncoderOptimized
from .format_manager_optimized import FormatManagerOptimized

# Versiones ultra optimizadas con caché
from .parser_ultra_optimized import ToonParserUltraOptimized
from .encoder_ultra_optimized import ToonEncoderUltraOptimized
from .format_manager_ultra_optimized import FormatManagerUltraOptimized

__version__ = "1.0.0"
__all__ = [
    "ToonParser", "ToonEncoder", "FormatManager",
    "ToonParserOptimized", "ToonEncoderOptimized", "FormatManagerOptimized",
    "ToonParserUltraOptimized", "ToonEncoderUltraOptimized", "FormatManagerUltraOptimized"
]