"""
Submódulo semiótico de CapibaraGPT.

Este submódulo agrupa los componentes principales para el análisis semiótico avanzado,
la interacción semiótica y la configuración of metrics y cuantización asociadas.

Componentes expuestos:
- SemioModule: Módulo de análisis semiótico configurable.
- SemioticInteraction: Módulo de interacción semiótica avanzada.
- SapirWhorfAdapter: Adaptador para modulación semántica y cognitiva basada en idioma.
- QuantizationConfig, ScalingConfig, InterpretationMetrics, TPUMetrics, SemioticMetrics: Configuraciones y métricas especializadas para el análisis semiótico.
"""
from .sapir_whorf_adapter import SapirWhorfAdapter

# Optional imports with fallbacks
try:
    from .semio import SemioModule
    SEMIO_AVAILABLE = True
except ImportError:
    SemioModule = None
    SEMIO_AVAILABLE = False

try:
    from .semiotic_interaction import (
        SemioticInteraction,
        QuantizationConfig,
        ScalingConfig,
        InterpretationMetrics,
        TPUMetrics,
        SemioticMetrics
    )
    SEMIOTIC_INTERACTION_AVAILABLE = True
except ImportError:
    SemioticInteraction = None
    QuantizationConfig = None
    ScalingConfig = None
    InterpretationMetrics = None
    TPUMetrics = None
    SemioticMetrics = None
    SEMIOTIC_INTERACTION_AVAILABLE = False

try:
    from .mnemosyne_semio_module import MnemosyneSemioModule
    MNEMOSYNE_AVAILABLE = True
except ImportError:
    MnemosyneSemioModule = None
    MNEMOSYNE_AVAILABLE = False

# Dynamic __all__ based on available components
__all__ = ["SapirWhorfAdapter"]

if SEMIO_AVAILABLE:
    __all__.append("SemioModule")

if SEMIOTIC_INTERACTION_AVAILABLE:
    __all__.extend([
        "SemioticInteraction",
        "QuantizationConfig",
        "ScalingConfig",
        "InterpretationMetrics",
        "TPUMetrics",
        "SemioticMetrics"
    ])

if MNEMOSYNE_AVAILABLE:
    __all__.append("MnemosyneSemioModule")
