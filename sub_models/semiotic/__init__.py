"""
CapibaraGPT Semiotic Submodule.

This submodule groups the main components for advanced semiotic analysis,
semiotic interaction, and configuration of metrics and associated quantization.

Exposed Components:
- SemioModule: Configurable semiotic analysis module.
- SemioticInteraction: Advanced semiotic interaction module.
- SapirWhorfAdapter: Adapter for language-based semantic and cognitive modulation.
- QuantizationConfig, ScalingConfig, InterpretationMetrics, TPUMetrics, SemioticMetrics: Specialized configurations and metrics for semiotic analysis.
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
