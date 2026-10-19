"""
Interfaces Module - CapibaraGPT v3

Interface definitions and protocols for type-safe contracts:
- icache: Cache interface protocol
- isubmodel: Core sub-model protocol
- isub_models: Sub-model configurations
- ultra_interface_system: Advanced interface protocols
"""

import logging

logger = logging.getLogger(__name__)

# Cache interface
try:
    from .icache import ICacheModule
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    ICacheModule = None

# Sub-model interface
try:
    from .isubmodel import ISubModel
    SUBMODEL_AVAILABLE = True
except ImportError:
    SUBMODEL_AVAILABLE = False
    ISubModel = None

# Sub-model configurations
try:
    from .isub_models import (
        PrecisionMode,
        ExpertActivationMode,
        ConfigTPU,
        ExpertContext,
        ExpertResult,
    )
    SUBMODEL_CONFIG_AVAILABLE = True
except ImportError:
    SUBMODEL_CONFIG_AVAILABLE = False
    PrecisionMode = None
    ExpertActivationMode = None
    ConfigTPU = None
    ExpertContext = None
    ExpertResult = None

# Ultra interface system
try:
    from .ultra_interface_system import (
        UltraInterfaceSystem,
        create_ultra_interface,
    )
    ULTRA_AVAILABLE = True
except ImportError:
    ULTRA_AVAILABLE = False
    UltraInterfaceSystem = None
    create_ultra_interface = None


__all__ = [
    # Cache
    "ICacheModule",
    # Sub-model
    "ISubModel",
    "PrecisionMode",
    "ExpertActivationMode",
    "ConfigTPU",
    "ExpertContext",
    "ExpertResult",
    # Ultra
    "UltraInterfaceSystem",
    "create_ultra_interface",
    # Flags
    "CACHE_AVAILABLE",
    "SUBMODEL_AVAILABLE",
    "SUBMODEL_CONFIG_AVAILABLE",
    "ULTRA_AVAILABLE",
]
