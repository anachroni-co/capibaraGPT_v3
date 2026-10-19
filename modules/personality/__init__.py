"""
Unified Personality System for CapibaraGPT

Exports the unified personality system that fuses:
- Human Gender Personality (core)
- Ethical System
- Optimized Cache
- Advanced Scoring
"""

import logging

logger = logging.getLogger(__name__)

try:
    from .unified_personality_system import (
        UnifiedPersonalitySystem,
        UnifiedPersonalityConfig,
        PersonalityIntegrationMixin,
        create_unified_personality_system,
    )
    UNIFIED_AVAILABLE = True
except Exception as e:
    UnifiedPersonalitySystem = None
    UnifiedPersonalityConfig = None
    PersonalityIntegrationMixin = None
    create_unified_personality_system = None
    UNIFIED_AVAILABLE = False
    logger.warning("Unified personality system unavailable: %s", e)

try:
    from .human_gender_personality import (
        ProductionHumanGenderConfig,
        ProductionHumanGenderPersonality,
    )
    GenderPersonalityConfig = ProductionHumanGenderConfig
    CORE_AVAILABLE = True
except Exception as e:
    ProductionHumanGenderConfig = None
    ProductionHumanGenderPersonality = None
    GenderPersonalityConfig = None
    CORE_AVAILABLE = False
    logger.warning("Human gender personality unavailable: %s", e)

try:
    from .cache_personality import cache_personality
    CACHE_AVAILABLE = True
except Exception as e:
    cache_personality = None
    CACHE_AVAILABLE = False
    logger.warning("Cache personality unavailable: %s", e)

__all__ = [
    # Unified System
    "UnifiedPersonalitySystem",
    "UnifiedPersonalityConfig",
    "create_unified_personality_system",
    "PersonalityIntegrationMixin",
    "UNIFIED_AVAILABLE",
    # Core Components
    "ProductionHumanGenderPersonality",
    "ProductionHumanGenderConfig",
    "GenderPersonalityConfig",
    "CORE_AVAILABLE",
    "cache_personality",
    "CACHE_AVAILABLE",
]
