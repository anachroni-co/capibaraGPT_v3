"""
sub_models module.

This module provides functionality for sub_models operations, including
the CSA (Counterfactual Scenario Analysis) Expert and other specialized models.
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def get_project_root():
    """Get the root path of the project."""
    return Path(__file__).parent.parent

# Version information
__version__ = "1.1.0"
__author__ = "CapibaraGPT Team"

# Import CSA Expert components
try:
    from .csa_expert import CSAExpert, CSAExpertConfig
    CSA_BASIC_AVAILABLE = True
    logger.info("CSA Expert basic module loaded successfully")
except ImportError as e:
    logger.warning(f"CSA Expert basic module not available: {e}")
    CSAExpert = None
    CSAExpertConfig = None
    CSA_BASIC_AVAILABLE = False

# Import TPU optimized version separately
try:
    from .csa_expert_tpu_optimized import TPUOptimizedCSAExpert, TPUOptimizedConfig
    CSA_TPU_AVAILABLE = True
    logger.info("CSA Expert TPU module loaded successfully")
except ImportError as e:
    logger.warning(f"CSA Expert TPU module not available: {e}")
    TPUOptimizedCSAExpert = None
    TPUOptimizedConfig = None
    CSA_TPU_AVAILABLE = False

# Import Reasoning Enhancement Expert
try:
    from .reasoning_enhancement import ReasoningEnhancementExpert, ReasoningConfig, create_reasoning_expert
    REASONING_AVAILABLE = True
    logger.info("Reasoning Enhancement Expert module loaded successfully")
except ImportError as e:
    logger.warning(f"Reasoning Enhancement Expert module not available: {e}")
    ReasoningEnhancementExpert = None
    ReasoningConfig = None
    create_reasoning_expert = None
    REASONING_AVAILABLE = False

CSA_AVAILABLE = CSA_BASIC_AVAILABLE or CSA_TPU_AVAILABLE

# Module exports
__all__ = [
    "get_project_root",
    "CSA_AVAILABLE",
    "REASONING_AVAILABLE"
]

# Add CSA exports if available
if CSA_AVAILABLE:
    __all__.extend([
        "CSAExpert",
        "CSAExpertConfig", 
        "TPUOptimizedCSAExpert",
        "TPUOptimizedConfig"
    ])

# Add Reasoning Enhancement exports if available
if REASONING_AVAILABLE:
    __all__.extend([
        "ReasoningEnhancementExpert",
        "ReasoningConfig",
        "create_reasoning_expert"
    ])
