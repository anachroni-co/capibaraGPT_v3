"""
layers ssm_hybrid_layers module.

# This module provides functionality for ssm_hybrid_layers.
"""

import os
import sys

import logging
import hashlib
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

from capibara.jax import jax
from flax import linen as nn
from functools import partial
from capibara.jax import numpy as jnp

# Import existing optimized components
from .base import BaseLayer, LayerConfig
from .self_attention import TpuAttentionCache, _attention_cache
from .neurogenesis import TpuNeurogenesisCache, _neurogenesis_cache

# Safe imports for training integration
try:
    from ..training.optimizations import ULTRA_OPTIMIZATIONS_AVAILABLE
    ULTRA_TRAINING_AVAILABLE = True
except ImportError:
    ULTRA_TRAINING_AVAILABLE = False
    ULTRA_OPTIMIZATIONS_AVAILABLE = False

# Safe imports for SSM components
try:
    from ..jax.nn.ultra_optimizations import (
        MambaBlock, S4Block, HybridSSMLayer
    )
    SSM_COMPONENTS_AVAILABLE = True
except ImportError:
    SSM_COMPONENTS_AVAILABLE = False
    MambaBlock = None
    S4Block = None
    HybridSSMLayer = None

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module ssm_hybrid_layers.py starting")
    return True

if __name__ == "__main__":
    main()
