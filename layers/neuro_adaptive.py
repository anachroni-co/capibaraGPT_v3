"""
layers neuro_adaptive module.

# This module provides functionality for neuro_adaptive.
"""

import os
import sys

import logging
# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../CapibaraGPT v3
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

from jax import ng
from flax import linen as nn
from functools import partial
from jax import numpy as jnp
from typing import Dict, Any, Optional, Tuple
from capibara.layers.base import BaseLayer, LayerConfig

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module neuro_adaptive.py starting")
    return True

if __name__ == "__main__":
    main()
