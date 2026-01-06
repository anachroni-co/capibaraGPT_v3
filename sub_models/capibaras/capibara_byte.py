"""
capibaras capibara_byte module.

# This module provides functionality for capibara_byte.
"""

import os
import sys
import logging

# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from capibara.jax import jax
from capibara.jax.numpy import jnp
from flax import linen as nn
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from functools import partial

from .tpu_base_config import TPUBaseConfig
from capibara.interfaces.isub_models import ConfigTPU, PrecisionMode
from capibara.core.arm_optimizations import ARM_OPTIMIZATIONS, HARDWARE_INFO

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module capibara_byte.py starting")
    return True

if __name__ == "__main__":
    main()
