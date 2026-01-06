"""
tpu_v4 builder module.

# This module provides functionality for builder.
"""

from typing import Dict, List, Optional, Any
import logging
import os
import sys
# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from capibara.jax import jax
from capibara.jax import numpy as jnp
from packaging import version

from .build_config import TpuV4BuildConfig, CMAKE_TEMPLATE, SETUP_PY_TEMPLATE

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module builder.py starting")
    return True

if __name__ == "__main__":
    main()
