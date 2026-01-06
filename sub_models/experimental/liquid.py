"""
experimental liquid module.

# This module provides functionality for liquid.
"""

import os
import sys
import logging

# This adds the project root folder to the Python search path
# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from typing import Dict, Any, Optional
from capibara.jax import jax  # type: ignore
from flax import linen as nn  # type: ignore
from pydantic import BaseModel, Field # type: ignore
from capibara.jax import numpy as jnp  # type: ignore

from capibara.interfaces.isub_models import ISubModel

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module liquid.py starting")
    return True

if __name__ == "__main__":
    main()
