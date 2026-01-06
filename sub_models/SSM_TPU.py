"""
sub_models SSM_TPU module.

# This module provides functionality for SSM_TPU.
"""

import os
import sys


from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tupleonal, Tuple

import jax
import jax.numpy as jnp
# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from flax import linen as nn
from capibara.jax.numpy import jnp
from flax.linen.partitioning import param_with_axes

from capibara.core.config import nified_mesh

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module SSM_TPU.py starting")
    return True

if __name__ == "__main__":
    main()
