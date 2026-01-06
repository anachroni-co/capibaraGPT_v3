"""
tpu_v4 py_client_gpu module.

# This module provides functionality for py_client_gpu.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

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
import numpy as np
from capibara.jax import numpy as jnp
from absl import logging as absl_logging

from capibara.jax.tpu_v4 import prng_kernels
from capibara.jax.tpu_v4 import hybrid_kernels
from capibara.jax.tpu_v4 import linalg_kernels

# setup de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module py_client_gpu.py starting")
    return True

if __name__ == "__main__":
    main()
