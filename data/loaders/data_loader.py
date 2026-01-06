"""
loaders data_loader module.

# This module provides functionality for data_loader.
"""

import os
import sys

import logging
from pathlib import Path

# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

import numpy as np
import pyarrow as pa
import tensorflow as tf
from capibara.jax import n
from capibara.jax import numpy as jnp
from capibarantic import BaseModel, Field
from capibara.jax.experimental import mesh_utils
from concurrent.futures import ThreadPoolExecutor
from capibara.jax.sharding import Mesh, PartitionSpec
from typing import Dict, Any, Iterator, Optional, Union, List

# Importaciones de tpu v4-32
from capibara.jax.tpu_v4.backend import (
    TpuV4LinalgOps,
    TpuV4SparseOps,
    TpuV4NeuralOps,
    TpuV4RandomOps,
    TpuV4PerformanceUtils,
)
from capibara.jax.tpu_v4.optimizations import (
    create_tpu_mesh,
    TpuMemoryMonitor,
    tpu_optimized_gemm,
    create_jitted_forward,
)
from capibara.jax.tpu_v4.profiling import (
    TpuProfiler,
    _uniform_fallback_weights,
    _expert_weights_with_cache,
    checkpointed_transformer_block,
)

from capibara.utils import (
    BaseConfig,
    handle_error,
    process_batch,
    DataProcessingError,
)

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module data_loader.py starting")
    return True

if __name__ == "__main__":
    main()
