"""
cot optimizations module.

# This module provides functionality for optimizations.
"""

import time
import functools
from typing import Any, Callable, Dict, Optional, Tuple, List

import jax
import jax.numpy as jnp
from jax.experimental import maps
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec

from capibara.config.cot_config import AdvancedCoTConfig
from capibara.config.memory_config import MemoryOptimizationConfig

if __name__ == "__main__":
    main()
