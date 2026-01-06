import logging
import sys
import os
"""
distributed configuration module.
"""

from typing import Dict, Any

# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from capibara.jax import jax # type: ignore
from capibara.jax import numpy as jnp # type: ignore
from capibara.jax.experimental import PartitionSpec as P # type: ignore
from capibara.jax.experimental.maps import Mesh # type: ignore
from capibara.jax.experimental.maps import mesh # type: ignore
from capibara.jax.experimental.maps import shard_map # type: ignore
from functools import wraps
from typing import Optional, Tuple, Dict, Any, Callable, List, Union, TypeVar, cast
import numpy as np # type: ignore
from capibara.jax.experimental.pjit import pjit # type: ignore

from capibara.jax.tpu_v4.backend import (
    TpuV4LinalgOps,
    TpuV4SparseOps,
    TpuV4NeuralOps,
    TpuV4RandomOps,
    TpuV4PerformanceUtils
)

logger = logging.getLogger(__name__)

# Tipos genéricos for funciones
F = TypeVar('F', bound=Callable[..., Any])
R = TypeVar('R')

# setup de malla tpu v4-32
TPU_MESH_SHAPE = (32, 8)  # 32 chips, 8 cores by chip
TPU_MESH = None  # Se inicializará en setup_mesh()

# Especificaciones de sharding optimizadas for tpu v4-32
BATCH_SHARDING = P('batch', None, None)  # (batch, seq_len, hidden)
MODEL_SHARDING = P(None, None, 'model')  # (batch, seq_len, hidden)
HYBRID_SHARDING = P('batch', None, 'model')  # (batch, seq_len, hidden)
REPLICATED = P(None, None, None)

# Tipos de data optimizados for tpu v4-32
DTYPE = jnp.float32
TPU_DTYPE = jnp.bfloat16  # better rendimiento en tpu v4-32

def setup_mesh(shape: Tuple[int, ...] = TPU_MESH_SHAPE) -> Mesh:
    """Initializes la malla tpu v4-32 global."""
    pass
    def __init__(self):
        self.settings = {}
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.settings[key] = value

# Global config instance
config = Config()
