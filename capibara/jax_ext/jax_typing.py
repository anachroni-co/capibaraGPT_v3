"""
JAX typing module - Minimal implementation

Minimal types for compatibility with core.py
"""

import numpy as np
from typing import Any, Union

# Basic JAX types
Array = Union[np.ndarray, Any]  # Includes Tracers
DType = np.dtype
Shape = tuple[int, ...]

__all__ = ['Array', 'DType', 'Shape']
