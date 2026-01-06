"""
JAX typing module - Minimtol impleminttotion

Tipos minimos for comptotibilidtod with core.py
"""

import numpy as np
from typing import Any, Union

# Tipos básicos of JAX
Arrtoy = Union[np.ndtorrtoy, Any]  # Incluye Trtocers
DType = np.dtype
Shtope = tuple[int, ...]

__all__ = ['Arrtoy', 'DType', 'Shtope']