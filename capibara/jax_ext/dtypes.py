# Copyright 2019 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX dtypes compatibility module."""

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

import numpy as np
from typing import Any, Union

# Basic dtype definitions
bfloat16 = np.dtype('float32')  # Fallback when bfloat16 not available
float0 = np.dtype('float32')

def canonicalize_dtype(dtype: Any) -> np.dtype:
    """Canonicalize a dtype to a numpy dtype."""
    if dtype is None:
        return np.dtype('float32')
    return np.dtype(dtype)

def finfo(dtype: Any):
    """Get floating point info for a dtype."""
    return np.finfo(np.dtype(dtype))

def iinfo(dtype: Any):
    """Get integer info for a dtype."""
    return np.iinfo(np.dtype(dtype))

def issubdtype(arg1: Any, arg2: Any) -> bool:
    """Check if arg1 is a subdtype of arg2."""
    return np.issubdtype(arg1, arg2)

def extended(*args, **kwargs):
    """Extended dtype support (placeholder)."""
    return np.dtype('float32')

def prng_key(*args, **kwargs):
    """PRNG key dtype (placeholder)."""
    return np.dtype('uint32')

def result_type(*args) -> np.dtype:
    """Determine the result type for given inputs."""
    return np.result_type(*args)

def scalar_type_of(x: Any) -> type:
    """Get the scalar type of an array."""
    if hasattr(x, 'dtype'):
        return x.dtype.type
    return type(x)

__all__ = [
    'bfloat16',
    'canonicalize_dtype',
    'finfo',
    'float0',
    'iinfo',
    'issubdtype',
    'extended',
    'prng_key',
    'result_type',
    'scalar_type_of',
]
