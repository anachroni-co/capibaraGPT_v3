"""
JAX _src module - Internal implementation

Minimal internal module for compatibility with existing code.
"""

# Re-export numpy from main module
from .. import numpy

# Create minimal core module
from . import core

__all__ = ['numpy', 'core']
