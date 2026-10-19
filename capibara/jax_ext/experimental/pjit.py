"""
JAX experimental pjit

Parallel JIT compilation utilities.
"""

try:
    from jax import pjit as jax_pjit

    def pjit(fn, in_shardings=None, out_shardings=None, **kwargs):
        """Parallel JIT compilation."""
        return jax_pjit(fn, in_shardings=in_shardings, out_shardings=out_shardings, **kwargs)

except ImportError:
    # Fallback - just return the function
    def pjit(fn, in_shardings=None, out_shardings=None, **kwargs):
        """Fallback pjit - returns function as-is."""
        return fn

__all__ = ['pjit']
