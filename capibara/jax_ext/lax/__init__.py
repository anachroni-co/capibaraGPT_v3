"""
JAX LAX Module - Low-level operations

Provides low-level operations including linear algebra.
"""

from .linalg import qr, svd, eigh, solve, inv, det, cholesky


def stop_gradient(x):
    """Fallback stop_gradient (identity)."""
    return x

__all__ = [
    'qr',
    'svd',
    'eigh',
    'solve',
    'inv',
    'det',
    'cholesky',
    'stop_gradient',
]
