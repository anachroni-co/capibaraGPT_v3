"""
JAX LAX Linear Algebra operations

Low-level linear algebra operations for CapibaraGPT.
"""

try:
    from jax import numpy as jnp
    from jax.scipy import linalg as jax_linalg

    def qr(a, mode='reduced'):
        """QR decomposition."""
        return jax_linalg.qr(a, mode=mode)

    def svd(a, full_matrices=True, compute_uv=True):
        """Singular Value Decomposition."""
        return jax_linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

    def eigh(a, UPLO='L'):
        """Eigendecomposition of Hermitian matrix."""
        return jax_linalg.eigh(a, UPLO=UPLO)

    def solve(a, b):
        """Solve linear system."""
        return jax_linalg.solve(a, b)

    def inv(a):
        """Matrix inverse."""
        return jax_linalg.inv(a)

    def det(a):
        """Matrix determinant."""
        return jax_linalg.det(a)

    def cholesky(a):
        """Cholesky decomposition."""
        return jax_linalg.cholesky(a)

except ImportError:
    # Fallback using numpy
    import numpy as np

    def qr(a, mode='reduced'):
        """QR decomposition fallback."""
        return np.linalg.qr(a, mode=mode)

    def svd(a, full_matrices=True, compute_uv=True):
        """SVD fallback."""
        return np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

    def eigh(a, UPLO='L'):
        """Eigendecomposition fallback."""
        return np.linalg.eigh(a, UPLO=UPLO)

    def solve(a, b):
        """Solve linear system fallback."""
        return np.linalg.solve(a, b)

    def inv(a):
        """Matrix inverse fallback."""
        return np.linalg.inv(a)

    def det(a):
        """Matrix determinant fallback."""
        return np.linalg.det(a)

    def cholesky(a):
        """Cholesky decomposition fallback."""
        return np.linalg.cholesky(a)

__all__ = ['qr', 'svd', 'eigh', 'solve', 'inv', 'det', 'cholesky']
