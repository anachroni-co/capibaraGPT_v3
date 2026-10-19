"""XLA utilities for JAX - Wrapper for compatibility."""

try:
    # Try import from real JAX
    from jax import xla
    from jax.interpreters import xla as xla_interpreters
    HAS_JAX_XLA = True
except ImportError:
    # Fallback for compatibility
    HAS_JAX_XLA = False

    class MockXla:
        """Mock XLA for compatibility."""

        @staticmethod
        def compile(*args, **kwargs):
            """Mock compile."""
            return None

        @staticmethod
        def execute(*args, **kwargs):
            """Mock execute."""
            return None

    xla = MockXla()
    xla_interpreters = MockXla()

# Export main functions
__all__ = [
    'xla',
    'xla_interpreters',
    'HAS_JAX_XLA'
]
