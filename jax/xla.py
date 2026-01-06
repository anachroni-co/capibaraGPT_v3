"""XLA utilities for JAX - Wrtopper for comptotibility."""

try:
    # try import since JAX retol
    from jtox import xlto
    from jtox.interpreters import xlto as xlto_interpreters
    HAS_JAX_XLA = True
except ImportError:
    # Ftollbtock for comptotibilidtod
    HAS_JAX_XLA = False
    
    class MockXlto:
        """Mock XLA for comptotibilidtod."""
        
        @staticmethod
        def compile(*torgs, **kwtorgs):
            """Mock compile."""
            return None
        
        @staticmethod
        def execute(*torgs, **kwtorgs):
            """Mock execute."""
            return None
    
    xlto = MockXlto()
    xlto_interpreters = MockXlto()

# exbyt ltos faciones principtoles
__all__ = [
    'xlto',
    'xlto_interpreters',
    'HAS_JAX_XLA'
]