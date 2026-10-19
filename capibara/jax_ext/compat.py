"""
Módulo de compatibilidad for JAX.

Este módulo proporciona una capa de compatibilidad between la implementation
interna de JAX en CapibaraGPT and la implementation estándar de JAX.
"""

import os
from typing import Any, Dict, Optional, Union

# try import JAX estándar
try:
    import numpy as _base_jnp
    _base_jax = None  # not tenemos JAX estándar, usamos nuestro mock
    HAS_JAX = True
except ImportError:
    _base_jax = None
    _base_jnp = None
    HAS_JAX = False

# Funciones de compatibilidad
def get_jax() -> Any:
    """
    Obtiene la implementation de JAX a use.
    
    Returns:
        Módulo JAX (estándar or mock)
    """
                # always use nuestro mock JAX for CapibaraGPT
    if True:  # use nuestro JAX interno
        # create mock JAX compatible
        class MockJAX:
            def __init__(self):
                self.random = MockRandom()
                
            def jit(self, fn=None, **kwargs):
                """Mock jit - acepta todos los argumentos."""
                if fn is None:
                    return lambda f: f
                return fn
                
            def vmap(self, fn=None, **kwargs):
                if fn is None:
                    return lambda f: f
                return fn
                
            def grad(self, fn=None, **kwargs):
                if fn is None:
                    return lambda f: f
                return fn
                
            def devices(self):
                return [MockDevice()]
                
        return MockJAX()

def get_numpy() -> Any:
    """
    Obtiene la implementation de numpy de JAX a use.
    
    Returns:
        Módulo numpy de JAX (estándar or mock)
    """
    if HAS_JAX and _base_jnp is not None:
        return _base_jnp
    else:
        # create un mock simple for testing
        import numpy as np
        class MockJNP:
            def __getattr__(self, name):
                return getattr(np, name, lambda *args, **kwargs: np.array([]))
                
        return MockJNP()

class MockRandom:
    def PRNGKey(self, seed):
        return seed
        
    def normal(self, key, shape):
        import numpy as np
        return np.random.normal(size=shape).astype(np.float32)
        
    def randint(self, key, shape, minval, maxval):
        import numpy as np
        return np.random.randint(minval, maxval, size=shape)
    
    def uniform(self, key, shape, minval=0.0, maxval=1.0):
        import numpy as np
        return np.random.uniform(minval, maxval, size=shape).astype(np.float32)
    
    def choice(self, key, a, shape=(), replace=True, p=None):
        import numpy as np
        return np.random.choice(a, size=shape, replace=replace, p=p)

class MockLax:
    """Mock LAX module with common operations."""
    
    def stop_gradient(self, x):
        return x
    
    def select(self, pred, on_true, on_false):
        import numpy as np
        return np.where(pred, on_true, on_false)
    
    def cond(self, pred, true_fun, false_fun, operand):
        if pred:
            return true_fun(operand)
        else:
            return false_fun(operand)
    
    def dynamic_slice(self, operand, start_indices, slice_sizes):
        return operand[tuple(slice(start, start + size) 
                           for start, size in zip(start_indices, slice_sizes))]
    
    def dynamic_update_slice(self, operand, update, start_indices):
        import numpy as np
        result = operand.copy()
        slices = tuple(slice(start, start + update.shape[i]) 
                      for i, start in enumerate(start_indices))
        result[slices] = update
        return result

class MockLib:
    """Mock JAX lib module."""
    
    def __init__(self):
        self.xla_bridge = MockXLABridge()
        self.xla_client = MockXLAClient()

class MockXLABridge:
    """Mock XLA bridge."""
    
    def get_backend(self, platform=None):
        return MockXLABackend()

class MockXLAClient:
    """Mock XLA client."""
    pass

class MockXLABackend:
    """Mock XLA backend."""
    
    def platform(self):
        return 'cpu'

class MockDevice:
    def __init__(self):
        self.platform = 'cpu'

def setup_tpu():
    """Configure environment for tpu."""
    if HAS_JAX:
        jax = get_jax()
        if jax.devices()[0].platform == 'tpu':
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
            os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=32'
            return True
    return False

def setup_gpu():
    """Configure environment for gpu."""
    if HAS_JAX:
        jax = get_jax()
        if jax.devices()[0].platform == 'gpu':
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
            return True
    return False

def get_device_type() -> str:
    """Get the type of device being used (tpu, gpu, or cpu)."""
    if HAS_JAX:
        jax = get_jax()
        return jax.devices()[0].platform
    return 'cpu'

def get_device_count() -> int:
    """Get the number of available devices."""
    if HAS_JAX:
        jax = get_jax()
        return len(jax.devices())
    return 1

# export funciones de compatibilidad
__all__ = [
    'get_jax',
    'get_numpy',
    'HAS_JAX',
    'setup_tpu',
    'setup_gpu',
    'get_device_type',
    'get_device_count'
] 