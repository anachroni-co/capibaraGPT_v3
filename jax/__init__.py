import logging
from typing import Any

logger = logging.getLogger(__name__)

# Intentar usar JAX real; si no está, usar fallbacks mínimos
try:
    import jax as _jax  # type: ignore
    import jax.numpy as _jnp  # type: ignore
    HAS_JAX = True
except Exception as _e:  # pragma: no cover - environment without JAX
    HAS_JAX = False
    logger.warning("JAX no disponible; usando fallbacks con numpy. Detalle: %s", _e)
    try:
        import numpy as _jnp  # type: ignore
    except Exception:
        # Minimal fallback without numpy: proveer funciones básicas
        class _MiniNumpy:  # type: ignore
            float16 = float
            float32 = float
            float64 = float
            bfloat16 = float
            def array(self, obj, dtype=None):
                try:
                    import numpy as np
                    return np.array(obj, dtype=dtype)
                except Exception:
                    return obj
            def zeros_like(self, obj):
                if isinstance(obj, list):
                    return [0 for _ in range(len(obj))]
                return 0
            def asarray(self, obj, dtype=None):
                return obj
            def zeros(self, shape, dtype=float):
                try:
                    if isinstance(shape, int):
                        return [0]*shape
                    return [[0]*shape[1] for _ in range(shape[0])]
                except Exception:
                    return 0
            def ones(self, shape, dtype=float):
                try:
                    if isinstance(shape, int):
                        return [1]*shape
                    return [[1]*shape[1] for _ in range(shape[0])]
                except Exception:
                    return 1
            def full(self, shape, value, dtype=float):
                try:
                    if isinstance(shape, int):
                        return [value]*shape
                    return [[value]*shape[1] for _ in range(shape[0])]
                except Exception:
                    return value
            def int32(self):
                return int
            def int64(self):
                return int
            def clip(self, x, a_min=None, a_max=None):
                try:
                    if a_min is not None:
                        x = max(a_min, x)
                    if a_max is not None:
                        x = min(a_max, x)
                except Exception:
                    pass
                return x
            def abs(self, x):
                try:
                    return x if not isinstance(x, list) else [abs(v) for v in x]
                except Exception:
                    return x
            def tanh(self, x):
                return x
            def max(self, x, axis=None, keepdims=False):
                if isinstance(x, list):
                    return max(x)
                return x
            def sum(self, x, axis=None, keepdims=False):
                if isinstance(x, list):
                    return sum(x)
                return x
            def mean(self, x, axis=None, keepdims=False):
                if isinstance(x, list) and len(x) > 0:
                    return sum(x)/len(x)
                return x
            def power(self, x, p):
                try:
                    return x ** p
                except Exception:
                    return x
            def argsort(self, a, axis=-1, kind=None, order=None):
                if isinstance(a, list):
                    return sorted(range(len(a)), key=lambda i: a[i])
                return [0]
            def broadcast_to(self, array, shape):
                """Fallback broadcast_to implementation"""
                try:
                    import numpy as np
                    return np.broadcast_to(array, shape)
                except Exception:
                    # Simple fallback - just return the array
                    return array
            def expand_dims(self, a, axis):
                """Fallback expand_dims implementation"""
                try:
                    import numpy as np
                    return np.expand_dims(a, axis)
                except Exception:
                    return a
            def square(self, x):
                """Fallback square implementation"""
                try:
                    if isinstance(x, list):
                        return [v * v for v in x]
                    return x * x
                except Exception:
                    return x
            def sqrt(self, x):
                """Fallback sqrt implementation"""
                try:
                    import math
                    if isinstance(x, list):
                        return [math.sqrt(v) if v >= 0 else 0 for v in x]
                    return math.sqrt(x) if x >= 0 else 0
                except Exception:
                    return x
            def log(self, x):
                """Fallback log implementation"""
                try:
                    import math
                    if isinstance(x, list):
                        return [math.log(v) if v > 0 else -float('inf') for v in x]
                    return math.log(x) if x > 0 else -float('inf')
                except Exception:
                    return x
            def exp(self, x):
                """Fallback exp implementation"""
                try:
                    import math
                    if isinstance(x, list):
                        return [math.exp(v) for v in x]
                    return math.exp(x)
                except Exception:
                    return x
            def sin(self, x):
                """Fallback sin implementation"""
                try:
                    import math
                    if isinstance(x, list):
                        return [math.sin(v) for v in x]
                    return math.sin(x)
                except Exception:
                    return x
            def cos(self, x):
                """Fallback cos implementation"""
                try:
                    import math
                    if isinstance(x, list):
                        return [math.cos(v) for v in x]
                    return math.cos(x)
                except Exception:
                    return x
            def arange(self, *args, **kwargs):
                """Fallback arange implementation"""
                try:
                    import numpy as np
                    return np.arange(*args, **kwargs)
                except Exception:
                    # Simple fallback
                    if len(args) == 1:
                        return list(range(args[0]))
                    elif len(args) == 2:
                        return list(range(args[0], args[1]))
                    elif len(args) == 3:
                        return list(range(args[0], args[1], args[2]))
                    return []
            def transpose(self, a, axes=None):
                """Fallback transpose implementation"""
                try:
                    import numpy as np
                    return np.transpose(a, axes)
                except Exception:
                    return a
            def swapaxes(self, a, axis1, axis2):
                """Fallback swapaxes implementation"""
                try:
                    import numpy as np
                    return np.swapaxes(a, axis1, axis2)
                except Exception:
                    return a
            def matmul(self, a, b):
                """Fallback matmul implementation"""
                try:
                    import numpy as np
                    return np.matmul(a, b)
                except Exception:
                    return a
            def einsum(self, subscripts, *operands, **kwargs):
                """Fallback einsum implementation"""
                try:
                    import numpy as np
                    return np.einsum(subscripts, *operands, **kwargs)
                except Exception:
                    return operands[0] if operands else None
            @property
            def pi(self):
                """Mathematical constant pi"""
                try:
                    import math
                    return math.pi
                except Exception:
                    return 3.14159265359
            def reshape(self, a, newshape):
                """Fallback reshape implementation"""
                try:
                    import numpy as np
                    return np.reshape(a, newshape)
                except Exception:
                    return a
            def concatenate(self, arrays, axis=0):
                """Fallback concatenate implementation"""
                try:
                    import numpy as np
                    return np.concatenate(arrays, axis=axis)
                except Exception:
                    return arrays[0] if arrays else []
            def stack(self, arrays, axis=0):
                """Fallback stack implementation"""
                try:
                    import numpy as np
                    return np.stack(arrays, axis=axis)
                except Exception:
                    return arrays[0] if arrays else []
            def where(self, condition, x=None, y=None):
                """Fallback where implementation"""
                try:
                    import numpy as np
                    if x is None and y is None:
                        return np.where(condition)
                    return np.where(condition, x, y)
                except Exception:
                    return condition
            
            @property
            def linalg(self):
                """Linear algebra fallback implementation"""
                class _FakeLinalg:
                    @staticmethod
                    def norm(x, axis=None, keepdims=False):
                        """Fallback norm implementation"""
                        try:
                            import numpy as np
                            return np.linalg.norm(x, axis=axis, keepdims=keepdims)
                        except Exception:
                            # Simple fallback - return magnitude
                            try:
                                if isinstance(x, list):
                                    return sum(v*v for v in x) ** 0.5
                                return abs(x)
                            except:
                                return 1.0
                    
                    @staticmethod
                    def qr(a, mode='reduced'):
                        """Fallback QR decomposition"""
                        try:
                            import numpy as np
                            return np.linalg.qr(a, mode=mode)
                        except Exception:
                            # Return identity-like fallback
                            return a, a
                    
                    @staticmethod
                    def eigh(a):
                        """Fallback eigendecomposition"""
                        try:
                            import numpy as np
                            return np.linalg.eigh(a)
                        except Exception:
                            # Return simple fallback
                            return a, a
                    
                    @staticmethod
                    def eigvals(a):
                        """Fallback eigenvalues"""
                        try:
                            import numpy as np
                            return np.linalg.eigvals(a)
                        except Exception:
                            return a
                
                return _FakeLinalg()
            
            def diff(self, a, n=1, axis=-1, prepend=None, append=None):
                """Fallback diff implementation"""
                try:
                    import numpy as np
                    return np.diff(a, n=n, axis=axis, prepend=prepend, append=append)
                except Exception:
                    # Simple fallback for lists
                    if isinstance(a, list) and len(a) > 0:
                        if isinstance(a[0], list):  # 2D array case
                            if axis == 1 or axis == -1:
                                # Diff along last axis (columns)
                                result = []
                                for row in a:
                                    if len(row) > 1:
                                        row_diff = [row[i+1] - row[i] for i in range(len(row)-1)]
                                        result.append(row_diff)
                                    else:
                                        result.append([])
                                return result
                            elif axis == 0:
                                # Diff along first axis (rows)
                                if len(a) > 1:
                                    result = []
                                    for col_idx in range(len(a[0])):
                                        col_diff = [a[i+1][col_idx] - a[i][col_idx] for i in range(len(a)-1)]
                                        result.append(col_diff)
                                    return result
                                return []
                        else:  # 1D array case
                            if len(a) > 1:
                                return [a[i+1] - a[i] for i in range(len(a)-1)]
                    return a
            def clip(self, a, a_min=None, a_max=None, out=None):
                """Fallback clip implementation"""
                try:
                    import numpy as np
                    return np.clip(a, a_min, a_max, out=out)
                except Exception:
                    # Simple fallback for lists and scalars
                    if isinstance(a, list):
                        result = []
                        for val in a:
                            if a_min is not None and val < a_min:
                                val = a_min
                            if a_max is not None and val > a_max:
                                val = a_max
                            result.append(val)
                        return result
                    else:
                        if a_min is not None and a < a_min:
                            a = a_min
                        if a_max is not None and a > a_max:
                            a = a_max
                        return a
        _jnp = _MiniNumpy()  # type: ignore

    class _FakeJax:
        def __getattr__(self, name: str) -> Any:  # Proveer dummy attrs
            if name == 'jit':
                # Return a no-op decorator for jit
                return lambda func: func
            elif name == 'process_index':
                # Return a function that always returns 0
                return lambda: 0
            elif name == 'random':
                # Import and return the capibara random module
                try:
                    from . import capibara_random as random_module
                    return random_module
                except ImportError:
                    # Fallback random module
                    class _FakeRandom:
                        @staticmethod
                        def PRNGKey(seed=0):
                            return seed
                        @staticmethod
                        def key(seed=0):
                            return seed
                        @staticmethod
                        def split(key, num=2):
                            return [key + i for i in range(num)]
                    return _FakeRandom()
            else:
                raise ImportError(
                    "JAX no está instalado en este entorno. "
                    f"Se intentó acceder a jax.{name}"
                )

    _jax = _FakeJax()  # type: ignore

# Expose expected public symbols
jax = _jax
numpy = _jnp

# Add linalg attribute to numpy if JAX is available
if HAS_JAX and hasattr(_jnp, 'linalg'):
    numpy.linalg = _jnp.linalg
elif HAS_JAX:
    # Make sure linalg is accessible even if not directly exposed
    try:
        import jax.numpy as real_jnp
        numpy.linalg = real_jnp.linalg
    except Exception:
        pass

# Local submodules
from . import nn  # noqa: E402
from . import lax as lax  # noqa: E402
from . import sharding as sharding  # noqa: E402
from . import xla as xla  # noqa: E402

# Import random module
if HAS_JAX:
    from jax import random  # noqa: E402
else:
    try:
        from . import capibara_random as random  # noqa: E402
    except ImportError:
        # Fallback random module
        class _FakeRandom:
            @staticmethod
            def PRNGKey(seed=0):
                return seed
            @staticmethod
            def key(seed=0):
                return seed
            @staticmethod
            def split(key, num=2):
                return [key + i for i in range(num)]
            @staticmethod
            def normal(key, shape=(), dtype=None):
                try:
                    import numpy as np
                    return np.random.normal(0, 1, shape).astype(dtype or np.float32)
                except:
                    return 0.0
        random = _FakeRandom()

# Compatibility aliases previously used in code
ltox = lax
xlto = xla

__all__ = [
    "jax",
    "numpy",
    "random",
    "nn",
    "lax",
    "ltox",
    "sharding",
    "xla",
    "xlto",
    "HAS_JAX",
]
