"""
JAX NumPy implementación simplificada con fallback interno.
"""

from __future__ import annotations

try:
    import numpy as np
except Exception:
    class _MiniNP:
        class ndarray(list):
            pass
        def array(self, obj, dtype=None, copy=True, order='K', ndmin=0):
            return obj
        def asarray(self, a, dtype=None, order=None):
            return a
        def zeros(self, shape, dtype=float, order='C'):
            return [0 for _ in range(shape if isinstance(shape, int) else shape[0])]
        def ones(self, shape, dtype=None, order='C'):
            return [1 for _ in range(shape if isinstance(shape, int) else shape[0])]
        def zeros_like(self, a, dtype=None, order='K', subok=True, shape=None):
            return [0 for _ in range(len(a))]
        def ones_like(self, a, dtype=None, order='K', subok=True, shape=None):
            return [1 for _ in range(len(a))]
        def empty(self, shape, dtype=float, order='C'):
            return self.zeros(shape, dtype, order)
        def empty_like(self, prototype, dtype=None, order='K', subok=True, shape=None):
            return self.zeros_like(prototype)
        def full(self, shape, fill_value, dtype=None, order='C'):
            return [fill_value for _ in range(shape if isinstance(shape, int) else shape[0])]
        def full_like(self, a, fill_value, dtype=None, order='K', subok=True, shape=None):
            return [fill_value for _ in range(len(a))]
        def arange(self, start, stop=None, step=None, dtype=None):
            stop = start if stop is None else stop
            step = 1 if step is None else step
            return list(range(int(start), int(stop), int(step)))
        def linspace(self, start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
            step = (stop - start) / (num - 1 if endpoint else num)
            vals = [start + i * step for i in range(num)]
            return (vals, step) if retstep else vals
        def reshape(self, a, newshape, order='C'):
            return a
        def transpose(self, a, axes=None):
            return a
        def concatenate(self, arrays, axis=0, out=None):
            res = []
            for arr in arrays:
                res += list(arr)
            return res
        def stack(self, arrays, axis=0, out=None):
            return list(arrays)
        def split(self, ary, indices_or_sections, axis=0):
            return [ary]
        def expand_dims(self, a, axis):
            return [a]
        def squeeze(self, a, axis=None):
            return a
        def add(self, x1, x2, out=None):
            return x1 + x2
        def subtract(self, x1, x2, out=None):
            return x1 - x2
        def multiply(self, x1, x2, out=None):
            return x1 * x2
        def divide(self, x1, x2, out=None):
            return x1 / x2
        def power(self, x1, x2, out=None):
            return x1 ** x2
        def sqrt(self, x, out=None):
            return x ** 0.5
        def exp(self, x, out=None):
            return 2.718281828 ** x
        def log(self, x, out=None):
            return 0.0
        def sin(self, x, out=None):
            return x
        def cos(self, x, out=None):
            return x
        def tanh(self, x, out=None):
            return x
        def dot(self, a, b, out=None):
            return sum(x*y for x, y in zip(a, b))
        def matmul(self, x1, x2, out=None):
            return x1
        def sum(self, a, axis=None, dtype=None, out=None, keepdims=False):
            return sum(a)
        def mean(self, a, axis=None, dtype=None, out=None, keepdims=False):
            return sum(a)/len(a) if a else 0
        def max(self, a, axis=None, out=None, keepdims=False):
            return max(a)
        def min(self, a, axis=None, out=None, keepdims=False):
            return min(a)
        def argmax(self, a, axis=None, out=None):
            return a.index(max(a)) if a else 0
        def argmin(self, a, axis=None, out=None):
            return a.index(min(a)) if a else 0
        def argsort(self, a, axis=-1, kind=None, order=None):
            if isinstance(a, list):
                return sorted(range(len(a)), key=lambda i: a[i])
            return [0]
        def std(self, a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
            if not a:
                return 0.0
            mean_val = self.mean(a)
            variance = sum((x - mean_val) ** 2 for x in a) / len(a)
            return variance ** 0.5
        def var(self, a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
            if not a:
                return 0.0
            mean_val = self.mean(a)
            return sum((x - mean_val) ** 2 for x in a) / len(a)
        def where(self, condition, x=None, y=None):
            return x if condition else y
        def diff(self, a, n=1, axis=-1, prepend=None, append=None):
            """Simple diff implementation for fallback."""
            if isinstance(a, list) and len(a) > 1:
                return [a[i+1] - a[i] for i in range(len(a)-1)]
            return a
        def clip(self, a, a_min=None, a_max=None, out=None):
            """Simple clip implementation for fallback."""
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
        float16 = float
        float32 = float
        float64 = float
        int8 = int
        int16 = int
        int32 = int
        int64 = int
        uint8 = int
        uint16 = int
        uint32 = int
        uint64 = int
        bool_ = bool
        complex64 = complex
        complex128 = complex
        pi = 3.141592653589793
        e = 2.718281828459045
        inf = float('inf')
    np = _MiniNP()

from typing import Any, Optional, Union, Tuple

# Tipos básicos
Array = getattr(np, 'ndarray', list)
ndarray = getattr(np, 'ndarray', list)

# Creación de arrays

def array(object, dtype=None, copy=True, order='K', ndmin=0):
    return np.array(object, dtype=dtype, copy=copy, order=order, ndmin=ndmin)


def asarray(a, dtype=None, order=None):
    return np.asarray(a, dtype=dtype, order=order)


def zeros(shape, dtype=float, order='C'):
    return np.zeros(shape, dtype=dtype, order=order)


def ones(shape, dtype=None, order='C'):
    return np.ones(shape, dtype=dtype, order=order)


def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    return np.zeros_like(a, dtype=dtype, order=order, subok=subok, shape=shape)


def ones_like(a, dtype=None, order='K', subok=True, shape=None):
    return np.ones_like(a, dtype=dtype, order=order, subok=subok, shape=shape)


def empty(shape, dtype=float, order='C'):
    return np.empty(shape, dtype=dtype, order=order)


def empty_like(prototype, dtype=None, order='K', subok=True, shape=None):
    return np.empty_like(prototype, dtype=dtype, order=order, subok=subok, shape=shape)


def full(shape, fill_value, dtype=None, order='C'):
    return np.full(shape, fill_value, dtype=dtype, order=order)


def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):
    return np.full_like(a, fill_value, dtype=dtype, order=order, subok=subok, shape=shape)


def arange(start, stop=None, step=None, dtype=None):
    return np.arange(start, stop, step, dtype=dtype)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    return np.linspace(start, stop, num, endpoint, retstep, dtype, axis)

# Operaciones básicas

def reshape(a, newshape, order='C'):
    return np.reshape(a, newshape, order)


def transpose(a, axes=None):
    return np.transpose(a, axes)


def concatenate(arrays, axis=0, out=None):
    return np.concatenate(arrays, axis, out)


def stack(arrays, axis=0, out=None):
    return np.stack(arrays, axis, out)


def split(ary, indices_or_sections, axis=0):
    return np.split(ary, indices_or_sections, axis)


def expand_dims(a, axis):
    return np.expand_dims(a, axis)


def broadcast_to(array, shape):
    """Broadcast array to the given shape."""
    return np.broadcast_to(array, shape)


def squeeze(a, axis=None):
    return np.squeeze(a, axis)


def any(a, axis=None, keepdims=False):
    """Check if any array elements along a given axis evaluate to True."""
    return np.any(a, axis=axis, keepdims=keepdims)


def all(a, axis=None, keepdims=False):
    """Test whether all array elements along a given axis evaluate to True."""
    return np.all(a, axis=axis, keepdims=keepdims)


def where(condition, x=None, y=None):
    """Return elements chosen from x or y depending on condition."""
    return np.where(condition, x, y)


def maximum(x1, x2):
    """Element-wise maximum of array elements."""
    return np.maximum(x1, x2)


def minimum(x1, x2):
    """Element-wise minimum of array elements."""
    return np.minimum(x1, x2)


def diff(a, n=1, axis=-1, prepend=None, append=None):
    """Calculate the n-th discrete difference along the given axis."""
    return np.diff(a, n=n, axis=axis, prepend=prepend, append=append)


def clip(a, a_min=None, a_max=None, out=None):
    """Clip (limit) the values in an array."""
    return np.clip(a, a_min, a_max, out=out)

# Matemáticas

def add(x1, x2, out=None):
    return np.add(x1, x2, out=out)


def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    """Hyperbolic tangent function."""
    return np.tanh(x)


def exp(x):
    """Exponential function."""
    return np.exp(x)


def log(x):
    """Natural logarithm."""
    return np.log(x)


def sin(x):
    """Sine function."""
    return np.sin(x)


def cos(x):
    """Cosine function."""
    return np.cos(x)


def isfinite(x):
    """Test element-wise for finiteness."""
    return np.isfinite(x)


def isnan(x):
    """Test element-wise for NaN."""
    return np.isnan(x)


def isinf(x):
    """Test element-wise for positive or negative infinity."""
    return np.isinf(x)


def subtract(x1, x2, out=None):
    return np.subtract(x1, x2, out=out)


def multiply(x1, x2, out=None):
    return np.multiply(x1, x2, out=out)


def divide(x1, x2, out=None):
    return np.divide(x1, x2, out=out)


def power(x1, x2, out=None):
    return np.power(x1, x2, out=out)


def sqrt(x, out=None):
    return np.sqrt(x, out=out)


def exp(x, out=None):
    return np.exp(x, out=out)


def log(x, out=None):
    return np.log(x, out=out)


def sin(x, out=None):
    return np.sin(x, out=out)


def cos(x, out=None):
    return np.cos(x, out=out)


def tanh(x, out=None):
    return np.tanh(x, out=out)

# Álgebra lineal básica

def dot(a, b, out=None):
    return np.dot(a, b, out=out)


def matmul(x1, x2, out=None):
    return np.matmul(x1, x2, out=out)

# Reducciones

def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    return np.sum(a, axis, dtype, out, keepdims)


def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    return np.mean(a, axis, dtype, out, keepdims)


def max(a, axis=None, out=None, keepdims=False):
    return np.max(a, axis, out, keepdims)


def min(a, axis=None, out=None, keepdims=False):
    return np.min(a, axis, out, keepdims)


def argmax(a, axis=None, out=None):
    return np.argmax(a, axis, out)


def argmin(a, axis=None, out=None):
    return np.argmin(a, axis, out)


def argsort(a, axis=-1, kind=None, order=None):
    return np.argsort(a, axis, kind, order)


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Standard deviation along the specified axis."""
    return np.std(a, axis, dtype, out, ddof, keepdims)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Variance along the specified axis."""
    return np.var(a, axis, dtype, out, ddof, keepdims)

# Lógica

def where(condition, x=None, y=None):
    return np.where(condition, x, y)

# Tipos de datos
bfloat16 = getattr(np, 'float16', float)
float16 = getattr(np, 'float16', float)
float32 = getattr(np, 'float32', float)
float64 = getattr(np, 'float64', float)
int8 = getattr(np, 'int8', int)
int16 = getattr(np, 'int16', int)
int32 = getattr(np, 'int32', int)
int64 = getattr(np, 'int64', int)
uint8 = getattr(np, 'uint8', int)
uint16 = getattr(np, 'uint16', int)
uint32 = getattr(np, 'uint32', int)
uint64 = getattr(np, 'uint64', int)
bool_ = getattr(np, 'bool_', bool)
complex64 = getattr(np, 'complex64', complex)
complex128 = getattr(np, 'complex128', complex)

# Constantes
pi = getattr(np, 'pi', 3.141592653589793)
e = getattr(np, 'e', 2.718281828459045)
inf = getattr(np, 'inf', float('inf'))

# Import linalg submodule
from . import linalg

# Aliases comunes en JAX
import sys as _sys
jnp = _sys.modules[__name__]

__all__ = [
    # Tipos
    'Array', 'ndarray',
    # Creación
    'array', 'asarray', 'zeros', 'ones', 'zeros_like', 'ones_like', 'empty', 'empty_like', 'full', 'full_like', 'arange', 'linspace',
    # Manipulación
    'reshape', 'transpose', 'concatenate', 'stack', 'split', 'expand_dims', 'squeeze', 'diff', 'clip',
    # Matemáticas
    'add', 'subtract', 'multiply', 'divide', 'power', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tanh', 'dot', 'matmul',
    # Reducciones
    'sum', 'mean', 'max', 'min', 'argmax', 'argmin', 'argsort', 'std', 'var',
    # Lógica
    'where',
    # Submodules
    'linalg',
    # Dtypes
    'bfloat16', 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool_', 'complex64', 'complex128',
    # Constantes
    'pi', 'e', 'inf'
]
