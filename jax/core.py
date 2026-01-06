"""
jax core module.

# This module provides functionality for core.
"""

import logging
from typing import Any, Dict, List, Optional

# Standard library imports
import inspect
import functools
import threading
import itertools as it
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial, total_ordering
from collections import Counter, defaultdict, deque, namedtuple
from collections.abc import (
    Collection, Hashable, Iterable, Iterator,
    Sequence, MutableSet, MutableMapping, Callable, 
)
from typing import (
    overload, Union,
    Any, ClassVar, Generic, NamedTuple, TypeVar,
)

import warnings
import operator
import math

# Third-party imports - use standard numpy to avoid conflicts
try:
    import numpy as np
except ImportError:
    # Fallback if numpy is not available
    class MockArray:
        def __init__(self, data):
            self.shape = ()
            self.dtype = object
            
    class MockDtype:
        def __init__(self, name='object'):
            self.name = name
    
    class MockNumpy:
        ndarray = MockArray
        dtype = MockDtype
        def array(self, x): return MockArray(x)
    np = MockNumpy()

# Local imports
from . import config
from . import jax_typing as typing
from . import effects
from . import deprecations
from . import info_util
from . import linear_util as lu
from . import pretty_printer as pp
from .lib import jax_jit, xla_client
from .tree_util import tree_flatten, tree_unflatten

# ✅ DEFINICIONES DE ERRORES LOCALES
class InconclusiveDimensionOperation(ValueError):
    """error when dimension operation cannot be resolved."""
    pass

class ConcretizationTypeError(TypeError):
    """error when trying to concretize a tracer."""
    pass

class TracerArrayConversionError(ValueError):
    """error when converting tracer to array."""
    pass

class TracerBoolConversionError(ValueError):
    """error when converting tracer to bool."""
    pass

class TracerIntegerConversionError(ValueError):
    """error when converting tracer to integer."""
    pass

class UnexpectedTracerError(ValueError):
    """error when encountering unexpected tracer."""
    pass

class JaxprTypeError(TypeError):
    """error in JAXpr type checking."""
    pass

class ShardingTypeError(Exception):
    """error in sharding configuration."""
    pass

# ✅ TIPOS BÁSICOS
Array = Union[np.ndarray, 'Tracer']
DimSize = Union[int, 'Tracer']
Shape = tuple[DimSize, ...]
AxisName = Any
AxisEnv = dict[AxisName, int]
TracerType = TypeVar('TracerType', bound='Tracer')
Effects = frozenset['Effect']
no_effects = frozenset()

# ================== tpu v4-32 OPTIMIZED CONFIGURATIONS ==================

# ✅ CONFIG FLAGS with FALLBACKS
def _create_config_flag(name: str, default_value, help_text: str):
    """Create config flag with fallback."""
    try:
        if hasattr(config, 'int_flag'):
            return config.int_flag(name, default_value, help=help_text)
        elif hasattr(config, 'bool_flag'):
            return config.bool_flag(name, default_value, help=help_text)  
        elif hasattr(config, 'float_flag'):
            return config.float_flag(name, default_value, help=help_text)
        else:
            return type('ConfigFlag', (), {'value': default_value})()
    except:
        return type('ConfigFlag', (), {'value': default_value})()

_TRACER_ERROR_NUM_TRACEBACK_FRAMES = _create_config_flag(
    'jax_tracer_error_num_traceback_frames', 3,
    'Set the number of stack frames in JAX tracer error messages.'
)

_TPU_V4_MEMORY_FRACTION = _create_config_flag(
    'jax_tpu_v4_memory_fraction', 0.85,
    'Fraction of TPU v4 memory to use (0.85 for stability).'
)

_TPU_V4_ENABLE_ASYNC_COLLECTIVE = _create_config_flag(
    'jax_tpu_v4_async_collective', True,
    'Enable asynchronous collective operations on TPU v4.'
)

_TPU_V4_BFLOAT16_AUTO_CAST = _create_config_flag(
    'jax_tpu_v4_bfloat16_auto_cast', True,
    'Automatically cast operations to bfloat16 for TPU v4 efficiency.'
)

class TPUv4MeshConfigurations:
    """Dynamic tpu v4-32 mesh configurations for different workloads."""
    
    SINGLE_PROGRAM = {'mesh_shape': (1, 32), 'axis_names': ('data',)}
    DATA_PARALLEL = {'mesh_shape': (4, 8), 'axis_names': ('data', 'model')}
    MODEL_PARALLEL = {'mesh_shape': (8, 4), 'axis_names': ('model', 'data')}
    BALANCED = {'mesh_shape': (4, 8), 'axis_names': ('x', 'y')}
    CULTURAL_ANALYSIS = {'mesh_shape': (2, 16), 'axis_names': ('cultural_depth', 'semantic_width')}
    ADAPTIVE_CLASSICAL = {'mesh_shape': (8, 4), 'axis_names': ('adaptive_coherence', 'classical_compute')}
    SPIKING_NEURAL = {'mesh_shape': (1, 32), 'axis_names': ('temporal_steps',)}
    LARGE_CONTEXT = {'mesh_shape': (8, 4), 'axis_names': ('sequence', 'embedding')}
    
    @classmethod
    def get_optimal_config(cls, workload_type: str, model_size: int = None) -> dict[str, Any]:
        """Get optimal mesh configuration based on workload type and model size."""
        base_config = getattr(cls, workload_type.upper(), cls.BALANCED)
        if model_size is not None:
            if model_size > 100_000_000_000:  # 100B+ parameters
                base_config = cls.MODEL_PARALLEL
            elif model_size > 10_000_000_000:  # 10B+ parameters  
                base_config = cls.BALANCED
            else:  # < 10B parameters
                base_config = cls.DATA_PARALLEL
        return base_config.copy()

# ================== BASIC TYPES AND ABSTRACTIONS ==================

class Effect:
    """Base effect class."""
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return self.name

class AbstractValue:
    """Base class for abstract values."""
    __slots__: list[str] = []
    is_high = False
    mutable = False

    def to_tangent_aval(self):
        return self

    def at_least_vspace(self):
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def update_weak_type(self, weak_type):
        return self

    def strip_weak_type(self) -> 'AbstractValue':
        return self

    def normalize(self) -> 'AbstractValue':
        return self

    def update(self, **kwargs):
        return self

    def str_short(self, short_dtypes=False):
        return str(self)

def concrete_or_error(force: Any, val: Any, context=""):
    """Try to concretize a value or raise an error."""
    try:
        return force(val)
    except:
        raise ConcretizationTypeError(f"Cannot concretize value {val} in context: {context}")

def concrete_dim_or_error(val: Any, context=""):
    """Try to concretize a dimension or raise an error."""
    if isinstance(val, int):
        return val
    else:
        return concrete_or_error(operator.index, val, context=context)

def is_constant_dim(d: DimSize) -> bool:
    """Check if a dimension is constant."""
    return isinstance(d, int)

def is_constant_shape(s: Shape) -> bool:
    """Check if all dimensions in a shape are constant."""
    return all(is_constant_dim(d) for d in s)

def definitely_equal(x, y):
    """Check if two values are definitely equal."""
    try:
        return x == y
    except:
        return False

def definitely_equal_shape(s1: Shape, s2: Shape) -> bool:
    """Check if two shapes are definitely equal."""
    return (len(s1) == len(s2) and
            all(definitely_equal(d1, d2) for d1, d2 in zip(s1, s2)))

def canonicalize_shape(shape: Shape, context: str="") -> tuple[Any, ...]:
    """Canonicalize a shape."""
    return tuple(shape)

def canonicalize_dim(d: DimSize, context: str="") -> DimSize:
    """Canonicalize a dimension."""
    return d

# ================== PRIMITIVE OPERATIONS ==================

class Primitive:
    """Base class for JAX primitives."""
    name: str
    multiple_results: bool = False
    call_primitive: bool = False
    map_primitive: bool = False
    ref_primitive: bool = False
    skip_canonicalization: bool = False

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    def bind(self, *args, **params):
        return self._true_bind(*args, **params)

    def _true_bind(self, *args, **params):
        # simple implementation for now
        return self.impl(*args, **params)

    def bind_with_trace(self, trace, args, params):
        return trace.process_primitive(self, args, params)

    def def_impl(self, impl):
        self.impl = impl
        return impl

    def def_abstract_eval(self, abstract_eval):
        self.abstract_eval = abstract_eval
        return abstract_eval

    def def_effectful_abstract_eval(self, effectful_abstract_eval):
        self.effectful_abstract_eval = effectful_abstract_eval
        return effectful_abstract_eval

    def def_bind_with_trace(self, bind_with_trace):
        self.bind_with_trace = bind_with_trace
        return bind_with_trace

    def impl(self, *args, **params):
        raise NotImplementedError(f"No implementation for {self.name}")

    def abstract_eval(self, *args, **params):
        raise NotImplementedError(f"No abstract evaluation for {self.name}")

    def get_bind_params(self, params):
        return [], params

    def is_high(self, **params) -> bool:
        return False

# ================== TRACING SYSTEM ==================

class Trace(Generic[TracerType]):
    """Base tracing class."""
    __slots__ = ("__weakref__", "_invalidated", "_weakref", "requires_low")

    def __init__(self):
        self._invalidated = False
        self.requires_low = False

    def process_primitive(self, primitive, tracers, params):
        raise NotImplementedError

    def invalidate(self):
        self._invalidated = True

    def is_valid(self):
        return not self._invalidated

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def process_call(self, call_primitive, f, tracers, params):
        raise NotImplementedError

    def process_map(self, map_primitive, f, tracers, params):
        raise NotImplementedError

    def full_raise(self, x):
        return x

    @property
    def main(self):
        return self

class Tracer:
    """Base tracer class."""
    __slots__ = ['_trace']
    __hash__ = None

    def __init__(self, trace: Trace):
        self._trace = trace

    def __bool__(self):
        raise TracerBoolConversionError("Cannot convert tracer to bool")

    def __int__(self):
        raise TracerIntegerConversionError("Cannot convert tracer to int")

    def __float__(self):
        raise ConcretizationTypeError("Cannot convert tracer to float")

    def __index__(self):
        raise TracerIntegerConversionError("Cannot convert tracer to index")

class EvalTrace(Trace):
    """simple evaluation trace."""
    
    def process_primitive(self, primitive, args, params):
        return primitive.impl(*args, **params)

    def process_call(self, primitive, f, tracers, params):
        return primitive.impl(f, *tracers, **params)
    
    process_map = process_call

# ================== JAXPR REPRESENTATION ==================

class Var:
    """Variable in a JAXpr."""
    __slots__ = ["count", "suffix", "aval"]

    count: int
    suffix: str
    aval: AbstractValue

    _counter = it.count()

    def __init__(self, suffix: str, aval: AbstractValue):
        self.count = next(self._counter)
        self.suffix = suffix
        self.aval = aval

    def __repr__(self):
        return f'{self.suffix}{self.count}'

class Literal:
    """Literal value in a JAXpr."""
    __slots__ = ["val", "aval"]

    def __init__(self, val, aval):
        self.val = val
        self.aval = aval

    def __repr__(self):
        return f'Literal({self.val})'

Atom = Union[Var, Literal]

class JaxprEqn:
    """Equation in a JAXpr."""
    __slots__ = ['invars', 'outvars', 'primitive', 'params', 'effects']

    def __init__(self, invars, outvars, primitive, params, effects=no_effects):
        self.invars = invars
        self.outvars = outvars
        self.primitive = primitive
        self.params = params
        self.effects = effects

    def __repr__(self):
        return f'{self.outvars} = {self.primitive.name} {self.invars}'

class Jaxpr:
    """JAX expression."""
    __slots__ = ['constvars', 'invars', 'outvars', 'eqns', 'effects']

    def __init__(self, constvars: Sequence[Var], invars: Sequence[Var],
                 outvars: Sequence[Atom], eqns: Sequence[JaxprEqn],
                 effects: Effects = no_effects):
        self.constvars = list(constvars)
        self.invars = list(invars)
        self.outvars = list(outvars)
        self.eqns = list(eqns)
        self.effects = effects

    def __str__(self):
        return f"Jaxpr(invars={self.invars}, outvars={self.outvars}, eqns={self.eqns})"

    __repr__ = __str__

class ClosedJaxpr:
    """Closed JAX expression."""
    __slots__ = ['jaxpr', 'consts']

    def __init__(self, jaxpr: Jaxpr, consts: Sequence):
        self.jaxpr = jaxpr
        self.consts = list(consts)

    def __repr__(self):
        return f"ClosedJaxpr({self.jaxpr})"

# ================== SHAPE AND array ABSTRACTIONS ==================

def _dtype_object(dtype):
    """Convert dtype to object."""
    return np.dtype(dtype)

class UnshapedArray(AbstractValue):
    """Untyped array abstraction."""
    __slots__ = ['dtype', 'weak_type']
    array_abstraction_level = 4

    def __init__(self, dtype, weak_type=False):
        self.dtype = _dtype_object(dtype)
        self.weak_type = weak_type

    def __eq__(self, other):
        return (type(self) is type(other) and self.dtype == other.dtype and
                self.weak_type == other.weak_type)

    def __hash__(self):
        return hash((self.dtype, self.weak_type))

    def str_short(self, short_dtypes=False) -> str:
        return str(self.dtype)

class ShapedArray(UnshapedArray):
    """Shaped array abstraction."""
    __slots__ = ['shape']
    array_abstraction_level = 2

    def __init__(self, shape, dtype, weak_type=False):
        super().__init__(dtype, weak_type)
        self.shape = canonicalize_shape(shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return math.prod(self.shape) if self.shape else 1

    def __eq__(self, other):
        return (super().__eq__(other) and 
                definitely_equal_shape(self.shape, other.shape))

    def __hash__(self):
        return hash((super().__hash__(), self.shape))

    def str_short(self, short_dtypes=False) -> str:
        shape_str = ','.join(map(str, self.shape))
        dtype_name = getattr(self.dtype, 'name', str(self.dtype))
        return f'{dtype_name}[{shape_str}]'

# ================== UTILITY FUNCTIONS ==================

def abstractify(x):
    """Create abstract value from concrete value."""
    if isinstance(x, np.ndarray):
        return ShapedArray(x.shape, x.dtype)
    elif isinstance(x, (int, float, complex, bool)):
        return ShapedArray((), np.array(x).dtype)
    else:
        return x

def get_aval(x):
    """Get abstract value."""
    if hasattr(x, 'aval'):
        return x.aval
    else:
        return abstractify(x)

def is_concrete(x):
    """Check if value is concrete."""
    return not isinstance(x, Tracer)

def to_concrete_value(x):
    """Convert to concrete value."""
    if is_concrete(x):
        return x
    else:
        raise ConcretizationTypeError(f"Cannot concretize {x}")

# ================== EXPORTS ==================

__all__ = [
    # Core abstractions
    'AbstractValue', 'ShapedArray', 'UnshapedArray',
    'Tracer', 'Trace', 'EvalTrace',
    'Jaxpr', 'ClosedJaxpr', 'JaxprEqn',
    'Var', 'Literal', 'Atom',
    'Primitive', 'Effect', 'Effects',
    
    # Utility functions
    'abstractify', 'get_aval', 'is_concrete', 'to_concrete_value',
    'concrete_or_error', 'concrete_dim_or_error',
    'canonicalize_shape', 'canonicalize_dim',
    'is_constant_dim', 'is_constant_shape',
    'definitely_equal', 'definitely_equal_shape',
    
    # Types
    'Array', 'DimSize', 'Shape', 'AxisName', 'AxisEnv',
    'TracerType', 'no_effects',
    
    # Errors
    'InconclusiveDimensionOperation', 'ConcretizationTypeError',
    'TracerArrayConversionError', 'TracerBoolConversionError',
    'TracerIntegerConversionError', 'UnexpectedTracerError',
    'JaxprTypeError', 'ShardingTypeError',
    
    # tpu configurations
    'TPUv4MeshConfigurations',
]