"""Type stubs for JAX core functionality."""

from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, TypeVar, Union,
    overload, runtime_checkable
)
import numpy as np
from typing_extensions import Literal

__all__ = [
    'Array', 'PRNGKey', 'Shape', 'Dtype',
    'random', 'numpy', 'lax', 'errors',
    'random_key', 'random_split', 'random_normal', 'random_uniform',
    'vmap', 'jit', 'grad', 'value_and_grad', 'config'
]

T = TypeVar('T')
Shape = Tuple[int, ...]
Dtype = np.dtype

@runtime_checkable
class Array(Protocol):
    """Protocol for JAX array objects."""
    
    shape: Shape
    dtype: Dtype
    ndim: int
    size: int
    
    def __array__(self, dtype: Optional[Dtype] = None) -> np.ndarray:
        """Convert to NumPy array."""
        ...
    
    def __getitem__(self, key: Any) -> 'Array':
        """Array indexing."""
        ...
    
    def __setitem__(self, key: Any, value: Any) -> None:
        """In-place array modification."""
        ...
    
    def reshape(self, *shape: int) -> 'Array':
        """Reshape the array."""
        ...
    
    def astype(self, dtype: Dtype) -> 'Array':
        """Convert array to different dtype."""
        ...
    
    def transpose(self, *axes: int) -> 'Array':
        """Transpose array dimensions."""
        ...

class PRNGKey(Array):
    """Pseudorandom number generator key."""
    ...

# Core modules
class JaxRandomModule(Protocol):
    """Type stub for jax.random module."""
    
    def normal(self, key: PRNGKey, shape: Shape, dtype: Dtype = ...) -> Array:
        ...
    
    def uniform(self, key: PRNGKey, shape: Shape, dtype: Dtype = ...) -> Array:
        ...
    
    def split(self, key: PRNGKey, num: int) -> Tuple[PRNGKey, ...]:
        ...

random: JaxRandomModule = ...
numpy: Any = ...  # Would be more precise in a complete stub
lax: Any = ...    # Would be more precise in a complete stub
errors: Any = ...

# Random functions
def random_key(seed: int) -> PRNGKey:
    """Creates a pseudorandom number generator key from an integer seed."""
    ...

def random_split(key: PRNGKey, num: int = 2) -> Tuple[PRNGKey, ...]:
    """Splits a PRNG key into multiple subkeys."""
    ...

def random_normal(
    key: PRNGKey,
    shape: Shape,
    dtype: Dtype = np.float32
) -> Array:
    """Generates random values from a normal distribution."""
    ...

def random_uniform(
    key: PRNGKey,
    shape: Shape,
    dtype: Dtype = np.float32,
    minval: float = 0.0,
    maxval: float = 1.0
) -> Array:
    """Generates random values from a uniform distribution."""
    ...

# Transformations
@overload
def vmap(
    fun: Callable[..., T],
    in_axes: Union[int, Sequence[Optional[int]]] = 0,
    out_axes: Union[int, Sequence[int]] = 0
) -> Callable[..., Union[T, Tuple[T, ...]]]: ...

@overload
def vmap(
    fun: Callable[..., Tuple[T, ...]],
    in_axes: Union[int, Sequence[Optional[int]]] = 0,
    out_axes: Union[int, Sequence[int]] = 0
) -> Callable[..., Tuple[T, ...]]: ...

def vmap(fun, in_axes=0, out_axes=0):
    """Vectorizes a function along specified axes."""
    ...

def jit(
    fun: Callable[..., T],
    static_argnums: Union[int, Sequence[int]] = (),
    static_argnames: Union[str, Sequence[str]] = (),
    donate_argnums: Union[int, Sequence[int]] = ()
) -> Callable[..., T]:
    """Compiles a function using XLA."""
    ...

def grad(
    fun: Callable[..., T],
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False
) -> Callable[..., Any]:
    """Computes the gradient of a function."""
    ...

def value_and_grad(
    fun: Callable[..., T],
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False
) -> Callable[..., Tuple[T, Any]]:
    """Computes both the value and gradient of a function."""
    ...

# Configuration
class config:
    """JAX configuration settings."""
    
    @staticmethod
    def enable_memory_stats() -> None:
        """Enable memory statistics tracking."""
        ...
    
    @staticmethod
    def disable_memory_stats() -> None:
        """Disable memory statistics tracking."""
        ...
    
    @staticmethod
    def read(key: str) -> Any:
        """Read a configuration value."""
        ...
    
    @staticmethod
    def update(key: str, value: Any) -> None:
        """Update a configuration value."""
        ...
    
    @staticmethod
    def get(key: str) -> Any:
        """Alias for read()."""
        ...
    
    @staticmethod
    def set(key: str, value: Any) -> None:
        """Alias for update()."""
        ...