"""Stubs mejorados para Flax con tipos precisos."""
from typing import (Any, Callable, Dict, List, Optional, Protocol, Sequence, 
                   Tuple, TypeVar, Union, runtime_checkable)
import numpy as np
from capibara.jax import jax
from capibara.jax.numpy import Array
from capibara.jax.random import PRNGKey

T = TypeVar('T')
Shape = Tuple[int, ...]
Dtype = np.dtype
ParameterDict = Dict[str, Dict[str, Array]]
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]

# Protocolos base ------------------------------------------------------
@runtime_checkable
class Module(Protocol):
    name: str
    parent: Optional['Module']
    
    def setup(self) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def init(self, rng: PRNGKey, *args: Any, **kwargs: Any) -> ParameterDict: ...
    def apply(self, variables: ParameterDict, *args: Any, **kwargs: Any) -> Array: ...
    def lazy_init(self, rng: PRNGKey, *args: Any, **kwargs: Any) -> ParameterDict: ...

# Capas comunes --------------------------------------------------------
class Dense(Module):
    features: int
    use_bias: bool
    kernel_init: Initializer
    bias_init: Initializer
    dtype: Dtype
    param_dtype: Dtype
    
    def __call__(self, inputs: Array) -> Array: ...

class LayerNorm(Module):
    epsilon: float
    dtype: Dtype
    use_bias: bool
    use_scale: bool
    bias_init: Initializer
    scale_init: Initializer
    
    def __call__(self, x: Array) -> Array: ...

class Dropout(Module):
    rate: float
    broadcast_dims: Sequence[int]
    
    def __call__(self, x: Array, deterministic: bool) -> Array: ...

# Decoradores ----------------------------------------------------------
def compact(fn: Callable[..., Any]) -> Callable[..., Any]: ...

def remat(
    fn: Callable[..., Any],
    prevent_cse: bool = True,
    static_argnums: Union[int, Sequence[int]] = (),
) -> Callable[..., Any]: ...

# Inicializadores ------------------------------------------------------
class initializers:
    @staticmethod
    def zeros() -> Initializer: ...
    @staticmethod
    def ones() -> Initializer: ...
    @staticmethod
    def uniform(scale: float = 0.01) -> Initializer: ...
    @staticmethod
    def normal(stddev: float = 0.01) -> Initializer: ...