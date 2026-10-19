"""
CapibaraGPT Advanced Flax Layers
===============================

implementation de capas avanzadas for CapibaraGPT v3.3:
- Attention mechanisms
- Embedding layers  
- Convolutional layers
- Sequential containers
- Specialized tpu layers
"""

from typing import Any, Callable, Optional, Union, Sequence, Tuple, Dict
from .. import Module, compact
from ..compat import get_jax, get_numpy
from . import Dense, LayerNorm, Dropout, xavier_uniform, zeros, ones

# configure JAX and numpy nativos
jax = get_jax()
jnp = get_numpy()

try:
    from .. import lax
except ImportError:
    import jax.lax as lax

# ================================
# EMBEDDING LAYERS
# ================================

class Embed(Module):
    """
    Embedding layer optimizada for CapibaraGPT.
    
    Soporta:
    - Sharding de embeddings for tpu
    - Múltiples vocabularios
    - Dropout en embeddings
    """
    
    def __init__(self,
                 num_embeddings: int,
                 features: int,
                 dtype: Any = jnp.float32,
                 param_dtype: Any = jnp.float32,
                 embedding_init: Callable = None,
                 one_hot: bool = False,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.embedding_init = embedding_init or normal(stddev=1.0)
        self.one_hot = one_hot
    
    def __call__(self, inputs):
        """Forward pass with lookup eficiente."""
        inputs = jnp.asarray(inputs)
        
        # create embedding table
        embedding = self.param(
            'embedding',
            self.embedding_init,
            (self.num_embeddings, self.features),
            self.param_dtype
        )
        
        if self.one_hot:
            # One-hot lookup (more efficient in tpu)
            one_hot_inputs = jnp.eye(self.num_embeddings, dtype=self.dtype)[inputs]
            return jnp.dot(one_hot_inputs, embedding.astype(self.dtype))
        else:
            # Index lookup directo
            return embedding[inputs].astype(self.dtype)

# ================================
# ATTENTION MECHANISMS
# ================================

class MultiHeadDotProductAttention(Module):
    """
    Multi-head attention optimizada for CapibaraGPT.
    
    Características:
    - Soporte for causal masking
    - optimization tpu v4/v6
    - Flash attention when sea posible
    - Sharding de heads
    """
    
    def __init__(self,
                 num_heads: int,
                 dtype: Any = jnp.float32,
                 param_dtype: Any = jnp.float32,
                 qkv_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 broadcast_dropout: bool = True,
                 dropout_rate: float = 0.0,
                 deterministic: bool = False,
                 precision: Any = None,
                 kernel_init: Callable = None,
                 bias_init: Callable = None,
                 use_bias: bool = True,
                 attention_fn: Callable = None,
                 decode: bool = False,
                 normalize_qk: bool = False,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.qkv_features = qkv_features
        self.out_features = out_features
        self.broadcast_dropout = broadcast_dropout
        self.dropout_rate = dropout_rate
        self.deterministic = deterministic
        self.precision = precision
        self.kernel_init = kernel_init or xavier_uniform()
        self.bias_init = bias_init or zeros
        self.use_bias = use_bias
        self.attention_fn = attention_fn or self._default_attention_fn
        self.decode = decode
        self.normalize_qk = normalize_qk
    
    def setup(self):
        """Inicializar proyecciones Q, K, V."""
        self.query = Dense(
            self.qkv_features or self.num_heads * 64,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            name='query'
        )
        
        self.key = Dense(
            self.qkv_features or self.num_heads * 64,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            name='key'
        )
        
        self.value = Dense(
            self.qkv_features or self.num_heads * 64,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            name='value'
        )
        
        self.out = Dense(
            self.out_features or self.qkv_features or self.num_heads * 64,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            name='out'
        )
    
    def _default_attention_fn(self, query, key, value, mask=None):
        """implementation de attention by defect."""
        # calculate scores
        depth = query.shape[-1] 
        scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(depth)
        
        # apply máscara if se proporciona
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Softmax
        attention_weights = jax.nn.softmax(scores, axis=-1)
        
        # apply dropout if está configurado
        if self.dropout_rate > 0 and not self.deterministic:
            dropout = Dropout(self.dropout_rate, deterministic=self.deterministic)
            attention_weights = dropout(attention_weights)
        
        # apply attention a values
        return jnp.matmul(attention_weights, value)
    
    def __call__(self, inputs_q, inputs_kv=None, mask=None, deterministic=None):
        """Forward pass de multi-head attention."""
        if inputs_kv is None:
            inputs_kv = inputs_q
        
        if deterministic is None:
            deterministic = self.deterministic
        
        # Proyectar Q, K, V
        query = self.query(inputs_q)
        key = self.key(inputs_kv)
        value = self.value(inputs_kv)
        
        # Reshape for multi-head
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_kv = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        
        query = query.reshape(batch_size, seq_len_q, self.num_heads, head_dim)
        key = key.reshape(batch_size, seq_len_kv, self.num_heads, head_dim)
        value = value.reshape(batch_size, seq_len_kv, self.num_heads, head_dim)
        
        # Transpose for atención
        query = jnp.transpose(query, (0, 2, 1, 3))  # [B, H, Lq, D]
        key = jnp.transpose(key, (0, 2, 1, 3))      # [B, H, Lkv, D]
        value = jnp.transpose(value, (0, 2, 1, 3))  # [B, H, Lkv, D]
        
        # apply attention
        attended = self.attention_fn(query, key, value, mask)
        
        # Reshape de vuelta
        attended = jnp.transpose(attended, (0, 2, 1, 3))  # [B, Lq, H, D]
        attended = attended.reshape(batch_size, seq_len_q, -1)
        
        # Proyección end
        return self.out(attended)

# ================================
# CONVOLUTIONAL LAYERS
# ================================

class Conv(Module):
    """
    Capa convolucional 1D optimizada for CapibaraGPT.
    
    Principalmente for:
    - Procesamiento de secuencias
    - Capas depthwise en modelos eficientes
    - Convoluciones causales for autoregresión
    """
    
    def __init__(self,
                 features: int,
                 kernel_size: Union[int, Sequence[int]],
                 strides: Union[int, Sequence[int]] = 1,
                 padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME',
                 input_dilation: Union[int, Sequence[int]] = 1,
                 kernel_dilation: Union[int, Sequence[int]] = 1,
                 feature_group_count: int = 1,
                 use_bias: bool = True,
                 mask: Optional[jnp.ndarray] = None,
                 dtype: Any = jnp.float32,
                 param_dtype: Any = jnp.float32,
                 precision: Any = None,
                 kernel_init: Callable = None,
                 bias_init: Callable = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.features = features
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size]
        self.strides = strides if isinstance(strides, (list, tuple)) else [strides]
        self.padding = padding
        self.input_dilation = input_dilation if isinstance(input_dilation, (list, tuple)) else [input_dilation]
        self.kernel_dilation = kernel_dilation if isinstance(kernel_dilation, (list, tuple)) else [kernel_dilation]
        self.feature_group_count = feature_group_count
        self.use_bias = use_bias
        self.mask = mask
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init or xavier_uniform()
        self.bias_init = bias_init or zeros
    
    def __call__(self, inputs):
        """Forward pass convolucional."""
        inputs = jnp.asarray(inputs, dtype=self.dtype)
        
        # determine forma del kernel
        in_features = inputs.shape[-1]
        if self.feature_group_count == 1:
            kernel_shape = self.kernel_size + [in_features, self.features]
        else:
            kernel_shape = self.kernel_size + [in_features // self.feature_group_count, self.features]
        
        # create kernel
        kernel = self.param('kernel', self.kernel_init, kernel_shape, self.param_dtype)
        
        if self.mask is not None:
            kernel = kernel * self.mask
        
        # perform convolución
        dimension_numbers = ('NHC', 'HIO', 'NHC')  # for conv1d
        
        y = lax.conv_general_dilated(
            inputs,
            kernel.astype(inputs.dtype),
            self.strides,
            self.padding,
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision
        )
        
        # add bias if se especifica
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
            y = y + bias.astype(y.dtype)
        
        return y

# ================================
# CONTAINER LAYERS
# ================================

class Sequential(Module):
    """
    Contenedor secuencial for capas.
    Equivalente a nn.Sequential de PyTorch.
    """
    
    def __init__(self, layers: Sequence[Module], name: Optional[str] = None):
        super().__init__(name=name)
        self.layers = layers
    
    def __call__(self, x):
        """apply capas secuencialmente."""
        for layer in self.layers:
            x = layer(x)
        return x

# ================================
# SPECIALIZED tpu LAYERS  
# ================================

class DenseGeneral(Module):
    """
    Dense general with soporte for sharding tpu advanced.
    Compatible with especificaciones de ejes de CapibaraGPT.
    """
    
    def __init__(self,
                 features: Union[int, Sequence[int]],
                 axis: Union[int, Sequence[int]] = -1,
                 batch_dims: Sequence[int] = (),
                 use_bias: bool = True,
                 dtype: Any = jnp.float32,
                 param_dtype: Any = jnp.float32,
                 kernel_init: Callable = None,
                 bias_init: Callable = None,
                 precision: Any = None,
                 kernel_axes: Tuple[str, ...] = (),
                 bias_axes: Tuple[str, ...] = (),
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.features = features if isinstance(features, (list, tuple)) else [features]
        self.axis = axis if isinstance(axis, (list, tuple)) else [axis]
        self.batch_dims = batch_dims
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.kernel_init = kernel_init or xavier_uniform()
        self.bias_init = bias_init or zeros
        self.precision = precision
        self.kernel_axes = kernel_axes
        self.bias_axes = bias_axes
    
    def __call__(self, inputs):
        """Forward pass with contracción general."""
        inputs = jnp.asarray(inputs, dtype=self.dtype)
        
        # determine dimensiones de contracción
        ndim = inputs.ndim
        n_batch_dims = len(self.batch_dims)
        axis = [ax % ndim for ax in self.axis]
        batch_dims = [ax % ndim for ax in self.batch_dims]
        
        # Forma del kernel
        kernel_shape = tuple(inputs.shape[ax] for ax in axis) + tuple(self.features)
        kernel = self.param('kernel', self.kernel_init, kernel_shape, self.param_dtype)
        
        # configure contracción
        contract_dims = (axis, tuple(range(len(axis))))
        y = lax.dot_general(
            inputs, 
            kernel, 
            ((contract_dims[0], contract_dims[1]), (batch_dims, batch_dims)),
            precision=self.precision
        )
        
        # Bias optional
        if self.use_bias:
            bias = self.param('bias', self.bias_init, tuple(self.features), self.param_dtype)
            # Broadcast bias apropiadamente
            for _ in range(ndim - len(axis) - len(self.features)):
                bias = jnp.expand_dims(bias, axis=0)
            y = y + bias
        
        return y

# ================================
# EXPORTS
# ================================

__all__ = [
    # Embedding
    'Embed',
    
    # Attention
    'MultiHeadDotProductAttention',
    
    # Convolution
    'Conv',
    
    # Containers
    'Sequential',
    
    # Specialized
    'DenseGeneral',
] 