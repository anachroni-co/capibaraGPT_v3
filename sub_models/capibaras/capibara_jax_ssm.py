"""SSM Ultra-Optimizado for tpu v4-32 with JAX and Flax."""

import logging

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental.mesh_utils import create_device_mesh
from typing import Optional, Tuple, Dict, Any
from functools import partial

logger = logging.getLogger(__name__)


def _build_device_mesh() -> Mesh:
    """Build a (batch, hidden) device mesh from the available devices.

    On TPU v4-32 this yields the full 32x1 mesh; on CPU/GPU it degrades
    gracefully to whatever devices exist (1x1 on a single host), so the
    module can be imported and unit-tested anywhere.
    """
    n = len(jax.devices())
    devices = create_device_mesh((n, 1))
    return Mesh(devices, axis_names=("batch", "hidden"))


_DEVICE_MESH: Optional[Mesh] = None


def get_device_mesh() -> Mesh:
    """Lazily create and cache the global device mesh (never at import time)."""
    global _DEVICE_MESH
    if _DEVICE_MESH is None:
        _DEVICE_MESH = _build_device_mesh()
    return _DEVICE_MESH

@partial(jax.jit, static_argnames=('config', 'training'))
def ssm_layer(params: Dict[str, Any], x: jnp.ndarray, context: jnp.ndarray, 
             config: Any, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Capa SSM optimizada for tpu.
    
    Args:
        params: Diccionario de parámetros del model
        x: tensor de input
        context: Estado inicial
        config: setup del model
        training: Modo de entrenamiento
        
    Returns:
        Tuple de (estado_final, salidas)
    """
    try:
        def scan_fn(carry, x_t):
            A, B, C, dt_proj = params['A'], params['B'], params['C'], params['dt_proj']
            delta = jnp.exp(dt_proj['kernel'] @ x_t + dt_proj['bias'])
            new_state = carry * jnp.exp(delta * A) + x_t @ B
            output = (new_state @ C).astype(x.dtype)
            return new_state, output
            
        return jax.lax.scan(scan_fn, context, x)
    except Exception as e:
        logger.error(f"Error en ssm_layer: {e}")
        raise

class TPUOptimizedSSM(nn.Module):
    """architecture SSM for entrenamiento distribuido en tpu."""
    
    hidden_size: int = 2048
    dropout_rate: float = 0.1
    use_glu: bool = True
    shard_axis: tuple = ('batch', 'hidden')

    def _validate_input(self, x: jnp.ndarray) -> None:
        """Valida las dimensiones and type de la input."""
        if x.ndim != 3:
            raise ValueError(f"Input debe ser 3D (batch, seq_len, dim), got {x.shape}")
        if jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)):
            raise ValueError("Input contiene valores NaN o Inf")

    def _convert_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte parámetros al type de tpu with caching."""
        if not hasattr(self, '_cached_params'):
            self._cached_params = {
                k: v.astype(jnp.bfloat16) if v.dtype == jnp.float32 else v
                for k, v in params.items()
            }
        return self._cached_params

    @nn.compact
    def __call__(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None, 
                training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass del model SSM.
        
        Args:
            x: tensor de input de forma (batch, seq_len, dim)
            context: Estado inicial optional
            training: Modo de entrenamiento
            
        Returns:
            Tuple de (salidas, estado_final)
        """
        try:
            self._validate_input(x)
            
            # setup de sharding
            hidden_shard = P(*self.shard_axis)
            batch_shard = P('batch', None)
            
            # Parámetros distribuidos
            A = self.param('A',
                          nn.initializers.zeros,
                          (self.hidden_size,),
                          sharding=hidden_shard)
            
            B = self.param('B',
                          nn.initializers.lecun_normal(),
                          (x.shape[-1], self.hidden_size),
                          sharding=hidden_shard)
            
            C = self.param('C',
                          nn.initializers.lecun_normal(),
                          (self.hidden_size, x.shape[-1]),
                          sharding=hidden_shard)
            
            dt_proj = nn.Dense(1,
                              kernel_init=nn.initializers.normal(0.02),
                              kernel_sharding=hidden_shard,
                              bias_sharding=hidden_shard,
                              name='dt_proj')

            # Normalización distribuida
            x = nn.LayerNorm(sharding=hidden_shard)(x)
            
            # SSM Distribuido
            initial_state = context if context is not None else jnp.zeros(
                (x.shape[0], self.hidden_size),
                dtype=x.dtype
            )
            
            params = dict(A=A, B=B, C=C, dt_proj=dt_proj.variables)
            params = self._convert_params(params)
            
            final_state, outputs = shard_map(
                partial(ssm_layer, config=self, training=training),
                mesh=get_device_mesh(),
                in_specs=(batch_shard, batch_shard),
                out_specs=(batch_shard, batch_shard),
                check_rep=False
            )(params, x, initial_state)

            # GLU Distribuido
            if self.use_glu:
                gate = nn.Dense(self.hidden_size,
                               sharding=hidden_shard,
                               name='gate')(outputs)
                outputs = outputs * jax.nn.sigmoid(gate)

            return nn.Dropout(self.dropout_rate, sharding=hidden_shard)(
                outputs, deterministic=not training
            ), final_state
            
        except Exception as e:
            logger.error(f"Error en forward pass: {e}")
            raise

# Bloque de Entrenamiento Distribuido
def train_step(state: Any, batch: Dict[str, jnp.ndarray], 
              context: Optional[jnp.ndarray] = None) -> Tuple[Any, jnp.ndarray]:
    """Paso de entrenamiento optimizado for tpu.
    
    Args:
        state: Estado current del model
        batch: Diccionario with inputs and targets
        context: Estado inicial optional
        
    Returns:
        Tuple de (nuevo_estado, pérdida)
    """
    try:
        def loss_fn(params):
            outputs, _ = TPUOptimizedSSM().apply(
                {'params': params},
                batch['inputs'],
                context,
                training=True
            )
            return jnp.mean((outputs - batch['targets']) ** 2)
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads), loss
        
    except Exception as e:
        logger.error(f"Error en train_step: {e}")
        raise

# setup de Paralelismo
def create_sharded_train_step():
    """Crea una function de entrenamiento distribuida."""
    return jax.jit(
        shard_map(
            train_step,
            mesh=get_device_mesh(),
            in_specs=(P('batch', 'hidden'), P('batch'), P('batch')),
            out_specs=(P('batch', 'hidden'), P()),
            check_rep=False
        ),
        donate_argnames=('state',)
    )

# initialization Optimizada
def init_model(key: jnp.ndarray, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """initialization de parámetros with sharding automático.
    
    Args:
        key: key aleatoria for initialization
        input_shape: Forma del tensor de input
        
    Returns:
        Variables inicializadas del model
    """
    try:
        model = TPUOptimizedSSM()
        variables = model.init(key, 
                             jnp.zeros(input_shape, dtype=jnp.bfloat16),
                             context=None)
        return variables
    except Exception as e:
        logger.error(f"Error en init_model: {e}")
        raise

# example de Uso en tpu Pod
if __name__ == "__main__":
    # setup de hiperparámetros
    BATCH_SIZE = 1024 * 32  # 32K ejemplos by paso
    SEQ_LEN = 2048
    HIDDEN_SIZE = 8192

    # initialization
    key = jax.random.PRNGKey(0)
    input_shape = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    
    # Particionado de data
    sharded_inputs = jax.device_put(
        jnp.zeros(input_shape, dtype=jnp.bfloat16),
        jax.sharding.NamedSharding(get_device_mesh(), P('batch', None, 'hidden'))
    )

    # Estado inicial del model
    state = init_model(key, input_shape)
    train_step_fn = create_sharded_train_step()

    # simulation de data de entrenamiento
    batch = {
        'inputs': sharded_inputs,
        'targets': sharded_inputs
    }
    
    # execution distribuida
    updated_state, loss = train_step_fn(state, batch, None)
    logger.info(f"Pérdida inicial: {loss}")
