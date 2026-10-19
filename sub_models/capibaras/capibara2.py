"""
Module that implements the Capibara2 model, a recurrent neural network using JAX/Flax.

This module provides an implementation of the Capibara2 model, which applies a
recurrent operation to input data efficiently. It includes functions for training
and evaluating the model.

Classes:
    Capibara2: Implements the Capibara2 model.

Functions:
    loss_fn: Computes the loss function.
    update: Performs a parameter update step.

Dependencies:
    - jax: For array operations and automatic differentiation.
    - flax: For neural network module definitions.
    - optax: For optimization algorithms.

Example:
    >>> config = Capibara2Config(hidden_size=256, dropout_rate=0.1)
    >>> model = Capibara2(config)
    >>> x = jnp.ones((32, 10, 256))  # batch, seq_len, dim
    >>> outputs = model(x, training=True)
"""

import os
import sys
import logging
# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../CapibaraGPT v3
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    pass  # Using proper imports instead of sys.path manipulation

import jax
from flax import linen as nn
from functools import partial
from dotenv import load_dotenv
from jax import numpy as jnp
from pydantic import BaseModel, Field
from .tpu_base_config import TPUBaseConfig

# Imports with fallbacks
try:
    from capibara.interfaces.isub_models import ISubModel
except ImportError:
    class ISubModel:
        pass

try:
    from capibara.config import CapibaraConfig
except ImportError:
    class CapibaraConfig:
        pass

try:
    from .capibara_jax_ssm import CapibaraJAXSSM, SSMConfig
except ImportError:
    class CapibaraJAXSSM:
        pass
    class SSMConfig:
        pass

try:
    from .capibara_byte import CapibaraByte, CapibaraByteConfig
except ImportError:
    class CapibaraByte:
        pass
    class CapibaraByteConfig:
        pass

from typing import Tuple, Optional, Dict, Any, Callable, Union, Literal

# Load environment variables
load_dotenv()

# Consistent logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Capibara2Config(TPUBaseConfig):
    """setup específica for Capibara2."""
    ssm_type: Literal["byte", "optimized_ssm"] = "optimized_ssm"
    byte_config: Optional[CapibaraByteConfig] = None
    ssm_config: Optional[Dict[str, Any]] = None

class Capibara2(nn.Module, ISubModel):
    """implementation mejorada de Capibara2 with soporte for diferentes backends SSM.
    
    Características:
    - Soporte for múltiples implementaciones SSM
    - setup unificada
    - Métricas detalladas
    - validation robusta
    """
    config: Capibara2Config
    
    def setup(self):
        """Inicializa el backend SSM seleccionado."""
        if self.config.ssm_type == "byte":
            if self.config.byte_config is None:
                raise ValueError("Se requiere byte_config para ssm_type='byte'")
            self.ssm = CapibaraByte(self.config.byte_config)
        else:
            if self.config.ssm_config is None:
                raise ValueError("Se requiere ssm_config para ssm_type='optimized_ssm'")
            self.ssm = CapibaraJAXSSM(self.config.ssm_config)
            
    def __call__(
        self,
        x: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        training: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Forward pass with validation and métricas.
        
        Args:
            x: tensor de input
            context: tensor de contexto optional
            training: Modo entrenamiento
            
        Returns:
            Dict with output and métricas
        """
        # validate input
        if not isinstance(x, jnp.ndarray):
            raise TypeError("x debe ser un jnp.ndarray")
            
        # process with SSM seleccionado
        result = self.ssm(x, context, training)
        
        # combine métricas
        metrics = {
            "ssm_type": self.config.ssm_type,
            **result["metrics"]
        }
        
        return {
            "output": result["output"],
            "metrics": metrics
        }

@jax.jit
def loss_fn(params, model, x, y):
    """
    Computes the loss function.

    Args:
        params: Current model parameters.
        model: The Capibara2 model instance.
        x: Input data for the loss computation.
        and: Target labels for the loss computation.

    Returns:
        A scalar value representing the loss.
    """
    y_pred = model.apply({'params': params}, x)
    return jnp.mean((y_pred - y) ** 2)

@partial(jax.jit, static_argnums=(1,))
def update(params, model, opt_state, x, y):
    """
    Performs a parameter update step.

    Args:
        params: Current model parameters.
        model: The Capibara2 model instance.
        opt_state: Current optimizer state.
        x: Input data for the update step.
        and: Target labels for the update step.

    Returns:
        A tuple containing the updated parameters and optimizer state.
    """
    grads = jax.grad(loss_fn)(params, model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Example usage
if __name__ == "__main__":
    try:
        logger.info("Starting Capibara2 example")
        
        # setup
        ssm_config = SSMConfig(hidden_size=128)
        config = Capibara2Config(
            hidden_size=128,
            dropout_rate=0.1,
            activation="gelu",
            ssm_config=ssm_config
        )

        # Test with diferentes batch sizes
        key = jax.random.PRNGKey(0)
        seq_len = 10
        
        for batch_size in [1, 32, 64]:
            logger.info(f"Testing with batch_size={batch_size}")
            
            # create data de example
            x = jax.random.normal(
                key,
                (batch_size, seq_len, config.hidden_size)
            )

            # Inicializar and execute model
            model = Capibara2(config=config)
            params = model.init(key, x)
            outputs, final_state = model.apply(params, x)
            
            logger.info(
                f"Test successful - Shapes: outputs={outputs.shape}, "
                f"state={final_state.shape}"
            )

        # Test with input invalid
        try:
            invalid_x = jnp.zeros((32, 10, 64))
            outputs, _ = model.apply(params, invalid_x)
        except ValueError as ve:
            logger.info(f"Caught expected ValueError: {ve}")

        logger.info("Capibara2 example completed successfully")

    except Exception as e:
        logger.error(f"Error in Capibara2 example: {str(e)}")
        raise
