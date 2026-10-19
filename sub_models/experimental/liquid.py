"""
Liquid layer implementation for CapibaraModel.

This layer expands the input dimension by an expansion factor, processes it,
then contracts back to the original dimension. Includes a residual connection
and optional dropout.

Example:
    >>> config = LiquidConfig(
    ...     hidden_size=256,
    ...     expansion_factor=4,
    ...     dropout_rate=0.1,
    ...     activation="gelu"
    ... )
    >>> layer = Liquid(config)
    >>> x = jnp.ones((32, 10, 256))
    >>> output = layer(x, training=True)
"""

import os

# Esto añade la folder raíz del proyecto a la path de búsqueda de Python
# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../CapibaraGPT v3
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    pass  # Using proper imports instead of sys.path manipulation

from typing import Dict, Any, Optional
import jax
from flax import linen as nn  # type: ignore
from pydantic import BaseModel, Field # type: ignore
from jax import numpy as jnp  # type: ignore

from capibara.interfaces.isub_models import ISubModel

logger = logging.getLogger(__name__)

class LiquidConfig(BaseModel):
    """Configuración para capa Liquid."""
    hidden_size: int
    expansion_factor: int = Field(default=4, gt=0)
    use_residual: bool = True
    activation: str = Field(default="gelu")
    dropout_rate: float = Field(default=0.1, ge=0, le=1)
    use_norm: bool = True

class Liquid(nn.Module, IExperimentalModel):
    """Capa with expansión/contracción dinámica."""
    
    config: LiquidConfig
    
    def setup(self):
        """Inicializa componentes."""
        self.expand = nn.Dense(
            self.config.hidden_size * self.config.expansion_factor
        )
        self.contract = nn.Dense(self.config.hidden_size)
        self.norm = nn.LayerNorm()
        
    def __call__(
        self,
        x: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        training: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Forward pass."""
        self.validate_input(x)
        
        # Expansión
        x_exp = self.expand(x)
        x_exp = getattr(nn, self.config.activation)(x_exp)  # activation dinámica
        x_exp = nn.Dropout(rate=self.config.dropout_rate)(x_exp, deterministic=not training)
        
        # Contracción
        x_out = self.contract(x_exp)
        
        # Residual
        if self.config.use_residual:
            x_out = x_out + kwargs.get("residual", x_out)
            
        # Normalización (optional)
        if hasattr(self.config, "use_norm") and self.config.use_norm:
            x_out = self.norm(x_out)
        
        return {
            "output": x_out,
            "metrics": {
                "input_norm": jnp.linalg.norm(x),
                "output_norm": jnp.linalg.norm(x_out),
                "expansion_ratio": self.config.expansion_factor,
                "dropout_active": float(training),  # Métrica adicional
            }
        }
        
    def get_config(self) -> Dict[str, Any]:
        return self.config.model_dump()
        
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "expansion_factor": self.config.expansion_factor,
            "use_residual": self.config.use_residual,
            "dropout_rate": self.config.dropout_rate,
            "use_norm": self.config.use_norm
        }
        
    def validate_input(self, x: jnp.ndarray) -> None:
        if x.ndim != 3:
            raise ValueError("Input must be 3D tensor")

if __name__ == "__main__":
    try:
        logger.info("Starting Liquid example")
        
        # setup
        config = LiquidConfig(
            hidden_size=128,
            expansion_factor=4,
            dropout_rate=0.1,
            activation="gelu"
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
            layer = Liquid(config=config)
            params = layer.init(key, x)
            output = layer.apply(params, x)
            
            logger.info(f"Test successful - Output shape: {output['output'].shape}")

        # Test with input invalid
        try:
            invalid_x = jnp.zeros((32, 10, 64))
            output = layer.apply(params, invalid_x)
        except ValueError as ve:
            logger.info(f"Caught expected ValueError: {ve}")

        logger.info("Liquid example completed successfully")

    except Exception as e:
        logger.error(f"Error in Liquid example: {str(e)}")
        raise
