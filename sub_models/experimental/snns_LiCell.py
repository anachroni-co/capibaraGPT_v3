"""
Spiking Neural Network (LIF-based) implementation for CapibaraModel.
Enhanced version with improved configuration, validation, and optimization.
"""

import os
# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../CapibaraGPT v3
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    pass  # Using proper imports instead of sys.path manipulation

import jax
from jax import numpy as jnp  # type: ignore
from flax import linen as nn  # type: ignore
import logging
from typing import NamedTuple, Tuple, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ValidationInfo #type: ignore
from functools import partial #type: ignore
from interfaces.isub_models import ISubModel

logger = logging.getLogger(__name__)

class LIFConfig(BaseModel):
    """Enhanced configuration for LIF neurons."""
    hidden_size: int = Field(..., gt=0, description="Dimensión del espacio oculto")
    tau_m: float = Field(default=20.0, gt=0, description="Constante de tiempo de membrana")
    v_rest: float = Field(default=-65.0, description="Potencial de reposo")
    v_reset: float = Field(default=-70.0, description="Potencial de reset")
    v_threshold: float = Field(default=-50.0, description="Umbral base")
    dropout_rate: float = Field(default=0.1, ge=0, lt=1, description="Tasa de dropout")
    max_seq_len: int = Field(default=512, gt=0, description="Longitud máxima de secuencia")
    
    @field_validator('tau_m', 'v_threshold')
    def validate_parameters(cls, v, info: ValidationInfo):
        """Valida parámetros neuronales."""
        if info.field_name == 'tau_m' and v <= 0:
            raise ValueError("tau_m debe ser positivo")
        if info.field_name == 'v_threshold' and v >= 0:
            raise ValueError("v_threshold debe ser negativo")
        return v

class LIFState(NamedTuple):
    """Enhanced state for LIF neurons."""
    voltage: jnp.ndarray
    spikes: jnp.ndarray
    threshold: jnp.ndarray
    
    @classmethod
    def initialize(cls, batch_size: int, hidden_size: int, dtype: Any = jnp.float32):
        """Inicializa estado with valores by defect."""
        return cls(
            voltage=jnp.zeros((batch_size, hidden_size), dtype=dtype),
            spikes=jnp.zeros((batch_size, hidden_size), dtype=dtype),
            threshold=jnp.ones((batch_size, hidden_size), dtype=dtype) * -50.0
        )

class SNNSLiCell(nn.Module, ISubModel):
    """Enhanced spiking neural network cell."""
    config: LIFConfig

    def setup(self):
        """Initialize components with error handling."""
        try:
            self.input_proj = nn.Dense(self.config.hidden_size)
            self.recurrent = nn.Dense(self.config.hidden_size)
            self.output = nn.Dense(self.config.hidden_size)
            self.norm = nn.LayerNorm()
            self.dropout = nn.Dropout(rate=self.config.dropout_rate)
            logger.info("Model components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    def _validate_input(self, x: jnp.ndarray) -> None:
        """Enhanced input validation."""
        try:
            if not isinstance(x, jnp.ndarray):
                raise TypeError(f"Expected jnp.ndarray, got {type(x)}")
            
            if x.ndim != 3:
                raise ValueError(
                    f"Expected 3D input (batch, seq_len, dim), got shape {x.shape}"
                )
            
            if x.shape[1] > self.config.max_seq_len:
                raise ValueError(
                    f"Sequence length {x.shape[1]} exceeds max_seq_len "
                    f"{self.config.max_seq_len}"
                )
            
            if jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)):
                raise ValueError("Input contains NaN or Inf values")
            
            logger.debug(f"Input validation passed. Shape: {x.shape}")
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise

    @partial(jax.jit, static_argnums=(0,))
    def _lif_step(
        self,
        x: jnp.ndarray,
        state: LIFState,
        training: bool = False
    ) -> Tuple[jnp.ndarray, LIFState]:
        """Enhanced LIF step with JIT support."""
        try:
            # Project input
            current = self.input_proj(x)
            if training:
                current = self.dropout(current, deterministic=not training)

            # Add recurrent dynamics
            rec = self.recurrent(state.voltage)
            current += rec

            # Update voltage with improved numerical stability
            dv = jnp.clip(
                (-(state.voltage - self.config.v_rest) + current) / self.config.tau_m,
                -100, 100
            )
            new_v = state.voltage + dv

            # Generate spikes with threshold adaptation
            spikes = (new_v >= state.threshold).astype(jnp.float32)
            new_v = jnp.where(spikes > 0, self.config.v_reset, new_v)

            # Update threshold with stability improvements
            new_threshold = (
                state.threshold
                + 0.1 * spikes
                - (state.threshold - self.config.v_threshold) / self.config.tau_m
            )
            new_threshold = jnp.clip(new_threshold, -100, 0)

            new_state = LIFState(
                voltage=new_v,
                spikes=spikes,
                threshold=new_threshold
            )
            return spikes, new_state

        except Exception as e:
            logger.error(f"Error in LIF step: {e}")
            raise RuntimeError("LIF step failed") from e

    def __call__(
        self,
        x: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        training: bool = False,
        **kwargs: Any
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, LIFState]]:
        """Enhanced forward pass with parallel processing support."""
        try:
            logger.debug(f"Starting forward pass. Input shape: {x.shape}")
            self._validate_input(x)

            batch_size, seq_len, dim = x.shape
            
            # Initialize or validate state
            initial_state = (
                LIFState.initialize(batch_size, self.config.hidden_size)
                if context is None
                else context
            )

            # Parallel processing with vmap if multiple devices
            if jax.device_count() > 1:
                logger.debug("Using parallel processing")
                scan_fn = jax.vmap(self._lif_step, in_axes=(1, None))
                x_t = jnp.moveaxis(x, 1, 0)
                final_state, all_spikes = scan_fn(x_t, initial_state)
            else:
                # Sequential processing
                def scan_fn(carry, x_t):
                    state = carry
                    spikes, new_state = self._lif_step(x_t, state, training)
                    return new_state, spikes

                x_t = jnp.swapaxes(x, 0, 1)
                final_state, all_spikes = jax.lax.scan(scan_fn, initial_state, x_t)

            # Post-process spikes
            all_spikes = jnp.swapaxes(all_spikes, 0, 1)
            outputs = self.output(all_spikes)
            
            logger.debug(f"Forward pass completed. Output shape: {outputs.shape}")
            
            if kwargs.get('return_state', False):
                return outputs, final_state
            return outputs

        except ValueError as ve:
            logger.error(f"Input validation error: {ve}")
            raise
        except TypeError as te:
            logger.error(f"Type error: {te}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in forward pass: {e}")
            raise RuntimeError("Forward pass failed") from e
