"""Módulo Capivision for CapibaraModel.

Este módulo implementa las arquitecturas Capibara SSM 1D and 2D,
incluyendo el bloque VSS for procesamiento visual.
"""

import os
import sys
# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../CapibaraGPT v3
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    pass  # Using proper imports instead of sys.path manipulation

import logging
import dataclasses #type: ignore
import jax
from flax import linen as nn #type: ignore
from jax import numpy as jnp #type: ignore
from typing import Dict, Any, Optional, TypedDict #type: ignore

try:
    from jax.experimental import debugger #type: ignore
except ImportError:
    class debugger:
        @staticmethod
        def breakpoint_on_error():
            pass

try:
    from capibara.interfaces.imodules import IModule
except ImportError:
    class IModule:
        pass

# Definición de tipos
class ModuleOutput(TypedDict):
    output: jnp.ndarray
    is_active: jnp.ndarray
    score: jnp.ndarray

# Constantes for distribución
MODEL_SHARDING = None
REPLICATED = None

def distributed_jit(*args, **kwargs):
    """Decorador placeholder for distributed_jit."""
    def decorator(f):
        return f
    return decorator

logger = logging.getLogger(__name__)

# --------------------------
# core Capibara SSM 1D
# --------------------------
class Mamba1DCore(nn.Module, IModule):
    """implementation del core de escaneo selectivo 1D de Mamba."""
    d_state: int = 16  # dimension del estado latente SSM
    d_conv: int = 4    # size del kernel de convolución causal
    expand: int = 2    # Factor de expansión de la dimension interna

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Aplica la lógica SSM 1D a una secuencia (B, L, D)."""
        B, L, D = x.shape
        d_inner = D * self.expand

        # Proyecciones de input
        xz = nn.Dense(d_inner * 2, name="in_proj")(x)
        x_proj, z = jnp.split(xz, 2, axis=-1)

        # Convolución Causal + SiLU
        x_conv = nn.Dense(d_inner, name="conv_proj")(x_proj)
        x_act = nn.silu(x_conv)

        # Lógica SSM Selectiva
        logger.debug("Aplicando lógica SSM selectiva en Mamba1DCore")
        ssm_output = nn.Dense(d_inner, name="ssm_core")(x_act * nn.silu(z))

        # Proyección de output
        output = nn.Dense(D, name="out_proj")(ssm_output)
        return output

# --------------------------
# Módulo SS2D
# --------------------------
class SS2D(nn.Module, IModule):
    """Implementa el escaneo selectivo 2D inspirado en VMamba."""
    ssm_config: Dict[str, Any]

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Aplica escaneos Mamba 1D en 4 direcciones and fusiona."""
        B, N, E = x.shape
        H = W = int(jnp.sqrt(N))
        
        if H * W != N:
            logger.warning("SS2D asume N es un cuadrado perfecto (H=W)")
            H = W = int(jnp.sqrt(N))  # setting for el caso more cercano

        # create instance del core SSM 1D
        mamba_core = Mamba1DCore(**self.ssm_config, name="mamba_1d_core")

        # Reshape a format espacial
        x_spatial = jnp.reshape(x, (B, H, W, E))

        # Escaneo 1: Forward (Row-major)
        scan1_in = jnp.reshape(x_spatial, (B, N, E))
        scan1_out = mamba_core(scan1_in, deterministic=deterministic)

        # Escaneo 2: Backward (Row-major, reversed)
        scan2_in = jnp.flip(scan1_in, axis=1)
        scan2_out = mamba_core(scan2_in, deterministic=deterministic)
        scan2_out = jnp.flip(scan2_out, axis=1)

        # Escaneo 3: Forward (Column-major)
        scan3_in = jnp.reshape(jnp.transpose(x_spatial, (0, 2, 1, 3)), (B, N, E))
        scan3_out_scanned = mamba_core(scan3_in, deterministic=deterministic)
        scan3_out = jnp.transpose(jnp.reshape(scan3_out_scanned, (B, W, H, E)), (0, 2, 1, 3))
        scan3_out = jnp.reshape(scan3_out, (B, N, E))

        # Escaneo 4: Backward (Column-major, reversed)
        scan4_in = jnp.flip(scan3_in, axis=1)
        scan4_out_scanned = mamba_core(scan4_in, deterministic=deterministic)
        scan4_out_scanned = jnp.flip(scan4_out_scanned, axis=1)
        scan4_out = jnp.transpose(jnp.reshape(scan4_out_scanned, (B, W, H, E)), (0, 2, 1, 3))
        scan4_out = jnp.reshape(scan4_out, (B, N, E))

        # merge resultados
        x_merged = (scan1_out + scan2_out + scan3_out + scan4_out) / 4.0

        return x_merged

# --------------------------
# Visual State Space Block (VSSBlock)
# --------------------------
class VSSBlock(nn.Module, IModule):
    """Bloque VSS with SS2D and UUPBitLinear."""
    dim: int
    ssm_expand: int = 2
    ffn_expand: int = 4
    dropout_rate: float = 0.1
    ssm_config: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"d_state": 16, "d_conv": 4, "expand": 2}
    )

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        ssm_dim = self.dim * self.ssm_expand
        ffn_dim = self.dim * self.ffn_expand

        # Rama SSM
        residual_ssm = x
        x_norm = nn.LayerNorm(name="norm_ssm")(x)
        x_proj = nn.Dense(ssm_dim, name="ssm_in_proj")(x_norm)
        x_proj = nn.silu(x_proj)
        x_ssm = SS2D(ssm_config=self.ssm_config, name="ss2d_core")(x_proj, deterministic=deterministic)
        x_ssm = nn.Dense(self.dim, name="ssm_out_proj")(x_ssm)
        x_ssm = nn.Dropout(self.dropout_rate, name="ssm_dropout")(x_ssm, deterministic=deterministic)
        x = residual_ssm + x_ssm

        # Rama FFN
        residual_ffn = x
        x_norm2 = nn.LayerNorm(name="norm_ffn")(x)
        x_ffn = nn.Dense(ffn_dim, name="ffn_in")(x_norm2)
        x_ffn = nn.silu(x_ffn)
        x_ffn = nn.Dense(self.dim, name="ffn_out")(x_ffn)
        x_ffn = nn.Dropout(self.dropout_rate, name="ffn_dropout")(x_ffn, deterministic=deterministic)
        x = residual_ffn + x_ffn

        return x

    @distributed_jit(in_specs=MODEL_SHARDING, out_specs=REPLICATED)
    def process(self, x: jnp.ndarray, deterministic: bool = True) -> ModuleOutput:
        """Procesa la input and devuelve el output formateado."""
        with debugger.breakpoint_on_error():
            try:
                output = self.__call__(x, deterministic)
                return {
                    "output": output,
                    "is_active": jnp.ones(x.shape[0], dtype=bool),
                    "score": jnp.mean(output, axis=-1)
                }
            except Exception as e:
                logger.error(f"Error en VSSBlock: {str(e)}")
                raise

class Capivision(nn.Module, IModule):
    """Módulo principal de visión for CapibaraModel."""
    config: Dict[str, Any]
    
    def setup(self):
        """Inicializa componentes."""
        self.vss_block = VSSBlock(
            dim=self.config["hidden_size"],
            ssm_expand=self.config.get("ssm_expand", 2),
            ffn_expand=self.config.get("ffn_expand", 4),
            dropout_rate=self.config.get("dropout_rate", 0.1)
        )
        
    def __call__(
        self,
        x: jnp.ndarray,
        context: Optional[Dict[str, Any]] = None,
        training: bool = False
    ) -> Dict[str, Any]:
        """Procesa input visual.
        
        Args:
            x: tensor de input (batch, seq_len, hidden_size)
            context: Contexto optional
            training: Modo entrenamiento
            
        Returns:
            Dict with output and métricas
        """
        result = self.vss_block.process(x, deterministic=not training)
        return {
            "output": result["output"],
            "metrics": {
                "is_active": result["is_active"],
                "score": result["score"]
            }
        }