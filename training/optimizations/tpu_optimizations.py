"""
Optimizaciones TPU v4-32 centralizadas para CapibaraGPT

Este module concentra optimizaciones específicas para TPU v4-32.
"""

import os
import logging
from pathlib import Path

from functools import partial
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Callable

import numpy as np
try:
    import jax  # type: ignore
    import toml  # type: ignore
    import optax  # type: ignore
    from flax.training import train_state  # type: ignore
    from capibara.jax.sharding import PartitionSpec as P  # type: ignore
    from capibara.jax.experimental import mesh_utils, shard_map  # type: ignore
except ImportError:
    # Fallbacks si faltan dependencias
    train_state = None
    P = None
    mesh_utils = None
    shard_map = None
    jax = None
    optax = None
    toml = None

logger = logging.getLogger(__name__)

# Global TPU v4-32
if jax:
    try:
        jax.config.update("jax_threefry_partitionable", True)  # type: ignore[attr-defined]
    except Exception:
        pass

@dataclass
class TPUConfig:
    """Configuration para TPU v4-32."""

    def __init__(self, config_path: Optional[str] = None):
        # Cargar desde TOML si está disponible
        config_root = Path(os.environ.get("CAPIBARA_CONFIG_ROOT", "capibara/config"))
        config_path = config_path or str(config_root / "configs_toml/production/hardware.toml")

        try:
            if toml and os.path.exists(config_path):
                config = toml.load(config_path)  # type: ignore[call-arg]
            else:
                config = {}

            # Hardware
            tpu_config = config.get("tpu_v4", {})
            self.cores = tpu_config.get("cores", 32)
            self.mesh_shape = tuple(tpu_config.get("mesh_shape", [4, 8]))
            self.memory_per_core_gb = tpu_config.get("memory_per_core_gb", 32)
            self.hbm_bandwidth_gbps = tpu_config.get("hbm_bandwidth_gbps", 1200)

            # Memoria
            self.memory_pressure_threshold = tpu_config.get("memory_pressure_threshold", 0.85)
            self.cleanup_threshold = tpu_config.get("cleanup_threshold", 0.9)
            self.enable_memory_monitoring = tpu_config.get("enable_memory_monitoring", True)
            self.enable_systolic_arrays = tpu_config.get("enable_systolic_arrays", True)

            # Optimización
            opt_config = config.get("optimization", {})
            self.enable_xla = opt_config.get("enable_xla", True)
            self.enable_auto_sharding = opt_config.get("enable_auto_sharding", True)
            self.optimization_level = opt_config.get("optimization_level", 3)
            self.enable_gradient_checkpointing = opt_config.get("enable_gradient_checkpointing", True)
            self.enable_remat = opt_config.get("enable_remat", True)
            self.batch_size_per_core = opt_config.get("batch_size_per_core", 16)
            self.learning_rate = opt_config.get("learning_rate", 1e-4)

        except Exception as e:
            logger.error(f"Error cargando configuración TPU: {e}")
            # Defaults seguros
            self.cores = 32
            self.mesh_shape = (4, 8)
            self.memory_per_core_gb = 32
            self.hbm_bandwidth_gbps = 1200
            self.memory_pressure_threshold = 0.85
            self.cleanup_threshold = 0.9
            self.enable_memory_monitoring = True
            self.enable_systolic_arrays = True
            self.enable_xla = True
            self.enable_auto_sharding = True
            self.optimization_level = 3
            self.enable_gradient_checkpointing = True
            self.enable_remat = True
            self.batch_size_per_core = 16
            self.learning_rate = 1e-4

class TPUOptimizer:
    """Optimizador centralizado para TPU v4-32."""

    def __init__(self, config: Optional[TPUConfig] = None):
        self.config = config or TPUConfig()
        if jax:
            self._setup_device()

    def _setup_device(self) -> None:
        """Configura el dispositivo TPU si está disponible."""
        try:
            devices = jax.devices("tpu") if jax else []  # type: ignore[attr-defined]
        except Exception:
            devices = []
        if not devices:
            logger.warning("No hay dispositivos TPU o JAX no está disponible")
            return

        # Configuración XLA
        if self.config.enable_xla:
            try:
                jax.config.update("jax_xla_backend", "tpu")  # type: ignore[attr-defined]
            except Exception:
                pass

        # Auto sharding
        if self.config.enable_auto_sharding:
            try:
                jax.config.update("jax_enable_auto_sharding", True)  # type: ignore[attr-defined]
            except Exception:
                pass

        # Nivel de optimización
        try:
            jax.config.update("jax_optimization_level", self.config.optimization_level)  # type: ignore[attr-defined]
        except Exception:
            pass

    def create_mesh(self) -> Tuple[Any, Any]:  # type: ignore[name-defined]
        """Crea la malla de dispositivos TPU."""
        if not mesh_utils or not jax:  # type: ignore[truthy-bool]
            return None, None
        devices = mesh_utils.create_device_mesh(self.config.mesh_shape)  # type: ignore[attr-defined]
        mesh = jax.sharding.Mesh(devices, ["data", "model"])  # type: ignore[attr-defined]
        return devices, mesh

    def get_sharding_rules(self) -> Dict[str, Any]:
        """Reglas de sharding predefinidas."""
        if not P:
            return {}
        return {
            "params": P("model", None),
            "batch_stats": P("model"),
            "aux": P("model"),
            "dropout": P(None),
            "intermediates": P(None),
        }

    def get_training_state(self, model_params: Dict[str, Any]):
        """Crea un estado de entrenamiento optimizado."""
        if not train_state or not optax:
            return None

        learning_rate = getattr(self.config, "learning_rate", 1e-4)
        tx = optax.adamw(
            learning_rate=learning_rate,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.01,
        )

        return train_state.TrainState.create(
            apply_fn=None,  # Se asigna después
            params=model_params.get("params", {}),
            tx=tx,
        )

    def get_batch_size(self) -> int:
        """Calculates el batch size óptimo."""
        return self.config.batch_size_per_core * self.config.cores

    def should_cleanup_memory(self) -> bool:
        """Verifies si se debe limpiar memoria."""
        if not self.config.enable_memory_monitoring or not jax:
            return False
        try:
            current_usage = jax.device_get(jax.device_memory_usage())  # type: ignore[attr-defined]
            return current_usage > self.config.memory_pressure_threshold
        except Exception:
            return False

    def force_cleanup(self) -> None:
        """Fuerza limpieza de memoria."""
        try:
            if jax and self.should_cleanup_memory():
                jax.clear_caches()  # type: ignore[attr-defined]
                logger.info("Forced TPU memory cleanup")
        except Exception:
            pass

def setup_tpu_environment():
    """Configura entorno TPU v4-32 completo."""
    config = TPUConfig()
    optimizer = TPUOptimizer(config)
    return optimizer

def verify_tpu_setup() -> Dict[str, Any]:
    """Verifies entorno TPU y devuelve información básica."""
    if not jax:
        return {"tpu_available": False, "error": "JAX no disponible"}

    try:
        devices = jax.devices("tpu")
        return {
            "tpu_available": len(devices) > 0,
            "device_count": len(devices),
            "devices": [str(d) for d in devices],
            "platform": getattr(getattr(jax.lib, "xla_bridge", None), "get_backend", lambda: type("B", (), {"platform": "unknown"})())().platform,
        }
    except Exception as e:
        return {"tpu_available": False, "error": str(e)}