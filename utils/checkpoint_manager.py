"""
Sistema de checkpointing optimizado para TPU v4-32.
Implementación standalone sin dependencias externas.
"""

import os 
import sys
import json
import time
import zlib
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

from capibara.jax import jax
from capibara.jax import numpy as jnp
from capibara.core.config import CheckpointConfig
from capibara.jax.experimental.array_serialization import pytree_serialization

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manager for checkpoints optimizado para TPU."""
    
    def __init__(self, config: CheckpointConfig):
        """Initializes the manager de checkpoints."""
        self.config = config
        self.base_dir = Path(config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal state
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._pending_saves = []
        self._checkpoint_meta = self._load_checkpoint_meta()
        
    def _load_checkpoint_meta(self) -> Dict[str, Any]:
        """Loads o crea los metadatos de checkpoints."""
        meta_path = self.base_dir / "checkpoint_meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        return {
            "checkpoints": [],
            "latest": None,
            "best": None
        }
        
    def _save_checkpoint_meta(self):
        """Saves los metadatos de checkpoints."""
        meta_path = self.base_dir / "checkpoint_meta.json"
        with open(meta_path, "w") as f:
            json.dump(self._checkpoint_meta, f, indent=2)
            
    def _compress_array(self, array: jnp.ndarray) -> bytes:
        """Comprime un array usando zlib."""
        if not self.config.use_compression:
            return array.tobytes()
        return zlib.compress(array.tobytes(), level=self.config.compression_level)
        
    def _decompress_array(self, data: bytes, shape: Tuple[int, ...], dtype: Any) -> jnp.ndarray:
        """Descomprime un array."""
        if not self.config.use_compression:
            return jnp.frombuffer(data, dtype=dtype).reshape(shape)
        return jnp.frombuffer(zlib.decompress(data), dtype=dtype).reshape(shape)
        
    def _shard_array(self, array: jnp.ndarray) -> List[bytes]:
        """Divide un array en shards."""
        if not self.config.use_sharded_checkpoints:
            return [self._compress_array(array)]
            
        shards = []
        bytes_per_element = array.dtype.itemsize
        elements_per_shard = self.config.shard_size_bytes // bytes_per_element
        
        for i in range(0, array.size, elements_per_shard):
            shard = array.ravel()[i:i + elements_per_shard]
            shards.append(self._compress_array(shard))
            
        return shards
        
    def _merge_shards(self, shards: List[bytes], shape: Tuple[int, ...], dtype: Any) -> jnp.ndarray:
        """Combina shards en un array."""
        if len(shards) == 1:
            return self._decompress_array(shards[0], shape, dtype)
            
        arrays = []
        for shard in shards:
            arrays.append(self._decompress_array(shard, (-1,), dtype))
            
        return jnp.concatenate(arrays).reshape(shape)
        
    def save(
        self,
        state: Any,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> str:
        """Saves un checkpoint."""
        checkpoint_id = f"checkpoint_{step:08d}"
        checkpoint_dir = self.base_dir / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Serialize state
        state_dict = pytree_serialization.to_state_dict(state)
        
        # Save each array in shards
        array_meta = {}
        futures = []
        
        for path, array in state_dict.items():
            array_dir = checkpoint_dir / path
            array_dir.mkdir(parents=True, exist_ok=True)
            
            shards = self._shard_array(array)
            array_meta[path] = {
                "shape": array.shape,
                "dtype": str(array.dtype),
                "num_shards": len(shards)
            }
            
            for i, shard in enumerate(shards):
                shard_path = array_dir / f"shard_{i:04d}.bin"
                if self.config.use_async_checkpointing:
                    futures.append(self._executor.submit(self._save_shard, shard_path, shard))
                else:
                    self._save_shard(shard_path, shard)
                    
        # Wait for completion of todas las escrituras
        for future in futures:
            future.result()
            
        # Save metadata
        meta = {
            "step": step,
            "timestamp": time.time(),
            "arrays": array_meta,
            "metrics": metrics or {}
        }
        
        with open(checkpoint_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        # Actualizar metadatos globales
        with self._lock:
            self._checkpoint_meta["checkpoints"].append(checkpoint_id)
            self._checkpoint_meta["latest"] = checkpoint_id
            if is_best:
                self._checkpoint_meta["best"] = checkpoint_id
                
            # Keep only the last N checkpoints
            while len(self._checkpoint_meta["checkpoints"]) > self.config.max_to_keep:
                to_remove = self._checkpoint_meta["checkpoints"].pop(0)
                if to_remove != self._checkpoint_meta["best"]:
                    self._remove_checkpoint(to_remove)
                    
            self._save_checkpoint_meta()
            
        return checkpoint_id
        
    def load(self, checkpoint_id: Optional[str] = None) -> Any:
        """Loads un checkpoint."""
        if checkpoint_id is None:
            checkpoint_id = self._checkpoint_meta["latest"]
            
        if checkpoint_id is None:
            raise ValueError("No hay checkpoints disponibles")
            
        checkpoint_dir = self.base_dir / checkpoint_id
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint {checkpoint_id} no encontrado")
            
        # Cargar metadatos
        with open(checkpoint_dir / "meta.json", "r") as f:
            meta = json.load(f)
            
        # Reconstruct state
        state_dict = {}
        futures = []
        
        for path, array_meta in meta["arrays"].items():
            array_dir = checkpoint_dir / path
            shape = tuple(array_meta["shape"])
            dtype = jnp.dtype(array_meta["dtype"])
            num_shards = array_meta["num_shards"]
            
            shards = []
            for i in range(num_shards):
                shard_path = array_dir / f"shard_{i:04d}.bin"
                if self.config.use_async_checkpointing:
                    futures.append((path, shape, dtype, self._executor.submit(self._load_shard, shard_path)))
                else:
                    shards.append(self._load_shard(shard_path))
                    
            if not self.config.use_async_checkpointing:
                state_dict[path] = self._merge_shards(shards, shape, dtype)
                
        # Wait for completion of todas las lecturas
        if self.config.use_async_checkpointing:
            for path, shape, dtype, future in futures:
                if path not in state_dict:
                    state_dict[path] = []
                state_dict[path].append(future.result())
                
            # Combinar shards
            for path, array_meta in meta["arrays"].items():
                shape = tuple(array_meta["shape"])
                dtype = jnp.dtype(array_meta["dtype"])
                state_dict[path] = self._merge_shards(state_dict[path], shape, dtype)
                
        # Reconstruct and validate state
        state = pytree_serialization.from_state_dict(state_dict)
        if self.config.validate_on_load:
            self._validate_state(state)
            
        return state
        
    def _save_shard(self, path: Path, data: bytes):
        """Saves un shard on disk."""
        with open(path, "wb") as f:
            f.write(data)
            
    def _load_shard(self, path: Path) -> bytes:
        """Loads un shard desde disco."""
        with open(path, "rb") as f:
            return f.read()
            
    def _remove_checkpoint(self, checkpoint_id: str):
        """Removes un checkpoint."""
        checkpoint_dir = self.base_dir / checkpoint_id
        if checkpoint_dir.exists():
            for path in checkpoint_dir.rglob("*"):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            checkpoint_dir.rmdir()
            
    def _validate_state(self, state: Any):
        """Validates state integrity."""
        return self._validate_checkpoint_integrity(state)
    
    def _validate_checkpoint_integrity(self, state: Any) -> bool:
        """
        Valida la integridad de un checkpoint.
        
        Args:
            state: Estado del modelo a validar
            
        Returns:
            True si el checkpoint es válido
        """
        try:
            # Validate that state is not empty
            if state is None:
                logger.error("Estado del checkpoint es None")
                return False
            
            # Validate state structure
            if hasattr(state, 'params'):
                # Verify that params no esté vacío
                if not state.params:
                    logger.error("Parámetros del modelo están vacíos")
                    return False
                
                # Verify that there are no NaN or Inf in parameters
                def check_arrays(pytree):
                    def check_array(arr):
                        if jnp.any(jnp.isnan(arr)) or jnp.any(jnp.isinf(arr)):
                            return False
                        return True
                    
                    from jax import tree_map
                    results = tree_map(check_array, pytree)
                    return all(jax.tree_util.tree_leaves(results))
                
                if not check_arrays(state.params):
                    logger.error("Parámetros contienen valores NaN o Inf")
                    return False
            
            # Validate that state has expected structure
            required_attrs = ['params']
            for attr in required_attrs:
                if not hasattr(state, attr):
                    logger.warning(f"Estado no tiene atributo requerido: {attr}")
            
            logger.info("✅ Validación de checkpoint exitosa")
            return True
            
        except Exception as e:
            logger.error(f"Error validando checkpoint: {e}")
            return False 
        except Exception as e:
            logger.error(f"Error al validar checkpoint: {e}")
            return False

# Compatibility alias
CapibaraCheckpointManager = CheckpointManager
