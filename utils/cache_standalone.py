"""
Sistema de cache optimizado para TPU v4-32.
Implementación standalone sin dependencias externas.
"""

import json
import time
import logging
import threading
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic, Callable, Protocol

from capibara.jax import jax
from capibara.jax import numpy as jnp
from capibara.core.config import CheckpointConfig

# TypeVar for Generic
T = TypeVar('T')

logger = logging.getLogger(__name__)

class CacheManager:
    """Manager for cache optimizado para TPU."""
    
    def __init__(self, config: Any):
        """Initializes the manager de cache."""
        from capibara.core.config import CheckpointConfig
        from capibara.utils.checkpoint_manager import CheckpointManager
        
        if not isinstance(config, CheckpointConfig):
            raise TypeError("config debe ser una instancia de CheckpointConfig")
            
        self.config = config
        self.checkpoint_manager = CheckpointManager(config)
        self.base_dir = Path(config.base_dir) / "cache"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal state
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._cache = {}
        self._cache_meta = self._load_cache_meta()
        
    def _load_cache_meta(self) -> Dict[str, Any]:
        """Loads o crea los metadatos de cache."""
        meta_path = self.base_dir / "cache_meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        return {
            "entries": {},
            "size": 0,
            "hits": 0,
            "misses": 0
        }
        
    def _save_cache_meta(self):
        """Saves los metadatos de cache."""
        meta_path = self.base_dir / "cache_meta.json"
        with open(meta_path, "w") as f:
            json.dump(self._cache_meta, f, indent=2)
            
    def get(self, key: str) -> Optional[Any]:
        """Gets un valor del cache."""
        with self._lock:
            if key in self._cache:
                self._cache_meta["hits"] += 1
                self._save_cache_meta()
                return self._cache[key]
                
            if key in self._cache_meta["entries"]:
                entry = self._cache_meta["entries"][key]
                checkpoint_id = entry["checkpoint_id"]
                try:
                    value = self.checkpoint_manager.load(checkpoint_id)
                    self._cache[key] = value
                    self._cache_meta["hits"] += 1
                    self._save_cache_meta()
                    return value
                except Exception as e:
                    logger.warning(f"Error cargando cache para {key}: {e}")
                    
            self._cache_meta["misses"] += 1
            self._save_cache_meta()
            return None
            
    def set(self, key: str, value: Any):
        """Saves un valor en el cache."""
        with self._lock:
            # Save in checkpoint
            checkpoint_id = self.checkpoint_manager.save(
                value,
                step=len(self._cache_meta["entries"]),
                metrics={"cache_key": key}
            )
            
            # Actualizar metadatos
            self._cache_meta["entries"][key] = {
                "checkpoint_id": checkpoint_id,
                "timestamp": time.time()
            }
            self._cache[key] = value
            self._save_cache_meta()
            
    def clear(self):
        """Limpia el cache."""
        with self._lock:
            self._cache.clear()
            for entry in self._cache_meta["entries"].values():
                checkpoint_id = entry["checkpoint_id"]
                self.checkpoint_manager._remove_checkpoint(checkpoint_id)
            self._cache_meta["entries"].clear()
            self._cache_meta["size"] = 0
            self._save_cache_meta()
            
    def get_stats(self) -> Dict[str, Any]:
        """Gets statistics del cache."""
        with self._lock:
            return {
                "size": len(self._cache_meta["entries"]),
                "hits": self._cache_meta["hits"],
                "misses": self._cache_meta["misses"],
                "hit_ratio": self._cache_meta["hits"] / (self._cache_meta["hits"] + self._cache_meta["misses"]) if (self._cache_meta["hits"] + self._cache_meta["misses"]) > 0 else 0
            }

class CacheError(Exception):
    """Error base para cache."""
    pass

class CacheEntry(Generic[T]):
    """Entrada de cache con TTL."""
    
    def __init__(self, value: T, ttl: Optional[float] = None):
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """Verifiesr si la entrada expiró."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

class StandaloneCache(Generic[T]):
    """Cache en memoria con persistencia opcional."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: Optional[float] = None,
        persist_dir: Optional[Union[str, Path]] = None
    ):
        self.max_size = max_size
        self.ttl = ttl
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.Lock()
        
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted()
            
        logger.info(f"✅ Cache inicializado (max_size={max_size}, ttl={ttl})")
    
    def get(self, key: str) -> Optional[T]:
        """
        Obtener valor del cache.
        
        Args:
            key: Clave a buscar
            
        Returns:
            Valor o None si no existe/expiró
        """
        with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self.cache[key]
                return None
                
            return entry.value
    
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """
        Guardar valor en cache.
        
        Args:
            key: Clave
            value: Valor
            ttl: TTL específico o None para usar el default
        """
        with self._lock:
            # Limpiar entradas expiradas
            self._cleanup()
            
            # Verificar límite
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Save new entry
            self.cache[key] = CacheEntry(
                value,
                ttl=ttl if ttl is not None else self.ttl
            )
            
            # Persistir si está habilitado
            if self.persist_dir:
                self._persist_entry(key)
    
    def delete(self, key: str) -> None:
        """
        Eliminar entrada del cache.
        
        Args:
            key: Clave a eliminar
        """
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                
                if self.persist_dir:
                    persist_file = self.persist_dir / f"{key}.json"
                    try:
                        persist_file.unlink()
                    except Exception as e:
                        logger.warning(f"⚠️ Error al eliminar {key}: {e}")
    
    def clear(self) -> None:
        """Limpiar todo el cache."""
        with self._lock:
            self.cache.clear()
            
            if self.persist_dir:
                try:
                    for file in self.persist_dir.glob("*.json"):
                        file.unlink()
                except Exception as e:
                    logger.warning(f"⚠️ Error al limpiar cache persistente: {e}")
    
    def _cleanup(self) -> None:
        """Limpiar entradas expiradas."""
        expired = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        for key in expired:
            self.delete(key)
    
    def _evict_oldest(self) -> None:
        """Removesr entrada más antigua."""
        if not self.cache:
            return
            
        oldest_key = min(
            self.cache.items(),
            key=lambda x: x[1].timestamp
        )[0]
        self.delete(oldest_key)
    
    def _persist_entry(self, key: str) -> None:
        """Persistir entrada a disco."""
        try:
            entry = self.cache[key]
            persist_file = self.persist_dir / f"{key}.json"
            
            data = {
                "value": entry.value,
                "timestamp": entry.timestamp,
                "ttl": entry.ttl
            }
            
            with open(persist_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"⚠️ Error al persistir {key}: {e}")
    
    def _load_persisted(self) -> None:
        """Cargar entradas persistidas."""
        try:
            for persist_file in self.persist_dir.glob("*.json"):
                key = persist_file.stem
                
                with open(persist_file, "r") as f:
                    data = json.load(f)
                
                entry = CacheEntry(
                    value=data["value"],
                    ttl=data["ttl"]
                )
                entry.timestamp = data["timestamp"]
                
                if not entry.is_expired():
                    self.cache[key] = entry
                else:
                    persist_file.unlink()
                    
            logger.info(f"📂 {len(self.cache)} entradas cargadas de disco")
            
        except Exception as e:
            logger.warning(f"⚠️ Error al cargar cache persistente: {e}")

class CacheDecorator:
    """Decorador para cachear funciones."""
    
    def __init__(
        self,
        cache: StandaloneCache,
        key_fn: Optional[Callable[..., str]] = None
    ):
        self.cache = cache
        self.key_fn = key_fn or (lambda *args, **kwargs: str(args) + str(kwargs))
    
    def __call__(self, fn: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = self.key_fn(*args, **kwargs)
            
            # Try to get from cache
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            
            # Calcular y guardar
            result = fn(*args, **kwargs)
            self.cache.set(key, result)
            return result
            
        return wrapper

# Factory functions
def create_cache(
    max_size: int = 1000,
    ttl: Optional[float] = None,
    persist_dir: Optional[Union[str, Path]] = None
) -> StandaloneCache:
    """
    Crear instancia de cache.
    
    Args:
        max_size: Tamaño máximo
        ttl: TTL default
        persist_dir: Directorio para persistencia
        
    Returns:
        Instancia de StandaloneCache
    """
    return StandaloneCache(
        max_size=max_size,
        ttl=ttl,
        persist_dir=persist_dir
    )

def cache_decorator(
    cache: StandaloneCache,
    key_fn: Optional[Callable[..., str]] = None
) -> CacheDecorator:
    """
    Crear decorador de cache.
    
    Args:
        cache: Instancia de cache
        key_fn: Función para generar claves
        
    Returns:
        Decorador de cache
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
                
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
            
        return wrapper
    return decorator
        
    def _load_cache_meta(self) -> Dict[str, Any]:
        """Loads o crea los metadatos de cache."""
        meta_path = self.base_dir / "cache_meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        return {
            "entries": {},
            "size": 0,
            "hits": 0,
            "misses": 0
        }
        
    def _save_cache_meta(self):
        """Saves los metadatos de cache."""
        meta_path = self.base_dir / "cache_meta.json"
        with open(meta_path, "w") as f:
            json.dump(self._cache_meta, f, indent=2)
            
    def get(self, key: str) -> Optional[Any]:
        """Gets un valor del cache."""
        with self._lock:
            if key in self._cache:
                self._cache_meta["hits"] += 1
                self._save_cache_meta()
                return self._cache[key]
                
            if key in self._cache_meta["entries"]:
                entry = self._cache_meta["entries"][key]
                checkpoint_id = entry["checkpoint_id"]
                try:
                    value = self.checkpoint_manager.load(checkpoint_id)
                    self._cache[key] = value
                    self._cache_meta["hits"] += 1
                    self._save_cache_meta()
                    return value
                except Exception as e:
                    logger.warning(f"Error cargando cache para {key}: {e}")
                    
            self._cache_meta["misses"] += 1
            self._save_cache_meta()
            return None
            
    def set(self, key: str, value: Any):
        """Saves un valor en el cache."""
        with self._lock:
            # Save in checkpoint
            checkpoint_id = self.checkpoint_manager.save(
                value,
                step=len(self._cache_meta["entries"]),
                metrics={"cache_key": key}
            )
            
            # Actualizar metadatos
            self._cache_meta["entries"][key] = {
                "checkpoint_id": checkpoint_id,
                "timestamp": time.time()
            }
            self._cache[key] = value
            self._save_cache_meta()
            
    def clear(self):
        """Limpia el cache."""
        with self._lock:
            self._cache.clear()
            for entry in self._cache_meta["entries"].values():
                checkpoint_id = entry["checkpoint_id"]
                self.checkpoint_manager._remove_checkpoint(checkpoint_id)
            self._cache_meta["entries"].clear()
            self._cache_meta["size"] = 0
            self._save_cache_meta()
            
    def get_stats(self) -> Dict[str, Any]:
        """Gets statistics del cache."""
        with self._lock:
            return {
                "size": len(self._cache_meta["entries"]),
                "hits": self._cache_meta["hits"],
                "misses": self._cache_meta["misses"],
                "hit_ratio": self._cache_meta["hits"] / (self._cache_meta["hits"] + self._cache_meta["misses"]) if (self._cache_meta["hits"] + self._cache_meta["misses"]) > 0 else 0
            }

class CacheError(Exception):
    """Error base para cache."""
    pass

class CacheEntry(Generic[T]):
    """Entrada de cache con TTL."""
    
    def __init__(self, value: T, ttl: Optional[float] = None):
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """Verifiesr si la entrada expiró."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

class StandaloneCache(Generic[T]):
    """Cache en memoria con persistencia opcional."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: Optional[float] = None,
        persist_dir: Optional[Union[str, Path]] = None
    ):
        self.max_size = max_size
        self.ttl = ttl
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.Lock()
        
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted()
            
        logger.info(f"✅ Cache inicializado (max_size={max_size}, ttl={ttl})")
    
    def get(self, key: str) -> Optional[T]:
        """
        Obtener valor del cache.
        
        Args:
            key: Clave a buscar
            
        Returns:
            Valor o None si no existe/expiró
        """
        with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self.cache[key]
                return None
                
            return entry.value
    
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """
        Guardar valor en cache.
        
        Args:
            key: Clave
            value: Valor
            ttl: TTL específico o None para usar el default
        """
        with self._lock:
            # Limpiar entradas expiradas
            self._cleanup()
            
            # Verificar límite
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Save new entry
            self.cache[key] = CacheEntry(
                value,
                ttl=ttl if ttl is not None else self.ttl
            )
            
            # Persistir si está habilitado
            if self.persist_dir:
                self._persist_entry(key)
    
    def delete(self, key: str) -> None:
        """
        Eliminar entrada del cache.
        
        Args:
            key: Clave a eliminar
        """
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                
                if self.persist_dir:
                    persist_file = self.persist_dir / f"{key}.json"
                    try:
                        persist_file.unlink()
                    except Exception as e:
                        logger.warning(f"⚠️ Error al eliminar {key}: {e}")
    
    def clear(self) -> None:
        """Limpiar todo el cache."""
        with self._lock:
            self.cache.clear()
            
            if self.persist_dir:
                try:
                    for file in self.persist_dir.glob("*.json"):
                        file.unlink()
                except Exception as e:
                    logger.warning(f"⚠️ Error al limpiar cache persistente: {e}")
    
    def _cleanup(self) -> None:
        """Limpiar entradas expiradas."""
        expired = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        for key in expired:
            self.delete(key)
    
    def _evict_oldest(self) -> None:
        """Removesr entrada más antigua."""
        if not self.cache:
            return
            
        oldest_key = min(
            self.cache.items(),
            key=lambda x: x[1].timestamp
        )[0]
        self.delete(oldest_key)
    
    def _persist_entry(self, key: str) -> None:
        """Persistir entrada a disco."""
        try:
            entry = self.cache[key]
            persist_file = self.persist_dir / f"{key}.json"
            
            data = {
                "value": entry.value,
                "timestamp": entry.timestamp,
                "ttl": entry.ttl
            }
            
            with open(persist_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"⚠️ Error al persistir {key}: {e}")
    
    def _load_persisted(self) -> None:
        """Cargar entradas persistidas."""
        try:
            for persist_file in self.persist_dir.glob("*.json"):
                key = persist_file.stem
                
                with open(persist_file, "r") as f:
                    data = json.load(f)
                
                entry = CacheEntry(
                    value=data["value"],
                    ttl=data["ttl"]
                )
                entry.timestamp = data["timestamp"]
                
                if not entry.is_expired():
                    self.cache[key] = entry
                else:
                    persist_file.unlink()
                    
            logger.info(f"📂 {len(self.cache)} entradas cargadas de disco")
            
        except Exception as e:
            logger.warning(f"⚠️ Error al cargar cache persistente: {e}")

class CacheDecorator:
    """Decorador para cachear funciones."""
    
    def __init__(
        self,
        cache: StandaloneCache,
        key_fn: Optional[Callable[..., str]] = None
    ):
        self.cache = cache
        self.key_fn = key_fn or (lambda *args, **kwargs: str(args) + str(kwargs))
    
    def __call__(self, fn: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = self.key_fn(*args, **kwargs)
            
            # Try to get from cache
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            
            # Calcular y guardar
            result = fn(*args, **kwargs)
            self.cache.set(key, result)
            return result
            
        return wrapper

# Factory functions
def create_cache(
    max_size: int = 1000,
    ttl: Optional[float] = None,
    persist_dir: Optional[Union[str, Path]] = None
) -> StandaloneCache:
    """
    Crear instancia de cache.
    
    Args:
        max_size: Tamaño máximo
        ttl: TTL default
        persist_dir: Directorio para persistencia
        
    Returns:
        Instancia de StandaloneCache
    """
    return StandaloneCache(
        max_size=max_size,
        ttl=ttl,
        persist_dir=persist_dir
    )

def cache_decorator(
    cache: StandaloneCache,
    key_fn: Optional[Callable[..., str]] = None
) -> CacheDecorator:
    """
    Crear decorador de cache.
    
    Args:
        cache: Instancia de cache
        key_fn: Función para generar claves
        
    Returns:
        Decorador de cache
    """
    return CacheDecorator(cache, key_fn) 