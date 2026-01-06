"""
Distributed Caching System for Consensus Operations

This module implements a multi-level distributed caching system for consensus operations:
- L1 Cache: In-memory LRU cache for immediate access
- L2 Cache: Redis-based distributed cache for cluster sharing
- L3 Cache: Disk-based persistent cache for long-term storage
- Intelligent cache warming and invalidation strategies
- Compression support (LZ4, Zstandard, gzip)
- Consistent hashing for cache distribution
- Cache analytics and performance monitoring
"""

import logging
import asyncio
import time
import pickle
import json
import hashlib
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import numpy as np
from collections import OrderedDict, defaultdict
import threading
import weakref

# Redis for distributed caching
try:
    import redis.asyncio as redis
    import redis.exceptions
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Compression libraries
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"               # In-memory cache
    L2_DISTRIBUTED = "l2_distributed"     # Redis distributed cache
    L3_PERSISTENT = "l3_persistent"       # Disk-based cache

class CompressionType(Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTANDARD = "zstd"

class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"                          # Least Recently Used
    LFU = "lfu"                          # Least Frequently Used
    TTL = "ttl"                          # Time To Live
    RANDOM = "random"                    # Random eviction
    ADAPTIVE = "adaptive"                # Adaptive based on usage patterns

@dataclass
class CacheConfig:
    """Configurestion for distributed cache system."""
    
    # L1 Cache (Memory)
    l1_max_size_mb: int = 256
    l1_ttl_seconds: int = 3600
    l1_eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    
    # L2 Cache (Redis)
    l2_enabled: bool = True
    l2_host: str = "localhost"
    l2_port: int = 6379
    l2_db: int = 0
    l2_password: Optional[str] = None
    l2_ttl_seconds: int = 7200
    l2_max_connections: int = 20
    
    # L3 Cache (Disk)
    l3_enabled: bool = True
    l3_directory: str = "cache/consensus"
    l3_max_size_mb: int = 1024
    l3_ttl_days: int = 7
    
    # Compression
    compression_type: CompressionType = CompressionType.LZ4
    compression_threshold_bytes: int = 1024
    
    # Performance
    enable_consistent_hashing: bool = True
    enable_cache_warming: bool = True
    enable_analytics: bool = True

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int = 3600
    compression_used: CompressionType = CompressionType.NONE
    cache_level: CacheLevel = CacheLevel.L1_MEMORY

@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    
    # Hit/miss statistics
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    
    # Performance metrics
    avg_get_time_ms: float = 0.0
    avg_set_time_ms: float = 0.0
    compression_ratio: float = 1.0
    
    # Storage metrics
    l1_size_mb: float = 0.0
    l2_size_mb: float = 0.0
    l3_size_mb: float = 0.0
    
    # Efficiency metrics
    overall_hit_rate: float = 0.0
    memory_efficiency: float = 0.0
    network_efficiency: float = 0.0

class L1MemoryCache:
    """L1 in-memory cache with LRU eviction."""
    
    def __init__(self, max_size_mb: int, ttl_seconds: int, eviction_policy: EvictionPolicy):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.eviction_policy = eviction_policy
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"L1 Memory Cache initialized ({max_size_mb}MB, {eviction_policy.value})")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from L1 cache."""
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if self._is_expired(entry):
                    self._remove_entry(key)
                    return None
                
                # Update access metadata
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # Move to end for LRU
                if self.eviction_policy == EvictionPolicy.LRU:
                    self.cache.move_to_end(key)
                
                return entry.value
            
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in L1 cache."""
        
        with self.lock:
            try:
                # Calculate size
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
                
                # Check if we need to evict
                while (self.current_size_bytes + size_bytes > self.max_size_bytes and 
                       len(self.cache) > 0):
                    self._evict_one()
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size_bytes=size_bytes,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    ttl_seconds=ttl_seconds or self.ttl_seconds,
                    cache_level=CacheLevel.L1_MEMORY
                )
                
                # Store entry
                if key in self.cache:
                    old_entry = self.cache[key]
                    self.current_size_bytes -= old_entry.size_bytes
                
                self.cache[key] = entry
                self.current_size_bytes += size_bytes
                
                return True
                
            except Exception as e:
                logger.error(f"L1 cache set failed for key {key}: {e}")
                return False
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        
        age = datetime.now() - entry.created_at
        return age.total_seconds() > entry.ttl_seconds
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        
        if key in self.cache:
            entry = self.cache[key]
            self.current_size_bytes -= entry.size_bytes
            del self.cache[key]
    
    def _evict_one(self):
        """Evict one entry based on eviction policy."""
        
        if not self.cache:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used (first item)
            key = next(iter(self.cache))
            self._remove_entry(key)
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            min_access_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k].access_count)
            self._remove_entry(min_access_key)
        
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k].created_at)
            self._remove_entry(oldest_key)
        
        else:  # Random
            import random
            key = random.choice(list(self.cache.keys()))
            self._remove_entry(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get L1 cache statistics."""
        
        with self.lock:
            return {
                "entries": len(self.cache),
                "size_mb": self.current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "usage_percentage": (self.current_size_bytes / self.max_size_bytes) * 100,
                "eviction_policy": self.eviction_policy.value
            }

class L2DistributedCache:
    """L2 Redis-based distributed cache."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_pool = None
        self.redis_client = None
        
        if REDIS_AVAILABLE and config.l2_enabled:
            self._initialize_redis()
        
        logger.info(f"L2 Distributed Cache initialized (Redis: {REDIS_AVAILABLE and config.l2_enabled})")
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        
        try:
            self.redis_pool = redis.ConnectionPool(
                host=self.config.l2_host,
                port=self.config.l2_port,
                db=self.config.l2_db,
                password=self.config.l2_password,
                max_connections=self.config.l2_max_connections,
                decode_responses=False  # Keep binary data
            )
            
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            logger.info(f"✅ Redis connection established: {self.config.l2_host}:{self.config.l2_port}")
            
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from L2 cache."""
        
        if not self.redis_client:
            return None
        
        try:
            # Get from Redis
            data = await self.redis_client.get(key)
            
            if data:
                # Decompress if needed
                if self.config.compression_type != CompressionType.NONE:
                    data = self._decompress_data(data)
                
                # Deserialize
                value = pickle.loads(data)
                return value
            
            return None
            
        except Exception as e:
            logger.warning(f"L2 cache get failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in L2 cache."""
        
        if not self.redis_client:
            return False
        
        try:
            # Serialize value
            data = pickle.dumps(value)
            
            # Compress if enabled and data is large enough
            if (self.config.compression_type != CompressionType.NONE and 
                len(data) > self.config.compression_threshold_bytes):
                data = self._compress_data(data)
            
            # Set in Redis with TTL
            ttl = ttl_seconds or self.config.l2_ttl_seconds
            await self.redis_client.setex(key, ttl, data)
            
            return True
            
        except Exception as e:
            logger.warning(f"L2 cache set failed for key {key}: {e}")
            return False
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using configured compression."""
        
        if self.config.compression_type == CompressionType.GZIP:
            return zlib.compress(data, level=6)
        elif self.config.compression_type == CompressionType.LZ4 and LZ4_AVAILABLE:
            return lz4.frame.compress(data)
        elif self.config.compression_type == CompressionType.ZSTANDARD and ZSTD_AVAILABLE:
            cctx = zstd.ZstdCompressor(level=3)
            return cctx.compress(data)
        else:
            return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data using configured compression."""
        
        try:
            if self.config.compression_type == CompressionType.GZIP:
                return zlib.decompress(data)
            elif self.config.compression_type == CompressionType.LZ4 and LZ4_AVAILABLE:
                return lz4.frame.decompress(data)
            elif self.config.compression_type == CompressionType.ZSTANDARD and ZSTD_AVAILABLE:
                dctx = zstd.ZstdDecompressor()
                return dctx.decompress(data)
            else:
                return data
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return data
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get L2 cache statistics."""
        
        if not self.redis_client:
            return {"enabled": False}
        
        try:
            info = await self.redis_client.info()
            return {
                "enabled": True,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(
                    info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1
                )
            }
        except Exception as e:
            logger.warning(f"Failed to get L2 cache stats: {e}")
            return {"enabled": False, "error": str(e)}

class L3PersistentCache:
    """L3 disk-based persistent cache."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.l3_directory)
        
        if config.l3_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Index file for fast lookups
            self.index_file = self.cache_dir / "cache_index.json"
            self.index = self._load_cache_index()
        
        self.lock = threading.Lock()
        
        logger.info(f"L3 Persistent Cache initialized ({config.l3_directory})")
    
    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
        
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from L3 cache."""
        
        if not self.config.l3_enabled:
            return None
        
        with self.lock:
            if key not in self.index:
                return None
            
            entry_info = self.index[key]
            
            # Check TTL
            created_at = datetime.fromisoformat(entry_info["created_at"])
            if datetime.now() - created_at > timedelta(days=self.config.l3_ttl_days):
                self._remove_entry(key)
                return None
            
            try:
                # Load from file
                file_path = self.cache_dir / entry_info["filename"]
                
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        data = f.read()
                    
                    # Decompress if needed
                    if entry_info.get("compressed", False):
                        data = self._decompress_data(data, entry_info.get("compression_type", "none"))
                    
                    # Deserialize
                    value = pickle.loads(data)
                    
                    # Update access time
                    entry_info["last_accessed"] = datetime.now().isoformat()
                    entry_info["access_count"] = entry_info.get("access_count", 0) + 1
                    
                    return value
                else:
                    # File missing, remove from index
                    self._remove_entry(key)
                    return None
                    
            except Exception as e:
                logger.warning(f"L3 cache get failed for key {key}: {e}")
                return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in L3 cache."""
        
        if not self.config.l3_enabled:
            return False
        
        with self.lock:
            try:
                # Serialize value
                data = pickle.dumps(value)
                original_size = len(data)
                
                # Compress if beneficial
                compression_type = CompressionType.NONE
                if original_size > self.config.compression_threshold_bytes:
                    compressed_data = self._compress_data(data)
                    if len(compressed_data) < original_size * 0.8:  # Only use if >20% reduction
                        data = compressed_data
                        compression_type = self.config.compression_type
                
                # Generate filename
                filename = f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.cache"
                file_path = self.cache_dir / filename
                
                # Write to file
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Update index
                self.index[key] = {
                    "filename": filename,
                    "size_bytes": len(data),
                    "original_size_bytes": original_size,
                    "created_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 0,
                    "compressed": compression_type != CompressionType.NONE,
                    "compression_type": compression_type.value,
                    "ttl_seconds": ttl_seconds or (self.config.l3_ttl_days * 24 * 3600)
                }
                
                # Save index periodically
                if len(self.index) % 10 == 0:
                    self._save_cache_index()
                
                return True
                
            except Exception as e:
                logger.error(f"L3 cache set failed for key {key}: {e}")
                return False
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data for L3 storage."""
        
        if self.config.compression_type == CompressionType.GZIP:
            return zlib.compress(data, level=6)
        elif self.config.compression_type == CompressionType.LZ4 and LZ4_AVAILABLE:
            return lz4.frame.compress(data)
        elif self.config.compression_type == CompressionType.ZSTANDARD and ZSTD_AVAILABLE:
            cctx = zstd.ZstdCompressor(level=3)
            return cctx.compress(data)
        else:
            return data
    
    def _decompress_data(self, data: bytes, compression_type: str) -> bytes:
        """Decompress data from L3 storage."""
        
        try:
            if compression_type == "gzip":
                return zlib.decompress(data)
            elif compression_type == "lz4" and LZ4_AVAILABLE:
                return lz4.frame.decompress(data)
            elif compression_type == "zstd" and ZSTD_AVAILABLE:
                dctx = zstd.ZstdDecompressor()
                return dctx.decompress(data)
            else:
                return data
        except Exception as e:
            logger.error(f"L3 decompression failed: {e}")
            return data
    
    def _remove_entry(self, key: str):
        """Remove entry from L3 cache."""
        
        if key in self.index:
            entry_info = self.index[key]
            
            # Remove file
            try:
                file_path = self.cache_dir / entry_info["filename"]
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file for {key}: {e}")
            
            # Remove from index
            del self.index[key]
    
    def cleanup_expired(self):
        """Clean up expired entries."""
        
        with self.lock:
            expired_keys = []
            cutoff_time = datetime.now() - timedelta(days=self.config.l3_ttl_days)
            
            for key, entry_info in self.index.items():
                created_at = datetime.fromisoformat(entry_info["created_at"])
                if created_at < cutoff_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                self._save_cache_index()
                logger.info(f"L3 cleanup: removed {len(expired_keys)} expired entries")

class DistributedConsensusCache:
    """
    🌐 Distributed Consensus Cache System
    
    Multi-level caching system optimized for consensus operations:
    - L1: Fast in-memory cache for immediate access
    - L2: Redis distributed cache for cluster sharing
    - L3: Persistent disk cache for long-term storage
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.metrics = CacheMetrics()
        
        # Initialize cache levels
        self.l1_cache = L1MemoryCache(
            config.l1_max_size_mb, 
            config.l1_ttl_seconds, 
            config.l1_eviction_policy
        )
        
        self.l2_cache = L2DistributedCache(config) if config.l2_enabled else None
        self.l3_cache = L3PersistentCache(config) if config.l3_enabled else None
        
        # Cache warming
        self.warming_queue = asyncio.Queue() if config.enable_cache_warming else None
        
        logger.info("🌐 Distributed Consensus Cache initialized")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache hierarchy."""
        
        start_time = time.time()
        
        try:
            # Try L1 cache first
            value = self.l1_cache.get(key)
            if value is not None:
                self.metrics.l1_hits += 1
                self._update_get_metrics(time.time() - start_time)
                return value
            
            self.metrics.l1_misses += 1
            
            # Try L2 cache
            if self.l2_cache:
                value = await self.l2_cache.get(key)
                if value is not None:
                    self.metrics.l2_hits += 1
                    
                    # Promote to L1
                    self.l1_cache.set(key, value)
                    
                    self._update_get_metrics(time.time() - start_time)
                    return value
                
                self.metrics.l2_misses += 1
            
            # Try L3 cache
            if self.l3_cache:
                value = self.l3_cache.get(key)
                if value is not None:
                    self.metrics.l3_hits += 1
                    
                    # Promote to L1 and L2
                    self.l1_cache.set(key, value)
                    if self.l2_cache:
                        await self.l2_cache.set(key, value)
                    
                    self._update_get_metrics(time.time() - start_time)
                    return value
                
                self.metrics.l3_misses += 1
            
            # Cache miss at all levels
            self._update_get_metrics(time.time() - start_time)
            return default
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return default
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache hierarchy."""
        
        start_time = time.time()
        
        try:
            success_count = 0
            
            # Set in L1 cache
            if self.l1_cache.set(key, value, ttl_seconds):
                success_count += 1
            
            # Set in L2 cache
            if self.l2_cache:
                if await self.l2_cache.set(key, value, ttl_seconds):
                    success_count += 1
            
            # Set in L3 cache for persistence
            if self.l3_cache:
                if self.l3_cache.set(key, value, ttl_seconds):
                    success_count += 1
            
            self._update_set_metrics(time.time() - start_time)
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    def _update_get_metrics(self, operation_time: float):
        """Update cache get metrics."""
        
        operation_time_ms = operation_time * 1000
        
        # Update average get time
        if self.metrics.avg_get_time_ms == 0:
            self.metrics.avg_get_time_ms = operation_time_ms
        else:
            self.metrics.avg_get_time_ms = (
                self.metrics.avg_get_time_ms * 0.9 + operation_time_ms * 0.1
            )
        
        # Update hit rate
        total_requests = (
            self.metrics.l1_hits + self.metrics.l1_misses +
            self.metrics.l2_hits + self.metrics.l2_misses +
            self.metrics.l3_hits + self.metrics.l3_misses
        )
        
        total_hits = self.metrics.l1_hits + self.metrics.l2_hits + self.metrics.l3_hits
        
        if total_requests > 0:
            self.metrics.overall_hit_rate = total_hits / total_requests
    
    def _update_set_metrics(self, operation_time: float):
        """Update cache set metrics."""
        
        operation_time_ms = operation_time * 1000
        
        # Update average set time
        if self.metrics.avg_set_time_ms == 0:
            self.metrics.avg_set_time_ms = operation_time_ms
        else:
            self.metrics.avg_set_time_ms = (
                self.metrics.avg_set_time_ms * 0.9 + operation_time_ms * 0.1
            )
    
    async def warm_cache(self, keys_and_values: List[Tuple[str, Any]]):
        """Warm cache with frequently accessed data."""
        
        if not self.config.enable_cache_warming:
            return
        
        logger.info(f"🔥 Warming cache with {len(keys_and_values)} entries")
        
        for key, value in keys_and_values:
            await self.set(key, value)
        
        logger.info("✅ Cache warming completed")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        
        # For L1 cache
        keys_to_remove = [key for key in self.l1_cache.cache.keys() if pattern in key]
        for key in keys_to_remove:
            self.l1_cache._remove_entry(key)
        
        # For L2 cache (Redis)
        if self.l2_cache and self.l2_cache.redis_client:
            try:
                keys = await self.l2_cache.redis_client.keys(f"*{pattern}*")
                if keys:
                    await self.l2_cache.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"L2 pattern invalidation failed: {e}")
        
        # For L3 cache
        if self.l3_cache:
            with self.l3_cache.lock:
                keys_to_remove = [key for key in self.l3_cache.index.keys() if pattern in key]
                for key in keys_to_remove:
                    self.l3_cache._remove_entry(key)
                
                if keys_to_remove:
                    self.l3_cache._save_cache_index()
        
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching pattern '{pattern}'")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all cache levels."""
        
        # Update size metrics
        l1_stats = self.l1_cache.get_stats()
        self.metrics.l1_size_mb = l1_stats["size_mb"]
        
        return {
            "cache_levels": {
                "l1_memory": {
                    **l1_stats,
                    "hits": self.metrics.l1_hits,
                    "misses": self.metrics.l1_misses,
                    "hit_rate": self.metrics.l1_hits / max(self.metrics.l1_hits + self.metrics.l1_misses, 1)
                },
                "l2_distributed": {
                    "enabled": self.l2_cache is not None,
                    "hits": self.metrics.l2_hits,
                    "misses": self.metrics.l2_misses,
                    "hit_rate": self.metrics.l2_hits / max(self.metrics.l2_hits + self.metrics.l2_misses, 1)
                },
                "l3_persistent": {
                    "enabled": self.l3_cache is not None,
                    "hits": self.metrics.l3_hits,
                    "misses": self.metrics.l3_misses,
                    "hit_rate": self.metrics.l3_hits / max(self.metrics.l3_hits + self.metrics.l3_misses, 1),
                    "entries": len(self.l3_cache.index) if self.l3_cache else 0
                }
            },
            "performance_metrics": {
                "overall_hit_rate": self.metrics.overall_hit_rate,
                "avg_get_time_ms": self.metrics.avg_get_time_ms,
                "avg_set_time_ms": self.metrics.avg_set_time_ms,
                "compression_ratio": self.metrics.compression_ratio
            },
            "storage_metrics": {
                "l1_size_mb": self.metrics.l1_size_mb,
                "l2_size_mb": self.metrics.l2_size_mb,
                "l3_size_mb": self.metrics.l3_size_mb,
                "total_size_mb": self.metrics.l1_size_mb + self.metrics.l2_size_mb + self.metrics.l3_size_mb
            },
            "configuration": {
                "compression_type": self.config.compression_type.value,
                "cache_warming_enabled": self.config.enable_cache_warming,
                "consistent_hashing": self.config.enable_consistent_hashing
            }
        }
    
    async def cleanup_all_levels(self):
        """Clean up all cache levels."""
        
        logger.info("🧹 Cleaning up all cache levels...")
        
        # L1 cleanup (force eviction of old entries)
        with self.l1_cache.lock:
            expired_keys = []
            for key, entry in self.l1_cache.cache.items():
                if self.l1_cache._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.l1_cache._remove_entry(key)
        
        # L3 cleanup (remove expired files)
        if self.l3_cache:
            self.l3_cache.cleanup_expired()
        
        logger.info("✅ All cache levels cleaned up")


# Utility functions and decorators
def consensus_cached(ttl: int = 3600, key_prefix: str = "consensus", 
                    cache_level: CacheLevel = CacheLevel.L1_MEMORY):
    """Decorator for caching consensus operations."""
    
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            cache_key = hashlib.sha256(key_data.encode()).hexdigest()[:16]
            
            # Try to get from cache
            cached_result = await _get_global_cache().get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await _get_global_cache().set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

def expert_response_cached(ttl: int = 1800):
    """Decorator specifically for caching expert responses."""
    return consensus_cached(ttl=ttl, key_prefix="expert_response")

def routing_decision_cached(ttl: int = 900):
    """Decorator specifically for caching routing decisions."""
    return consensus_cached(ttl=ttl, key_prefix="routing_decision")


# Global cache instance
_global_cache: Optional[DistributedConsensusCache] = None

def _get_global_cache() -> DistributedConsensusCache:
    """Get or create global cache instance."""
    
    global _global_cache
    if _global_cache is None:
        config = CacheConfig()
        _global_cache = DistributedConsensusCache(config)
    return _global_cache

def get_consensus_cache(config: Optional[CacheConfig] = None) -> DistributedConsensusCache:
    """Get consensus cache with optional configuration."""
    
    if config is None:
        return _get_global_cache()
    else:
        return DistributedConsensusCache(config)


# Export main components
__all__ = [
    'DistributedConsensusCache',
    'ConsensusMemoryPool',
    'ConsensusMemoryManager',
    'CacheConfig',
    'CacheLevel',
    'CompressionType',
    'EvictionPolicy',
    'CacheMetrics',
    'consensus_cached',
    'expert_response_cached',
    'routing_decision_cached',
    'get_consensus_cache'
]


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Create cache configuration
        config = CacheConfig(
            l1_max_size_mb=128,
            l2_enabled=False,  # Disable Redis for testing
            l3_enabled=True,
            l3_directory="test_cache",
            compression_type=CompressionType.GZIP,
            enable_cache_warming=True
        )
        
        # Create distributed cache
        cache = DistributedConsensusCache(config)
        
        print("🌐 Distributed Consensus Cache Test")
        print("=" * 40)
        
        # Test basic operations
        print("\n📦 Testing basic cache operations...")
        
        test_data = {
            "query_1": {"response": "This is a test response", "quality": 8.5},
            "expert_scores": [0.8, 0.9, 0.7, 0.85],
            "routing_decision": {"experts": ["expert1", "expert2"], "cost": 0.005}
        }
        
        # Set values
        for key, value in test_data.items():
            success = await cache.set(key, value)
            print(f"✅ Set {key}: {success}")
        
        # Get values
        for key in test_data.keys():
            retrieved = await cache.get(key)
            if retrieved is not None:
                print(f"✅ Retrieved {key}: {type(retrieved)}")
            else:
                print(f"❌ Failed to retrieve {key}")
        
        # Test cache warming
        print("\n🔥 Testing cache warming...")
        
        warming_data = [
            ("warm_key_1", {"data": "warming test 1"}),
            ("warm_key_2", {"data": "warming test 2"}),
            ("warm_key_3", {"data": "warming test 3"})
        ]
        
        await cache.warm_cache(warming_data)
        
        # Verify warmed data
        for key, _ in warming_data:
            value = await cache.get(key)
            if value:
                print(f"✅ Warmed data retrieved: {key}")
        
        # Test caching decorator
        print("\n🎭 Testing caching decorators...")
        
        @expert_response_cached(ttl=300)
        async def mock_expert_response(query: str, expert_id: str):
            # Simulate expensive operation
            await asyncio.sleep(0.1)
            return {
                "expert_id": expert_id,
                "response": f"Expert {expert_id} response to: {query}",
                "processing_time": 100
            }
        
        # First call (cache miss)
        start_time = time.time()
        result1 = await mock_expert_response("test query", "expert_1")
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = await mock_expert_response("test query", "expert_1")
        second_call_time = time.time() - start_time
        
        print(f"✅ First call (miss): {first_call_time:.3f}s")
        print(f"✅ Second call (hit): {second_call_time:.3f}s")
        print(f"✅ Speedup: {first_call_time / second_call_time:.1f}x")
        
        # Get comprehensive statistics
        print("\n📊 Cache Statistics:")
        stats = cache.get_comprehensive_stats()
        
        print(f"Overall Hit Rate: {stats['performance_metrics']['overall_hit_rate']:.2%}")
        print(f"Avg Get Time: {stats['performance_metrics']['avg_get_time_ms']:.2f}ms")
        print(f"Avg Set Time: {stats['performance_metrics']['avg_set_time_ms']:.2f}ms")
        
        for level, level_stats in stats["cache_levels"].items():
            if level_stats.get("enabled", True):
                print(f"\n{level.upper()}:")
                print(f"  Hit Rate: {level_stats['hit_rate']:.2%}")
                print(f"  Hits: {level_stats['hits']}")
                print(f"  Misses: {level_stats['misses']}")
        
        # Test cleanup
        print(f"\n🧹 Testing cache cleanup...")
        await cache.cleanup_all_levels()
        
        final_stats = cache.get_comprehensive_stats()
        print(f"Final L1 entries: {final_stats['cache_levels']['l1_memory']['entries']}")
        print(f"Final L3 entries: {final_stats['cache_levels']['l3_persistent']['entries']}")
        
        print(f"\n✅ Distributed cache test completed")
    
    import asyncio
    asyncio.run(main())