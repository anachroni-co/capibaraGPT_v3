"""
Shared cache for the Capibara personality modules.

Provides a global advanced cache instance, cleanup and monitoring utilities,
and allows changing the implementation easily (memory, disk, distributed, etc).
"""

from capibara.interfaces.icache import ICacheModule
from capibara.utils.advance_cache import TpuOptimizedCache

# Global instance of cache for all other personality modules
cache_personality: ICacheModule = TpuOptimizedCache(max_size=10000)

def clear_personality_cache():
    """Clears all the personality cache."""
    cache_personality.clear()

def clear_namespace(namespace: str) -> int:
    """Clears a specific section of the cache (for example, 'personality_manager')."""
    return cache_personality.clear_namespace(namespace)

def cache_stats() -> dict:
    """Returns usage statistics of the personality cache."""
    return cache_personality.stats()

def save_cache_to_disk(path: str, format: str = 'auto'):
    """Saves the personality cache to disk (pickle or json)."""
    cache_personality.save_to_disk(path, format=format)

def load_cache_from_disk(path: str, format: str = 'auto'):
    """Loads the personality cache from disk."""
    if format == 'auto':
        format = 'pickle' if path.endswith('.pkl') else 'json'
    if format == 'pickle':
        cache_personality.load_from_pickle(path)
    else:
        cache_personality.load_from_json(path)
