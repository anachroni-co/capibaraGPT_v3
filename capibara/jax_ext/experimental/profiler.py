"""
JAX experimental profiler

Profiling utilities.
"""

import time

import logging
logger = logging.getLogger(__name__)

def device_memory_stats():
    """Get device memory statistics."""
    return {"used": 0, "total": 1024*1024*1024}  # Mock 1GB

def start_trace(logdir):
    """Start profiling trace."""
    logger.debug(f"Starting trace in {logdir}")
    return time.time()

def stop_trace():
    """Stop profiling trace."""
    logger.debug("Stopping trace")
    return time.time()

__all__ = ['device_memory_stats', 'start_trace', 'stop_trace']
