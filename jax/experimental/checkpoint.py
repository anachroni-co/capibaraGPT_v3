
"""
JAX Experimental Checkpoint utilities
"""

import os
import pickle
from typing import Any, Dict, Optional, Callable
import jax
import functools


def save_checkpoint(state: Dict[str, Any], filepath: str, overwrite: bool = False) -> None:
    """Save model checkpoint to file."""
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"Checkpoint file {filepath} already exists")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)


def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """Load model checkpoint from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file {filepath} not found")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def checkpoint_exists(filepath: str) -> bool:
    """Check if checkpoint file exists."""
    return os.path.exists(filepath)


def checkpoint(fn: Callable) -> Callable:
    """
    Checkpoint decorator for memory optimization.
    
    This is a simplified version that tries to use JAX's native checkpointing
    if available, otherwise falls back to a simple passthrough.
    """
    try:
        # Try to use JAX's native checkpoint if available
        if hasattr(jax, 'checkpoint'):
            return jax.checkpoint(fn)
        elif hasattr(jax, 'remat'):
            return jax.remat(fn)
        else:
            # Fallback: return the function unchanged
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapped
    except Exception:
        # If anything fails, just return the original function
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapped


__all__ = [
    'save_checkpoint',
    'load_checkpoint', 
    'checkpoint_exists',
    'checkpoint'
]