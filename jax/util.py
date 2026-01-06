"""
jax utility functions.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

def safe_operation(func, *args, **kwargs):
    """Execute operation safely with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return None

def validate_input(data: Any) -> bool:
    """Validate input data."""
    return data is not None
