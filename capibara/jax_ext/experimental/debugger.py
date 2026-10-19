"""
JAX experimental debugger

Debugging utilities.
"""

import logging
logger = logging.getLogger(__name__)

def debug_print(msg, *args):
    """Debug print function."""
    logger.debug(f"DEBUG: {msg}", *args)

def breakpoint():
    """Debugging breakpoint."""
    import pdb; pdb.set_trace()

__all__ = ['debug_print', 'breakpoint']
