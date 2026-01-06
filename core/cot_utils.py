"""Chain of Thought Utility Functions for CapibaraGPT Core.

This module provides utility functions and helpers for Chain of Thought (CoT)
reasoning operations. It includes safe execution wrappers, input validation,
and configuration loading utilities.

The utilities enable:
- Safe operation execution with error handling
- Input data validation
- Integration with CoT configuration system
- Memory optimization configuration support

Key Components:
    - safe_operation: Execute operations with automatic error handling
    - validate_input: Validate input data before processing

Example:
    Safe operation execution:

    >>> from capibara.core.cot_utils import safe_operation
    >>>
    >>> # Execute a function safely
    >>> def risky_operation(x):
    ...     return 1 / x
    >>>
    >>> result = safe_operation(risky_operation, 5)
    >>> print(f"Result: {result}")
    >>>
    >>> # Returns None on error instead of raising exception
    >>> result = safe_operation(risky_operation, 0)
    >>> print(f"Result: {result}")  # None

    Input validation:

    >>> from capibara.core.cot_utils import validate_input
    >>>
    >>> data = {"text": "Hello world"}
    >>> if validate_input(data):
    ...     print("Valid input")
    >>> else:
    ...     print("Invalid input")

Note:
    This module integrates with EnhancedChainOfThoughtConfig and
    MemoryOptimizationConfig for comprehensive CoT configuration management.

See Also:
    - capibara.core.cot: Main Chain of Thought implementation
    - capibara.config.memory_config: Memory optimization configuration
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

import toml

from capibara.core.cot import EnhancedChainOfThoughtConfig
from capibara.config.memory_config import MemoryOptimizationConfig

logger = logging.getLogger(__name__)

def safe_operation(func, *args, **kwargs):
    """Execute operation safely with automatic error handling.

    Wraps any function call in try-except block to prevent crashes from
    unexpected errors. Returns None on failure instead of propagating exceptions.

    Args:
        func (callable): Function to execute safely.
        *args: Positional arguments to pass to func.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        Any: Return value from func if successful, None if exception occurred.

    Example:
        >>> def divide(a, b):
        ...     return a / b
        >>>
        >>> # Successful operation
        >>> result = safe_operation(divide, 10, 2)
        >>> print(result)  # 5.0
        >>>
        >>> # Failed operation returns None
        >>> result = safe_operation(divide, 10, 0)
        >>> print(result)  # None
        >>>
        >>> # With keyword arguments
        >>> result = safe_operation(divide, a=20, b=4)
        >>> print(result)  # 5.0

    Note:
        Errors are logged at ERROR level before returning None. This is useful
        for non-critical operations where you want to continue execution even
        if the operation fails.

        For critical operations where errors should halt execution, use
        standard try-except blocks instead.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return None

def validate_input(data: Any) -> bool:
    """Validate input data for processing.

    Performs basic validation to ensure data is not None. This is a simple
    sanity check before processing input data in CoT operations.

    Args:
        data (Any): Data to validate. Can be any type.

    Returns:
        bool: True if data is valid (not None), False otherwise.

    Example:
        >>> from capibara.core.cot_utils import validate_input
        >>>
        >>> # Valid inputs
        >>> validate_input("text")  # True
        >>> validate_input({"key": "value"})  # True
        >>> validate_input([1, 2, 3])  # True
        >>> validate_input(0)  # True (zero is valid)
        >>> validate_input("")  # True (empty string is valid)
        >>>
        >>> # Invalid input
        >>> validate_input(None)  # False

    Note:
        This is a minimal validation. For more sophisticated validation
        (type checking, schema validation, etc.), implement custom validators
        in your application code.

        The function only checks for None. Empty containers, empty strings,
        and zero values are considered valid since they may be meaningful
        in certain contexts.
    """
    return data is not None
