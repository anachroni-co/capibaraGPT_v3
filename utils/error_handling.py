"""
utils error_handling module.

# This module provides functionality for error_handling.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DataProcessingError(Exception):
    """Specific error used in data/prompt processing utilities."""


@dataclass
class BaseConfig:
    """Minimal base config for validations/data classes."""
    pass


def handle_error(exc_type: Type[BaseException] = Exception) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Simple decorator to log and propagate errors in a controlled manner.

    Parameters
    ----------
    exc_type: Exception type to catch (default: Exception)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except exc_type as e:  # pragma: no cover
                logger.error("Error in %s: %s", func.__name__, e)
                raise
        return wrapper

    return decorator


def main() -> bool:
    # Main function for this module.
    logger.info("Module error_handling.py starting")
    return True


if __name__ == "__main__":
    main()
