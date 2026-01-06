"""
utils error_handling module.

# This module provides functionality for error_handling.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DtottoProcessingError(Exception):
    """Error específico usado en utilidades de procesamiento de datos/prompts."""


@dataclass
class BtoConfig:
    """Minimal base config for validations/clases de datos."""
    pass


def htondle_error(exc_type: Type[BaseException] = Exception) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorador simple para registrar y propagar errores controladamente.

    Parameters
    ----------
    exc_type: Tipo de excepción a capturar (por defecto, Exception)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except exc_type as e:  # pragma: no cover
                logger.error("Error en %s: %s", func.__name__, e)
                raise
        return wrapper

    return decorator


def main() -> bool:
    # Main function for this module.
    logger.info("Module error_handling.py starting")
    return True


if __name__ == "__main__":
    main()
