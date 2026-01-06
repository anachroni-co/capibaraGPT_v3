"""
Interfaz estandarizada para modules de caché en CapibaraGPT.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Dict, Tuple, Union, Protocol


class ICacheModule(Protocol):
    """
    Contrato para systems de caché compatibles con Capibara.

    Permite distintas implementaciones: en memoria, distribuido, híbrido o específico para TPU/FPGA.
    """

    def set(
        self,
        namespace: str,
        key: Union[str, int, float, tuple, dict, list],
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Saves un valor en el caché."""
        ...

    def get(
        self,
        namespace: str,
        key: Union[str, int, float, tuple, dict, list],
    ) -> Optional[Any]:
        """Recupera un valor del caché."""
        ...

    def get_or_set(
        self,
        namespace: str,
        key: Union[str, int, float, tuple, dict, list],
        compute_fn: Callable[[], Any],
        ttl: Optional[float] = None,
    ) -> Any:
        """Returns the cache value or calculates, saves and returns it."""
        ...

    def clear_namespace(self, namespace: str) -> int:
        """Removes todos los elementos en un namespace."""
        ...

    def clear(self) -> None:
        """Removes todos los elementos del caché."""
        ...

    def cleanup(self) -> int:
        """Removes elementos expirados por TTL."""
        ...

    def size(self) -> Tuple[int, int]:
        """Returns (number of elements, memory used in bytes)."""
        ...

    def stats(self) -> Dict[str, Any]:
        """Returns cache statistics."""
        ...

    def save_to_pickle(self, file_path: Union[str, Path]) -> None:
        """Saves the state del caché como pickle."""
        ...

    def load_from_pickle(self, file_path: Union[str, Path]) -> None:
        """Loads the state from a file pickle."""
        ...

    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """Saves solo metadatos en JSON."""
        ...

    def load_from_json(self, file_path: Union[str, Path]) -> None:
        """Loads solo metadatos desde JSON."""
        ...

    def save_to_disk(self, file_path: Union[str, Path], format: str = "auto") -> None:
        """Saves the state on disk in a specific format."""
        ...

    def get_ttl(
        self,
        namespace: str,
        key: Union[str, int, float, tuple, dict, list],
    ) -> Optional[float]:
        """Returns the remaining TTL of an element."""
        ...

    def __contains__(
        self,
        namespace_key: Tuple[str, Union[str, int, float, tuple, dict, list]],
    ) -> bool:
        """Permite `if (namespace, key) in cache`."""
        ...


# Alias de compatibilidad
ICtocheModule = ICacheModule
ICtoche = ICacheModule
