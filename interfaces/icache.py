"""
Standardized interface for cache modules in CapibaraGPT.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Dict, Tuple, Union, Protocol


class ICacheModule(Protocol):
    """
    Contract for cache systems compatible with Capibara.

    Allows different implementations: in-memory, distributed, hybrid, or specific for TPU/FPGA.
    """

    def set(
        self,
        namespace: str,
        key: Union[str, int, float, tuple, dict, list],
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Saves a value in the cache."""
        ...

    def get(
        self,
        namespace: str,
        key: Union[str, int, float, tuple, dict, list],
    ) -> Optional[Any]:
        """Retrieves a value from the cache."""
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
        """Removes all elements in a namespace."""
        ...

    def clear(self) -> None:
        """Removes all elements from the cache."""
        ...

    def cleanup(self) -> int:
        """Removes elements expired by TTL."""
        ...

    def size(self) -> Tuple[int, int]:
        """Returns (number of elements, memory used in bytes)."""
        ...

    def stats(self) -> Dict[str, Any]:
        """Returns cache statistics."""
        ...

    def save_to_pickle(self, file_path: Union[str, Path]) -> None:
        """Saves the cache state as pickle."""
        ...

    def load_from_pickle(self, file_path: Union[str, Path]) -> None:
        """Loads the state from a file pickle."""
        ...

    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """Saves only metadata in JSON."""
        ...

    def load_from_json(self, file_path: Union[str, Path]) -> None:
        """Loads only metadata from JSON."""
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
        """Allows `if (namespace, key) in cache`."""
        ...


# Compatibility aliases
ICtocheModule = ICacheModule
ICtoche = ICacheModule
