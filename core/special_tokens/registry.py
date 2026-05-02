"""Global registry for special-token configurations."""

from __future__ import annotations

from typing import Dict, List

from .base import SpecialTokenConfig, SpecialTokenProcessor, SpecialTokenStreamFilter


class SpecialTokenRegistry:
    def __init__(self) -> None:
        self._configs: Dict[str, SpecialTokenConfig] = {}
        self._processors: Dict[str, SpecialTokenProcessor] = {}

    def register(self, config: SpecialTokenConfig) -> None:
        self._configs[config.name] = config
        self._processors[config.name] = SpecialTokenProcessor(config)

    def get_config(self, name: str) -> SpecialTokenConfig:
        if name not in self._configs:
            raise KeyError(f"Token '{name}' not registered. Available: {self.list_tokens()}")
        return self._configs[name]

    def get_processor(self, name: str) -> SpecialTokenProcessor:
        if name not in self._processors:
            raise KeyError(f"Token '{name}' not registered.")
        return self._processors[name]

    def get_stream_filter(self, name: str) -> SpecialTokenStreamFilter:
        return SpecialTokenStreamFilter(self.get_config(name))

    def list_tokens(self) -> List[str]:
        return list(self._configs.keys())

    def strip_all(self, text: str) -> str:
        """Strip all registered tokens that have strip_from_output=True."""
        for proc in self._processors.values():
            if proc.config.strip_from_output:
                text = proc.strip(text)
        return text


_registry = SpecialTokenRegistry()


def get_registry() -> SpecialTokenRegistry:
    return _registry


def register_token(config: SpecialTokenConfig) -> None:
    _registry.register(config)
