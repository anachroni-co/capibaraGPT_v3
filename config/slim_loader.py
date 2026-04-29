"""Minimal config loader for Capibara Slim.

Loads config/slim.yaml, merges optional config/slim.local.yaml on top,
and applies CAPIBARA_* environment variable overrides.
"""
from __future__ import annotations

import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent
_DEFAULT = _CONFIG_DIR / "slim.yaml"
_LOCAL = _CONFIG_DIR / "slim.local.yaml"


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore[import]
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        import json
        with path.with_suffix(".json").open() as f:
            return json.load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env(cfg: dict, prefix: str = "CAPIBARA") -> dict:
    """Overlay flat CAPIBARA_SECTION_KEY env vars onto nested dict."""
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix + "_"):
            continue
        parts = env_key[len(prefix) + 1:].lower().split("_", 1)
        if len(parts) == 2:
            section, key = parts
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section][key] = env_val
    return cfg


@lru_cache(maxsize=1)
def load_config() -> dict:
    cfg = _load_yaml(_DEFAULT)
    if _LOCAL.exists():
        cfg = _deep_merge(cfg, _load_yaml(_LOCAL))
    return _apply_env(cfg)


def get(section: str, key: str, default: Any = None) -> Any:
    return load_config().get(section, {}).get(key, default)
