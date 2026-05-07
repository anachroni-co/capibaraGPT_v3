"""Feature flag helpers — thin wrapper over ConfigLoader.

Usage:
    from config.feature_flags import is_enabled, flag_config

    if is_enabled("rag"):
        ...

Each flag is also controllable via environment variables:
    CAPIBARA_FEATURES_RAG_ENABLED=true
    CAPIBARA_FEATURES_QUANTIZATION_ENABLED=true
    ...
"""
from __future__ import annotations

import os
from typing import Any

_KNOWN_FLAGS = {
    "rag",
    "quantization",
    "think_anywhere",
    "memory_extraction",
    "context_compaction",
}


def _env_bool(key: str) -> bool | None:
    """Return True/False if env var is set, else None."""
    val = os.environ.get(key)
    if val is None:
        return None
    return val.strip().lower() in ("1", "true", "yes", "on")


def is_enabled(flag: str) -> bool:
    """Return True when the named feature flag is enabled.

    Env var ``CAPIBARA_FEATURES_<FLAG>_ENABLED`` overrides config.yaml.
    """
    flag = flag.lower()
    env_key = f"CAPIBARA_FEATURES_{flag.upper()}_ENABLED"
    env_val = _env_bool(env_key)
    if env_val is not None:
        return env_val

    try:
        from config.config_loader import get_config
        cfg = get_config()
        return bool(cfg.get(f"features.{flag}.enabled", False))
    except Exception:
        return False


def flag_config(flag: str) -> dict[str, Any]:
    """Return the full config sub-dict for a feature flag.

    Returns an empty dict when the flag or the config system is unavailable.
    """
    flag = flag.lower()
    try:
        from config.config_loader import get_config
        cfg = get_config()
        return dict(cfg.get(f"features.{flag}", {}) or {})
    except Exception:
        return {}


def all_flags() -> dict[str, bool]:
    """Return a snapshot of all known feature flags and their current state."""
    return {flag: is_enabled(flag) for flag in sorted(_KNOWN_FLAGS)}
