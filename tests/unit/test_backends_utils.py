"""Unit tests for core/backends utility functions (BACKLOG-006).

Covers detect_available_hardware, get_device_info, sync_device, memory_stats,
and the registry helpers without requiring GPU or TPU hardware.
"""
from __future__ import annotations

import pytest

from core.backends.utils import (
    detect_available_hardware,
    get_device_info,
    sync_device,
    memory_stats,
)
from core.backends.registry import (
    list_available_backends,
    get_default_backend,
    set_default_backend,
    detect_best_backend,
)
from core.backends.base import BackendType, BackendConfig


# ---------------------------------------------------------------------------
# detect_available_hardware
# ---------------------------------------------------------------------------


def test_detect_available_hardware_returns_dict():
    hw = detect_available_hardware()
    assert isinstance(hw, dict)
    assert "cpu" in hw
    assert "gpu" in hw
    assert "tpu" in hw


def test_detect_available_hardware_cpu_always_present():
    hw = detect_available_hardware()
    assert hw["cpu"]["available"] is True
    assert isinstance(hw["cpu"]["cores"], int)
    assert hw["cpu"]["cores"] >= 1


def test_detect_available_hardware_gpu_field_structure():
    hw = detect_available_hardware()
    assert "available" in hw["gpu"]
    assert "devices" in hw["gpu"]
    assert isinstance(hw["gpu"]["devices"], list)


def test_detect_available_hardware_tpu_field_structure():
    hw = detect_available_hardware()
    assert "available" in hw["tpu"]
    assert "devices" in hw["tpu"]


def test_detect_available_hardware_is_cached():
    hw1 = detect_available_hardware()
    hw2 = detect_available_hardware()
    assert hw1 is hw2


# ---------------------------------------------------------------------------
# get_device_info
# ---------------------------------------------------------------------------


def test_get_device_info_auto_returns_dict():
    info = get_device_info()
    assert isinstance(info, dict)
    assert "backend" in info


def test_get_device_info_cpu_explicit():
    info = get_device_info("cpu")
    assert info["backend"] == "cpu"
    assert "cores" in info
    assert "architecture" in info
    assert "processor" in info


def test_get_device_info_gpu_without_hardware():
    # GPU not present in CI; the function must not raise.
    info = get_device_info("gpu")
    assert info["backend"] == "gpu"


def test_get_device_info_tpu_without_hardware():
    info = get_device_info("tpu")
    assert info["backend"] == "tpu"


# ---------------------------------------------------------------------------
# sync_device
# ---------------------------------------------------------------------------


def test_sync_device_cpu_is_noop():
    # On CPU-only machines auto-detect returns nothing to sync.
    sync_device("auto")  # must not raise


def test_sync_device_explicit_cpu():
    sync_device("cpu")  # must not raise


def test_sync_device_gpu_without_hardware():
    sync_device("gpu")  # must not raise (catches exceptions internally)


def test_sync_device_tpu_without_hardware():
    sync_device("tpu")  # must not raise


# ---------------------------------------------------------------------------
# memory_stats
# ---------------------------------------------------------------------------


def test_memory_stats_auto_returns_dict():
    stats = memory_stats("auto")
    assert isinstance(stats, dict)
    assert "allocated_gb" in stats
    assert "reserved_gb" in stats
    assert "total_gb" in stats
    assert "backend" in stats


def test_memory_stats_cpu_path():
    # CPU path should not raise; values default to 0.0 (no GPU/TPU).
    stats = memory_stats("cpu")
    assert stats["backend"] == "cpu"
    assert stats["allocated_gb"] == pytest.approx(0.0)


def test_memory_stats_gpu_without_hardware():
    stats = memory_stats("gpu")
    assert stats["backend"] == "gpu"
    assert stats["allocated_gb"] == pytest.approx(0.0)


def test_memory_stats_tpu_without_hardware():
    stats = memory_stats("tpu")
    assert stats["backend"] == "tpu"


# ---------------------------------------------------------------------------
# BackendConfig
# ---------------------------------------------------------------------------


def test_backend_config_defaults():
    cfg = BackendConfig()
    assert cfg.device == "auto"
    assert cfg.memory_fraction == pytest.approx(0.9)
    assert cfg.use_mixed_precision is True
    assert cfg.distributed is False


def test_backend_config_custom():
    from core.backends.base import DType
    cfg = BackendConfig(device="cpu", memory_fraction=0.5, use_mixed_precision=False)
    assert cfg.device == "cpu"
    assert cfg.memory_fraction == pytest.approx(0.5)
    assert cfg.use_mixed_precision is False


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def test_list_available_backends_returns_dict():
    result = list_available_backends()
    assert isinstance(result, dict)
    # CPU backend must always be registered and available.
    assert result.get("cpu") is True


def test_get_default_backend_is_cpu_on_ci():
    # On a CI machine without GPU/TPU the default backend is CPU.
    backend = get_default_backend()
    assert backend is not None


def test_detect_best_backend_returns_backend_type():
    bt = detect_best_backend()
    assert isinstance(bt, BackendType)


def test_set_and_get_default_backend():
    original = get_default_backend()
    try:
        set_default_backend(BackendType.CPU)
        assert get_default_backend() == BackendType.CPU
    finally:
        if original is not None:
            set_default_backend(original)
