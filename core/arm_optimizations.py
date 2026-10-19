"""
ARM optimization utilities for CapibaraGPT.

This module provides **explicit**, non-silent handling of ARM-specific features.
If advanced features are unavailable, capabilities are reported as False and
attempting to use them raises a clear error.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _is_arm() -> bool:
    machine = platform.machine().lower()
    return machine in {"aarch64", "arm64"}


def _detect_axion() -> bool:
    if not _is_arm():
        return False
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            cpuinfo = handle.read().lower()
        return "neoverse" in cpuinfo or "armv9" in cpuinfo
    except Exception:
        return False


HARDWARE_INFO: Dict[str, Any] = {
    "is_arm": _is_arm(),
    "is_axion": _detect_axion(),
}


ARM_CAPABILITIES: Dict[str, Any] = {
    "kleidi_integration": False,
    "onnx_runtime_arm": False,
    "arm_quantization": False,
    "multi_instance_balancer": False,
}
ARM_CAPABILITIES["total_features"] = sum(1 for v in ARM_CAPABILITIES.values() if v)


@dataclass
class ARMOptimizationSuite:
    """Explicit ARM optimization interface."""

    config: Optional[Dict[str, Any]] = None

    @property
    def available(self) -> bool:
        return ARM_CAPABILITIES.get("total_features", 0) > 0

    def _require(self, feature: str) -> None:
        if not self.available:
            raise RuntimeError(
                f"ARM optimization '{feature}' requested but no ARM features are available."
            )

    def kleidi_forward(self, *args, **kwargs):
        self._require("kleidi_forward")
        raise RuntimeError("kleidi_forward is not available in this environment.")

    def sve2_attention(self, *args, **kwargs):
        self._require("sve2_attention")
        raise RuntimeError("sve2_attention is not available in this environment.")

    def quantized_matmul(self, *args, **kwargs):
        self._require("quantized_matmul")
        raise RuntimeError("quantized_matmul is not available in this environment.")

    def optimized_forward(self, *args, **kwargs):
        self._require("optimized_forward")
        raise RuntimeError("optimized_forward is not available in this environment.")

    def get_comprehensive_status(self) -> Dict[str, Any]:
        return {
            "hardware": HARDWARE_INFO,
            "sve_optimizations": {"available": False},
            "memory_pool": {"available": False},
            "quantization": {"available": False},
            "onnx_runtime": {"available": False},
            "capabilities": ARM_CAPABILITIES,
        }


def create_arm_optimization_suite(config: Optional[Dict[str, Any]] = None) -> ARMOptimizationSuite:
    if ARM_CAPABILITIES.get("total_features", 0) <= 0:
        raise RuntimeError("ARM optimization suite is not available on this system.")
    return ARMOptimizationSuite(config=config)


def load_arm_config_from_toml(path: str) -> Dict[str, Any]:
    try:
        import toml  # type: ignore
    except Exception as exc:
        raise RuntimeError("TOML support is required to load ARM config.") from exc
    return toml.load(path)


@dataclass
class ARMAxionInferenceOptimizer:
    """Explicit no-op optimizer for compatibility."""

    enabled: bool = False

    def optimize_inference(self, inputs):
        if not self.enabled:
            logger.warning("ARMAxionInferenceOptimizer is disabled; returning inputs unchanged.")
            return inputs
        raise RuntimeError("ARM inference optimizations are not available.")


ARM_OPTIMIZATIONS: Dict[str, Any] = {
    "arm_optimization_suite": None,
    "arm_kleidi": False,
    "sve2_vectorization": False,
    "arm_quantization_available": False,
    "onnx_runtime_available": False,
    "multi_instance_balancing": False,
}


__all__ = [
    "HARDWARE_INFO",
    "ARM_CAPABILITIES",
    "ARMOptimizationSuite",
    "ARM_OPTIMIZATIONS",
    "create_arm_optimization_suite",
    "load_arm_config_from_toml",
    "ARMAxionInferenceOptimizer",
]
