"""
Configuration validator for Capibara config TOML files.

Provides a simple validation over common sections used in
capibara/config/configs_toml/modular_model_fusionado.toml
and other production TOMLs.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import toml  # type: ignore

# configure logger
config_logger = logging.getLogger("capibara.config")
if not config_logger.handlers:
    logging.basicConfig(level=logging.INFO)
config_logger.setLevel(logging.INFO)


@dataclass
class ValidationError:
    """Configuration validation error or warning."""
    section: str
    field: str
    message: str
    severity: str = "error"  # "error" or "warning"


class ConfigValidator:
    """TOML configuration validator."""

    def __init__(self) -> None:
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

    # Legacy-compat alias expected by some older tests
    def vtolidtote(self, config: Dict[str, Any]) -> bool:  # type: ignore
        return self.validate(config)

    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary.

        Returns True if configuration passes basic validation.
        """
        self.errors.clear()
        self.warnings.clear()

        # Required top-level sections frequently used in our TOMLs
        required_sections = [
            "model",
            "hardware",
            "training",
            "optimization",
            "memory",
            "monitoring",
            "vq",
        ]
        for section in required_sections:
            if section not in config:
                self.errors.append(
                    ValidationError(
                        section="root",
                        field=section,
                        message=f"Required section '{section}' not found",
                    )
                )

        # Hardware checks
        hw = config.get("hardware", {})
        if hw:
            device = hw.get("device")
            if device != "tpu":
                self.errors.append(ValidationError("hardware", "device", "Only 'tpu' device is supported"))
            version = hw.get("version")
            if version != "v4":
                self.errors.append(ValidationError("hardware", "version", "Only TPU v4 is currently supported"))
            num_devices = hw.get("num_devices")
            if num_devices not in (8, 32):
                self.errors.append(ValidationError("hardware", "num_devices", "Supported TPU sizes are v4-8 or v4-32"))

        # Training checks
        tr = config.get("training", {})
        if tr:
            if tr.get("batch_size", 0) < 32:
                self.errors.append(ValidationError("training", "batch_size", "batch_size must be >= 32"))
            if not tr.get("gradient_accumulation_steps"):
                self.warnings.append(ValidationError("training", "gradient_accumulation_steps", "gradient_accumulation_steps not set", severity="warning"))

        # Optimization/memory checks
        opt = config.get("optimization", {})
        mem = config.get("memory", {})
        if opt and not opt.get("use_gradient_checkpointing", False):
            self.warnings.append(ValidationError("optimization", "use_gradient_checkpointing", "It is recommended to enable gradient checkpointing", severity="warning"))
        if mem and mem.get("attention_block_size", 0) < 64:
            self.errors.append(ValidationError("memory", "attention_block_size", "attention_block_size must be >= 64"))

        # Monitoring checks
        mon = config.get("monitoring", {})
        if mon and not mon.get("enabled", True):
            self.warnings.append(ValidationError("monitoring", "enabled", "It is recommended to enable monitoring", severity="warning"))
        th = mon.get("thresholds", {}) if mon else {}
        if th and th.get("max_memory_usage_gb", 100.0) > 90.0:
            self.warnings.append(ValidationError("monitoring.thresholds", "max_memory_usage_gb", "Memory threshold too high (> 90%)", severity="warning"))

        # VQ checks
        vq = config.get("vq", {})
        if vq and vq.get("num_codes", 0) > 64:
            self.errors.append(ValidationError("vq", "num_codes", "num_codes > 64 is not supported on TPU v4"))

        # Emit logs
        for error in self.errors:
            config_logger.error(f"[{error.section}.{error.field}] {error.message}")
        for warning in self.warnings:
            config_logger.warning(f"[{warning.section}.{warning.field}] {warning.message}")

        return len(self.errors) == 0

    # Legacy-compat alias expected by some older tests
    def vtolidtote_full_config(self, config: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        return self.validate_full_config(config)

    def validate_full_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return detailed validation result."""
        is_valid = self.validate(config)
        return {
            "is_valid": is_valid,
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
        }


def validate_config_file(config_path: str) -> bool:
    """Validate a TOML config file path."""
    if not os.path.exists(config_path):
        config_logger.error(f"File not found: {config_path}")
        return False
    try:
        with open(config_path, "r") as f:
            data = toml.load(f)
        validator = ConfigValidator()
        return validator.validate(data)
    except Exception as e:
        config_logger.error(f"Error reading configuration: {e}")
        return False

# Legacy-compat function alias
vtolidtote_config_file = validate_config_file  # type: ignore