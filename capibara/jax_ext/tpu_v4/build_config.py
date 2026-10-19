"""
TPU v4 Build Configuration

Configuration classes and templates for building TPU v4 optimized kernels.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class TpuV4BuildConfig:
    """Configuration for TPU v4 build process."""

    # Build settings
    build_dir: str = "build"
    output_dir: str = "dist"
    clean_build: bool = False

    # TPU settings
    tpu_version: str = "v4-32"
    num_chips: int = 32
    hbm_memory_gb: int = 32

    # Optimization settings
    enable_bf16: bool = True
    enable_int8: bool = True
    enable_spmd: bool = True

    # Compiler settings
    xla_flags: List[str] = field(default_factory=lambda: [
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true",
        "--xla_tpu_enable_async_collective_fusion=true",
    ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "build_dir": self.build_dir,
            "output_dir": self.output_dir,
            "tpu_version": self.tpu_version,
            "num_chips": self.num_chips,
            "enable_bf16": self.enable_bf16,
            "enable_int8": self.enable_int8,
        }


# Build templates
CMAKE_TEMPLATE = """
cmake_minimum_required(VERSION 3.18)
project(capibara_tpu_v4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TPU v4 specific flags
add_compile_definitions(TPU_V4_32=1)
add_compile_definitions(HBM_MEMORY_GB=32)

# Find dependencies
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Build targets
add_library(tpu_kernels SHARED
    kernels/gemm_kernel.cc
    kernels/attention_kernel.cc
)
"""

SETUP_PY_TEMPLATE = """
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

setup(
    name="capibara-tpu-v4",
    version="1.0.0",
    description="TPU v4-32 optimized kernels for CapibaraGPT",
    ext_modules=[],
    cmdclass={"build_ext": build_ext},
)
"""


class Config:
    """Simple configuration manager."""

    def __init__(self):
        self.settings: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.settings[key] = value


# Global config instance
config = Config()
