"""Test-Time Compute Scaling (TTC) integration package for Capibara-6.
Provides unified API and scaler implementations integrated with TPU kernels and core monitors.
"""

from __future__ import annotations

from .test_time_scaling import (
    ComputeStrategy,
    TestTimeConfig,
    TestTimeComputeScaler,
    CapibaraTestTimeScaler,
)
from .test_time_api import TestTimeComputeAPI

__all__ = [
    "ComputeStrategy",
    "TestTimeConfig",
    "TestTimeComputeScaler",
    "CapibaraTestTimeScaler",
    "TestTimeComputeAPI",
]