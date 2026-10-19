"""Compatibility re-export.

The canonical SpikeSSM implementation lives in
sub_models/experimental/spike_ssm.py (fixed scan/tracer-leak version).
This module keeps the historical `capibara.ssm.spike_ssm` import path alive.
"""

from sub_models.experimental.spike_ssm import (  # noqa: F401
    SpikeSSM,
    AdaptiveSpikeSSM,
    create_spike_ssm,
)

__all__ = ["SpikeSSM", "AdaptiveSpikeSSM", "create_spike_ssm"]
