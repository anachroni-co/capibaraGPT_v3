"""
Training data capture pipeline.

Intercepts model outputs at inference time and converts high-signal
interactions into (prompt, response) training pairs:

  - Web-search grounded responses     → verified factual data
  - Fact-check verdicts               → contradiction-aware pairs
  - External API responses (routing)  → knowledge-distillation data
  - Uncertain spans re-answered       → confidence-calibration data

Entry point:
    from training.data_capture import TrainingDataCapture
    capture = TrainingDataCapture(output_dir="data/captured")
"""

from .capture import TrainingDataCapture, CapturedSample, CaptureConfig
from .router import ConfidenceRouter, RouterConfig

__all__ = [
    "TrainingDataCapture",
    "CapturedSample",
    "CaptureConfig",
    "ConfidenceRouter",
    "RouterConfig",
]
