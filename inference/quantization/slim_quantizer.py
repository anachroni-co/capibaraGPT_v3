"""Capibara Slim — numpy-based INT8 quantization (no torch dependency).

Provides WeightQuantizer (per-channel symmetric), SlimQuantizer (applies to
torch state dicts or nn.Module), and related helpers.
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

QUANTIZATION_AVAILABLE = True  # pure-numpy, always available


@dataclass
class QuantizationConfig:
    weight_percentile: float = 99.9
    kv_percentile: float = 99.5
    target_dtype: str = "int8"
    preserve_float16: bool = True


class WeightQuantizer:
    """Per-channel symmetric INT8 quantization for 2-D weight matrices."""

    def __init__(self, config: Optional[QuantizationConfig] = None) -> None:
        self.config = config or QuantizationConfig()

    def quantize_per_channel_symmetric(
        self,
        weights: np.ndarray,
        percentile: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (int8_weights [H,W], scales [H] float16)."""
        if weights.ndim != 2:
            raise ValueError(
                f"Expected 2D weight tensor, got shape {weights.shape}"
            )
        pct = percentile if percentile is not None else self.config.weight_percentile
        scales = (
            np.percentile(np.abs(weights), pct, axis=1, keepdims=True) / 127.0
        )
        scales = np.clip(scales, 1e-8, None)
        q = np.round(weights / scales).astype(np.int8)
        return q, scales.squeeze(1).astype(np.float16)

    def _is_quantizable_weight(self, arr: np.ndarray) -> bool:
        return (
            isinstance(arr, np.ndarray)
            and arr.ndim == 2
            and arr.dtype in (np.float32, np.float16)
        )


@dataclass
class SlimQuantizationStats:
    total_tensors: int
    quantized_tensors: int
    skipped_tensors: int

    def __str__(self) -> str:
        return (
            f"quantized={self.quantized_tensors}/{self.total_tensors}  "
            f"skipped={self.skipped_tensors}"
        )


class SlimQuantizer:
    """INT8 per-channel quantization for Slim model state dicts.

    Works on dicts of numpy arrays or torch tensors.
    Returns a new state dict with 2-D float tensors replaced by
    ``<name>.W_q`` (int8) + ``<name>.S`` (float16 scales) pairs.
    """

    def __init__(self, config: Optional[QuantizationConfig] = None) -> None:
        self.config = config or QuantizationConfig()
        self._q = WeightQuantizer(self.config)

    def quantize_state_dict(
        self, state_dict: Dict[str, Any]
    ) -> Tuple[SlimQuantizationStats, Dict[str, np.ndarray]]:
        """Return (stats, quantized_dict)."""
        result: Dict[str, np.ndarray] = {}
        total = quantized = skipped = 0

        for name, tensor in state_dict.items():
            total += 1
            arr = (
                tensor.detach().cpu().float().numpy()
                if hasattr(tensor, "detach")
                else np.asarray(tensor, dtype=np.float32)
            )
            if self._q._is_quantizable_weight(arr):
                q, scales = self._q.quantize_per_channel_symmetric(arr)
                result[f"{name}.W_q"] = q
                result[f"{name}.S"] = scales
                quantized += 1
            else:
                result[name] = arr
                skipped += 1

        return SlimQuantizationStats(total, quantized, skipped), result

    def apply_to_model(self, model: Any) -> SlimQuantizationStats:
        """Quantize nn.Linear weights in-place (requires torch)."""
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("apply_to_model requires PyTorch") from exc

        total = quantized = skipped = 0
        for _, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            w = module.weight.detach().cpu().float().numpy()
            total += 1
            if self._q._is_quantizable_weight(w):
                q, scales = self._q.quantize_per_channel_symmetric(w)
                w_dq = q.astype(np.float32) * scales[:, np.newaxis]
                with torch.no_grad():
                    module.weight.copy_(torch.from_numpy(w_dq))
                quantized += 1
            else:
                skipped += 1

        return SlimQuantizationStats(total, quantized, skipped)


def create_weight_quantizer(config_dict: Optional[Dict] = None) -> WeightQuantizer:
    cfg = QuantizationConfig(**(config_dict or {}))
    return WeightQuantizer(cfg)
