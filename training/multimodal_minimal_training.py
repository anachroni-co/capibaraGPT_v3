"""Minimal multimodal training loop using CapibaraGPT encoders."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.encoders.audio_encoder import AudioEncoder
from core.encoders.video_encoder import VideoEncoder
from core.encoders.multimodal_combiner import MultimodalCombiner
from modules.specialized_processors import (
    MultimodalFusionProcessor,
    ProcessorConfig,
    ProcessorType,
)

logger = logging.getLogger(__name__)


@dataclass
class MultimodalTrainingConfig:
    """Configuration for the minimal multimodal training loop."""

    steps: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 2


def create_dummy_batch(batch_size: int) -> List[Dict[str, Any]]:
    """Create a dummy multimodal batch."""
    batch = []
    for _ in range(batch_size):
        batch.append(
            {
                "text": "clip description",
                "image": np.zeros((1, 512), dtype=np.float32),
                "audio": np.zeros(1600, dtype=np.float32),
                "video": np.zeros((2, 4, 4, 3), dtype=np.float32),
            }
        )
    return batch


def train_minimal_multimodal(config: MultimodalTrainingConfig) -> Dict[str, Any]:
    """Run a tiny multimodal training loop using encoders + fusion."""
    audio_encoder = AudioEncoder()
    video_encoder = VideoEncoder()
    combiner = MultimodalCombiner()

    fusion_processor = MultimodalFusionProcessor(
        ProcessorConfig(processor_type=ProcessorType.MULTIMODAL_FUSION)
    )

    weight = 1.0
    losses = []

    for step in range(config.steps):
        batch = create_dummy_batch(config.batch_size)
        batch_loss = 0.0

        for sample in batch:
            audio_features = audio_encoder.encode(sample["audio"])["features"]
            video_features = video_encoder.encode(sample["video"])["features"]
            vision_features = combiner.combine(sample["image"], video_features)["combined_features"]["vision"]

            fused = fusion_processor.fuse_modalities(
                {
                    "text": sample["text"],
                    "image": vision_features,
                    "audio": audio_features,
                    "video": video_features,
                }
            )
            fused_features = np.asarray(fused["fused_features"], dtype=np.float32)
            output = fused_features * weight
            loss = float(np.mean(output**2))
            batch_loss += loss

        batch_loss /= config.batch_size
        grad = 2 * batch_loss * weight
        weight -= config.learning_rate * grad
        losses.append(batch_loss)
        logger.info("Step %d | loss=%.6f | weight=%.6f", step, batch_loss, weight)

    return {"losses": losses, "final_weight": weight}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = MultimodalTrainingConfig()
    results = train_minimal_multimodal(config)
    logger.info("Training complete: %s", results)


if __name__ == "__main__":
    main()
