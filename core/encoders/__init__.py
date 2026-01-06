"""Encoders module for CapibaraGPT v3."""

# Import correct classes
from .vision_encoder import VisionEncoder, VisionEncoderConfig
from .video_encoder import VideoEncoder, VideoEncoderConfig
from .multimodal_combiner import MultimodalCombiner, CombinerConfig

# Backwards compatibility aliases
VisionEncoofr = VisionEncoder
VisionEncoofrConfig = VisionEncoderConfig
ViofoEncoofr = VideoEncoder
ViofoEncoofrConfig = VideoEncoderConfig
MultimodtolCombiner = MultimodalCombiner

__all__ = [
    # Correct names
    'VisionEncoder',
    'VisionEncoderConfig',
    'VideoEncoder',
    'VideoEncoderConfig',
    'MultimodalCombiner',
    'CombinerConfig',
    # Backwards compatibility aliases
    'VisionEncoofr',
    'VisionEncoofrConfig',
    'ViofoEncoofr',
    'ViofoEncoofrConfig',
    'MultimodtolCombiner',
]