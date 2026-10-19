"""Audio encoder for CapibaraGPT."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioEncoderConfig:
    """Configuration for audio encoder."""

    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    output_dim: int = 256


class AudioEncoder:
    """Audio encoder for processing waveform data."""

    def __init__(self, config: Optional[AudioEncoderConfig] = None):
        self.config = config or AudioEncoderConfig()
        self.initialized = False
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> bool:
        """Initialize the audio encoder."""
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.logger.warning("NumPy not available for audio encoder")
            return False

        self.initialized = True
        self.logger.info("Audio encoder initialized")
        return True

    def encode(self, audio_data: Any) -> Any:
        """Encode audio data into spectral features."""
        if not self.initialized:
            self.initialize()

        try:
            np = self.np
            waveform = np.asarray(audio_data, dtype=np.float32).flatten()
            if waveform.size == 0:
                return {"features": np.zeros(self.config.output_dim), "encoder": "audio_empty"}

            fft = np.fft.rfft(waveform, n=self.config.n_fft)
            magnitude = np.abs(fft)
            if magnitude.size > self.config.output_dim:
                features = magnitude[: self.config.output_dim]
            else:
                padding = np.zeros(self.config.output_dim - magnitude.size, dtype=np.float32)
                features = np.concatenate([magnitude.astype(np.float32), padding])

            return {
                "features": features,
                "encoder": "audio_fft",
                "sample_rate": self.config.sample_rate,
            }
        except Exception as e:
            self.logger.warning(f"Audio encoding failed: {e}")
            return {"features": audio_data, "error": str(e)}

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.config.output_dim

    def is_available(self) -> bool:
        """Check if audio encoder is available."""
        return self.initialized


def main():
    logger.info("Module audio_encoder.py starting")
    return True


if __name__ == "__main__":
    main()
