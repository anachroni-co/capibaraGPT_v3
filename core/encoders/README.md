# Encoders Module

Minimal, working encoders for vision, video, audio, and simple multimodal fusion.
All implementations are CPU-safe and use NumPy when available.

## Architecture

```
core/encoders/
+-- __init__.py            # Exports + compatibility aliases
+-- vision_encoder.py      # Vision encoder (basic passthrough)
+-- video_encoder.py       # Video encoder (basic passthrough)
+-- audio_encoder.py       # Audio encoder (FFT features)
+-- multimodal_combiner.py # Simple fusion of vision + video
+-- README.md
```

## Quick Start

```python
from capibara.core.encoders import (
    VisionEncoder, VisionEncoderConfig,
    VideoEncoder, VideoEncoderConfig,
    AudioEncoder, AudioEncoderConfig,
    MultimodalCombiner, CombinerConfig,
)

vision = VisionEncoder(VisionEncoderConfig(output_dim=512))
video = VideoEncoder(VideoEncoderConfig(output_dim=512))
audio = AudioEncoder(AudioEncoderConfig(output_dim=256))
combiner = MultimodalCombiner(CombinerConfig(fusion_type="concatenate"))
```

## Vision Encoder

```python
import numpy as np
from capibara.core.encoders import VisionEncoder

encoder = VisionEncoder()
image = np.random.rand(224, 224, 3)
encoded = encoder.encode(image)
print(encoded["encoder"], encoded["shape"])  # vision_basic, (224, 224, 3)
```

## Video Encoder

```python
import numpy as np
from capibara.core.encoders import VideoEncoder

encoder = VideoEncoder()
video = np.random.rand(32, 224, 224, 3)  # T, H, W, C
encoded = encoder.encode(video)
print(encoded["encoder"], encoded.get("fps"))  # video_basic, 30
```

## Audio Encoder

```python
import numpy as np
from capibara.core.encoders import AudioEncoder

encoder = AudioEncoder()
waveform = np.random.randn(16000).astype(np.float32)
encoded = encoder.encode(waveform)
print(encoded["encoder"], encoded["features"].shape)  # audio_fft, (256,)
```

## Multimodal Combiner

```python
from capibara.core.encoders import MultimodalCombiner

combiner = MultimodalCombiner()
combined = combiner.combine({"vision": [1, 2]}, {"video": [3, 4]})
print(combined["fusion_type"])  # concatenate
```

## Notes

- These encoders are intentionally minimal and CPU-safe.
- The API is stable even when optional libraries are missing.
