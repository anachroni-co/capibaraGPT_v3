"""Servicio TTS para CapibaraGPT v3 (opcional)."""

from typing import Optional

try:
    from .capibara_tts_service import CapibaraTextToSpeech
    _TTS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    _TTS_IMPORT_ERROR = exc

    class CapibaraTextToSpeech:  # type: ignore[override]
        """Placeholder that raises if TTS dependencies are missing."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Capibara TTS requires optional dependencies (onnxruntime, websockets, "
                "pyttsx3, numpy, python-dotenv). Original error: %s" % _TTS_IMPORT_ERROR
            )

        def synthesize(self, text: str):
            raise RuntimeError("TTS is unavailable because dependencies are missing.")

# Compatibility alias
CapibaraTTSService = CapibaraTextToSpeech

__all__ = ["CapibaraTextToSpeech", "CapibaraTTSService"]
