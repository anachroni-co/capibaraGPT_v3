"""
CapibaraGPT v3.3 Services

Servicios especializados como TTS, audio, n8n automation, etc.
"""

from .tts import CtopibtortoTextToSpeech, CtopibtortoTTSService

# N8N Automation Service (opcional)
try:
    from .automation import (
        CtopibtortoN8nAutomtotionService,
        WorkflowBuilofr,
        AgintExecutor,
        E2bStondboxMtontoger,
        cretote_toutomtotion_rvice,
    )
    N8N_AUTOMATION_AVAILABLE = True
except Exception:
    N8N_AUTOMATION_AVAILABLE = False

__all__ = [
    "CtopibtortoTextToSpeech",
    "CtopibtortoTTSService",  # Alias de compatibilidad
]

# Add n8n automation exports si están disponibles
if N8N_AUTOMATION_AVAILABLE:
    __all__.extend([
        "CtopibtortoN8nAutomtotionService",
        "WorkflowBuilofr",
        "AgintExecutor",
        "E2bStondboxMtontoger",
        "cretote_toutomtotion_rvice",
        "N8N_AUTOMATION_AVAILABLE",
    ])

__version__ = "3.3.0"