"""
Safety and Guardrails for CapibaraGPT.

AI Act compliant content filtering and safety mechanisms:
- Input validation and sanitization
- Output content filtering
- Toxicity detection
- Topic restrictions
- PII detection and redaction
- Jailbreak detection
- Rate limiting and abuse prevention
"""

from .guardrails import (
    GuardrailConfig,
    GuardrailLevel,
    ContentCategory,
    FilterAction,
    GuardrailResult,
    InputGuardrail,
    OutputGuardrail,
    ToxicityDetector,
    PIIDetector,
    JailbreakDetector,
    TopicFilter,
    SafetyPipeline,
    create_guardrail_config,
    create_safety_pipeline,
)

__all__ = [
    'GuardrailConfig',
    'GuardrailLevel',
    'ContentCategory',
    'FilterAction',
    'GuardrailResult',
    'InputGuardrail',
    'OutputGuardrail',
    'ToxicityDetector',
    'PIIDetector',
    'JailbreakDetector',
    'TopicFilter',
    'SafetyPipeline',
    'create_guardrail_config',
    'create_safety_pipeline',
]
