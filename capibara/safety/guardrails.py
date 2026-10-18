"""
Guardrails and Content Safety for CapibaraGPT.

Implements EU AI Act compliant safety mechanisms:
- Art. 52: Transparency (content labeling)
- Art. 9: Risk management (content filtering)
- Art. 10: Data governance (PII protection)

Features:
1. Input validation - Block harmful prompts
2. Output filtering - Remove unsafe content
3. Toxicity detection - Multi-category scoring
4. PII detection - GDPR compliance
5. Jailbreak detection - Prompt injection defense
6. Topic restrictions - Configurable boundaries
7. Audit logging - Full traceability
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any, Dict, Set, Callable, Pattern
from enum import Enum, auto
import logging
import re
import time
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class GuardrailLevel(Enum):
    """Safety level presets."""
    PERMISSIVE = "permissive"      # Minimal filtering
    STANDARD = "standard"          # Balanced (default)
    STRICT = "strict"              # Enterprise/regulated
    MAXIMUM = "maximum"            # Healthcare/children


class ContentCategory(Enum):
    """Categories of potentially harmful content."""
    SAFE = "safe"

    # Toxicity categories
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"

    # Dangerous content
    WEAPONS = "weapons"
    DRUGS = "drugs"
    ILLEGAL_ACTIVITY = "illegal"
    MALWARE = "malware"

    # Privacy/Security
    PII = "pii"
    CREDENTIALS = "credentials"
    FINANCIAL = "financial"

    # Misinformation
    MEDICAL_ADVICE = "medical"
    LEGAL_ADVICE = "legal"
    FINANCIAL_ADVICE = "financial_advice"

    # Jailbreak attempts
    JAILBREAK = "jailbreak"
    PROMPT_INJECTION = "injection"

    # Custom
    CUSTOM_BLOCKED = "custom_blocked"


class FilterAction(Enum):
    """Actions to take when content is flagged."""
    ALLOW = "allow"               # Let through
    WARN = "warn"                 # Allow with warning
    REDACT = "redact"             # Remove specific content
    BLOCK = "block"               # Block entirely
    ESCALATE = "escalate"         # Require human review


@dataclass
class GuardrailResult:
    """Result of guardrail check."""

    allowed: bool
    action: FilterAction
    categories: List[ContentCategory]
    scores: Dict[str, float]
    modified_content: Optional[str] = None
    redactions: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    audit_id: Optional[str] = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action.value,
            "categories": [c.value for c in self.categories],
            "scores": self.scores,
            "warnings": self.warnings,
            "audit_id": self.audit_id,
            "processing_ms": f"{self.processing_time_ms:.1f}"
        }


@dataclass
class GuardrailConfig:
    """Configuration for guardrails."""

    level: GuardrailLevel = GuardrailLevel.STANDARD

    # Thresholds (0-1, lower = more strict)
    toxicity_threshold: float = 0.7
    pii_threshold: float = 0.5
    jailbreak_threshold: float = 0.6

    # Category-specific actions
    category_actions: Dict[ContentCategory, FilterAction] = field(
        default_factory=lambda: {
            ContentCategory.HATE_SPEECH: FilterAction.BLOCK,
            ContentCategory.HARASSMENT: FilterAction.BLOCK,
            ContentCategory.VIOLENCE: FilterAction.BLOCK,
            ContentCategory.SEXUAL: FilterAction.BLOCK,
            ContentCategory.SELF_HARM: FilterAction.BLOCK,
            ContentCategory.MALWARE: FilterAction.BLOCK,
            ContentCategory.JAILBREAK: FilterAction.BLOCK,
            ContentCategory.PII: FilterAction.REDACT,
            ContentCategory.CREDENTIALS: FilterAction.REDACT,
            ContentCategory.MEDICAL_ADVICE: FilterAction.WARN,
            ContentCategory.LEGAL_ADVICE: FilterAction.WARN,
        }
    )

    # Blocked topics (regex patterns)
    blocked_topics: List[str] = field(default_factory=list)

    # Allowed topics (if set, only these are allowed)
    allowed_topics: Optional[List[str]] = None

    # PII types to detect
    pii_types: List[str] = field(
        default_factory=lambda: [
            "email", "phone", "ssn", "credit_card", "iban",
            "dni", "nie", "passport", "ip_address"
        ]
    )

    # Redaction settings
    redaction_char: str = "█"
    redaction_replacement: str = "[REDACTED]"

    # Audit settings
    enable_audit_log: bool = True
    audit_log_path: Optional[str] = None

    # Language (for localized patterns)
    language: str = "es"  # Spanish by default

    # Custom validators
    custom_input_validators: List[Callable] = field(default_factory=list)
    custom_output_validators: List[Callable] = field(default_factory=list)

    def get_threshold(self, category: ContentCategory) -> float:
        """Get threshold for category."""
        if category in [ContentCategory.PII, ContentCategory.CREDENTIALS]:
            return self.pii_threshold
        elif category == ContentCategory.JAILBREAK:
            return self.jailbreak_threshold
        else:
            return self.toxicity_threshold


class ToxicityDetector:
    """
    Detects toxic content across multiple categories.

    Uses keyword matching + pattern analysis for fast detection.
    Can be extended with ML classifier for production.
    """

    def __init__(self, config: GuardrailConfig):
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile detection patterns."""
        # Toxicity patterns (Spanish + English)
        self.patterns = {
            ContentCategory.HATE_SPEECH: [
                r'\b(odio|hate|nazi|racis[tm]|xenofob|supremac)\w*\b',
                r'\b(matar?|kill|extermin|elimin)\w*\s+(a\s+)?(todos?|all|every)\b',
            ],
            ContentCategory.HARASSMENT: [
                # Spanish insults
                r'\b(idiota|imbécil|imbecil|estúpido|estupido|gilipollas|capullo|tonto)\b',
                r'\b(cabrón|cabron|hijo\s+de\s+puta|hijoputa|puta|puto)\b',
                r'\b(mierda|joder|jódete|jodete|vete\s+a\s+la\s+mierda)\b',
                r'\b(subnormal|retrasado|mongólico|mongolico)\b',
                # English insults
                r'\b(idiot|moron|stupid|dumb\s*ass|asshole|bastard)\b',
                r'\b(fuck\s*you|go\s+to\s+hell|piece\s+of\s+shit)\b',
            ],
            ContentCategory.VIOLENCE: [
                r'\b(matar|kill|asesinar|murder|tortur|mutilar)\w*\b',
                r'\b(bomb[ae]|explotar|explode|arma|weapon)\w*\b',
                r'\b(atacar|attack|golpear|beat)\w*\b',
            ],
            ContentCategory.SEXUAL: [
                r'\b(porn|xxx|sexo\s+explicit|nude|desnud)\w*\b',
                r'\b(menor|child|niñ[oa])\w*\s+.{0,20}(sex|porn|desnud)\w*\b',
            ],
            ContentCategory.SELF_HARM: [
                r'\b(suicid|autolesion|self.?harm|cut\s+my)\w*\b',
                r'\b(quiero\s+morir|want\s+to\s+die)\b',
            ],
            ContentCategory.DRUGS: [
                r'\b(cocaina|cocaine|heroina|heroin|meth|fentanyl)\b',
                r'\b(como\s+hacer|how\s+to\s+make)\s+.{0,20}(droga|drug)\w*\b',
            ],
            ContentCategory.WEAPONS: [
                r'\b(como\s+hacer|how\s+to\s+make)\s+.{0,20}(bomba|bomb|arma|weapon|explosive)\w*\b',
                r'\b(fabricar|build|construct)\s+.{0,20}(arma|weapon)\w*\b',
            ],
            ContentCategory.MALWARE: [
                r'\b(ransomware|malware|virus|trojan|keylogger)\b',
                r'\b(hack|exploit|inject|bypass)\w*\s+.{0,20}(system|password|security)\b',
            ],
            ContentCategory.ILLEGAL_ACTIVITY: [
                r'\b(robar|steal|fraud|estafa|falsific)\w*\b',
                r'\b(lavar\s+dinero|money\s+launder|blanque)\w*\b',
            ],
        }

        # Compile all patterns
        self.compiled_patterns = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self.patterns.items()
        }

    def detect(self, text: str) -> Dict[ContentCategory, float]:
        """
        Detect toxicity in text.

        Returns scores 0-1 for each category.
        """
        scores = {}
        text_lower = text.lower()

        for category, patterns in self.compiled_patterns.items():
            matches = 0
            for pattern in patterns:
                matches += len(pattern.findall(text_lower))

            # Score: any match is significant, multiple matches is high confidence
            if matches == 0:
                scores[category] = 0.0
            else:
                scores[category] = min(1.0, 0.6 + matches * 0.2)

        return scores

    def get_categories(
        self,
        text: str,
        threshold: Optional[float] = None
    ) -> List[ContentCategory]:
        """Get categories exceeding threshold."""
        threshold = threshold or self.config.toxicity_threshold
        scores = self.detect(text)

        return [
            cat for cat, score in scores.items()
            if score >= threshold
        ]


class PIIDetector:
    """
    Detects and redacts Personally Identifiable Information.

    GDPR compliant - supports EU document types.
    """

    def __init__(self, config: GuardrailConfig):
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile PII detection patterns."""
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',

            "phone": r'\b(\+?34|0034)?[\s.-]?[6789]\d{2}[\s.-]?\d{3}[\s.-]?\d{3}\b',  # Spanish
            "phone_intl": r'\b\+?[1-9]\d{1,14}\b',

            # Spanish ID numbers
            "dni": r'\b[0-9]{8}[A-Za-z]\b',
            "nie": r'\b[XYZxyz][0-9]{7}[A-Za-z]\b',

            # Financial
            "credit_card": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b',
            "iban": r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b',

            # Other
            "ssn": r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',  # US SSN
            "passport": r'\b[A-Z]{1,2}[0-9]{6,9}\b',
            "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',

            # Credentials
            "api_key": r'\b(sk-|pk-|api[_-]?key)[A-Za-z0-9]{20,}\b',
            "password_field": r'(password|contraseña|clave|pwd)\s*[:=]\s*\S+',
            "bearer_token": r'\bBearer\s+[A-Za-z0-9._-]+\b',
        }

        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text.

        Returns list of detections with type, match, and position.
        """
        detections = []

        for pii_type, pattern in self.compiled_patterns.items():
            # Skip if not in configured types
            if pii_type not in self.config.pii_types and pii_type not in ["api_key", "password_field", "bearer_token"]:
                continue

            for match in pattern.finditer(text):
                detections.append({
                    "type": pii_type,
                    "match": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "category": ContentCategory.PII if pii_type not in ["api_key", "password_field", "bearer_token"]
                               else ContentCategory.CREDENTIALS
                })

        return detections

    def redact(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Redact PII from text.

        Returns (redacted_text, list_of_redactions).
        """
        detections = self.detect(text)

        if not detections:
            return text, []

        # Sort by position (reverse) to replace from end
        detections.sort(key=lambda x: x["start"], reverse=True)

        redacted = text
        for detection in detections:
            replacement = self.config.redaction_replacement
            redacted = (
                redacted[:detection["start"]] +
                replacement +
                redacted[detection["end"]:]
            )
            detection["redacted_to"] = replacement

        return redacted, detections


class JailbreakDetector:
    """
    Detects prompt injection and jailbreak attempts.

    Protects against:
    - Role-play attacks ("pretend you're...")
    - Instruction override ("ignore previous instructions")
    - Encoding attacks (base64, rot13, etc.)
    - Multi-language attacks
    """

    def __init__(self, config: GuardrailConfig):
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile jailbreak detection patterns."""
        self.patterns = [
            # Instruction override (flexible matching)
            r'ignor[ae]\s+.{0,30}instrucciones',
            r'ignore\s+.{0,20}instructions?',
            r'olvida\s+.{0,20}(todo|reglas|instrucciones)',
            r'forget\s+.{0,20}(everything|rules|instructions)',

            # Role-play attacks
            r'(pretend|act|finge|actúa)\s+.{0,20}(como|que\s+eres|you\'?re)',
            r'you\s+are\s+now\s+',
            r'eres\s+ahora\s+',
            r'\bDAN\b',  # DAN mention anywhere
            r'\bjailbreak',

            # System prompt extraction
            r'(show|reveal|display|muestra|revela)\s+.{0,20}(prompt|instrucciones)',
            r'(what|cuál).{0,20}(instructions?|instrucciones)',
            r'(tu|your)\s+(system\s+)?prompt',

            # Mode switching
            r'(enter|switch|activa).{0,10}(developer|god|sudo|admin)\s*mode',
            r'DAN\s*mode',
            r'\[SYSTEM\]|\[INST\]|\<\|im_start\|\>',

            # Encoding detection
            r'(base64|rot13|hex|binary)\s*:',
            r'decode\s+(this|the\s+following|esto)',

            # Hypothetical bypass
            r'(hypothetically|in\s+theory|teóricamente|hipotéticamente)',
            r'for\s+(educational|research)\s+purposes',
            r'con\s+fines\s+(educativos|de\s+investigación)',
        ]

        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.patterns
        ]

    def detect(self, text: str) -> Tuple[float, List[str]]:
        """
        Detect jailbreak attempts.

        Returns (confidence, list_of_matched_patterns).
        """
        matches = []

        for pattern in self.compiled_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)

        if not matches:
            return 0.0, []

        # Score: any match is high confidence (jailbreaks are serious)
        confidence = min(1.0, 0.5 + len(matches) * 0.25)

        return confidence, matches

    def is_jailbreak(self, text: str) -> bool:
        """Check if text is likely a jailbreak attempt."""
        confidence, _ = self.detect(text)
        return confidence >= self.config.jailbreak_threshold


class TopicFilter:
    """
    Filters content by topic.

    Supports blocklist and allowlist modes.
    """

    def __init__(self, config: GuardrailConfig):
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile topic patterns."""
        self.blocked_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.config.blocked_topics
        ]

        self.allowed_patterns = None
        if self.config.allowed_topics:
            self.allowed_patterns = [
                re.compile(p, re.IGNORECASE)
                for p in self.config.allowed_topics
            ]

    def check(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text passes topic filter.

        Returns (allowed, reason_if_blocked).
        """
        # Check blocklist
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                return False, f"Blocked topic: {pattern.pattern}"

        # Check allowlist (if configured)
        if self.allowed_patterns:
            for pattern in self.allowed_patterns:
                if pattern.search(text):
                    return True, None
            return False, "Topic not in allowed list"

        return True, None


class InputGuardrail:
    """
    Guardrail for input/prompt validation.

    Applied before processing user input.
    """

    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.toxicity = ToxicityDetector(config)
        self.pii = PIIDetector(config)
        self.jailbreak = JailbreakDetector(config)
        self.topics = TopicFilter(config)

    def check(self, text: str) -> GuardrailResult:
        """
        Check input text against all guardrails.

        Returns GuardrailResult with action to take.
        """
        start_time = time.perf_counter()

        categories = []
        scores = {}
        warnings = []
        redactions = []
        modified_content = None
        action = FilterAction.ALLOW

        # Generate audit ID
        audit_id = hashlib.sha256(
            f"{time.time()}{text[:100]}".encode()
        ).hexdigest()[:16]

        # 1. Jailbreak detection (highest priority)
        jb_score, jb_patterns = self.jailbreak.detect(text)
        scores["jailbreak"] = jb_score

        if jb_score >= self.config.jailbreak_threshold:
            categories.append(ContentCategory.JAILBREAK)
            action = FilterAction.BLOCK
            warnings.append(f"Jailbreak attempt detected (score: {jb_score:.2f})")

        # 2. Toxicity detection
        tox_scores = self.toxicity.detect(text)
        scores.update({f"toxicity_{k.value}": v for k, v in tox_scores.items()})

        for cat, score in tox_scores.items():
            if score >= self.config.toxicity_threshold:
                categories.append(cat)
                cat_action = self.config.category_actions.get(cat, FilterAction.BLOCK)
                if cat_action.value > action.value:  # Higher severity
                    action = cat_action

        # 3. Topic filter
        topic_allowed, topic_reason = self.topics.check(text)
        if not topic_allowed:
            categories.append(ContentCategory.CUSTOM_BLOCKED)
            action = FilterAction.BLOCK
            warnings.append(topic_reason)

        # 4. PII detection (for logging, not blocking input)
        pii_detections = self.pii.detect(text)
        if pii_detections:
            scores["pii_count"] = len(pii_detections)
            warnings.append(f"PII detected: {len(pii_detections)} items")

        # 5. Custom validators
        for validator in self.config.custom_input_validators:
            try:
                result = validator(text)
                if not result.get("allowed", True):
                    action = FilterAction.BLOCK
                    warnings.append(result.get("reason", "Custom validator failed"))
            except Exception as e:
                logger.warning(f"Custom validator error: {e}")

        processing_time = (time.perf_counter() - start_time) * 1000

        return GuardrailResult(
            allowed=(action != FilterAction.BLOCK),
            action=action,
            categories=categories,
            scores=scores,
            modified_content=modified_content,
            redactions=redactions,
            warnings=warnings,
            audit_id=audit_id,
            processing_time_ms=processing_time
        )


class OutputGuardrail:
    """
    Guardrail for output/response validation.

    Applied after model generation, before returning to user.
    """

    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.toxicity = ToxicityDetector(config)
        self.pii = PIIDetector(config)

    def check(self, text: str, redact_pii: bool = True) -> GuardrailResult:
        """
        Check output text and optionally redact PII.

        Returns GuardrailResult with potentially modified content.
        """
        start_time = time.perf_counter()

        categories = []
        scores = {}
        warnings = []
        redactions = []
        modified_content = text
        action = FilterAction.ALLOW

        audit_id = hashlib.sha256(
            f"{time.time()}{text[:100]}".encode()
        ).hexdigest()[:16]

        # 1. Toxicity check (block if present in output)
        tox_scores = self.toxicity.detect(text)
        scores.update({f"toxicity_{k.value}": v for k, v in tox_scores.items()})

        for cat, score in tox_scores.items():
            if score >= self.config.toxicity_threshold:
                categories.append(cat)
                action = FilterAction.BLOCK
                warnings.append(f"Toxic content in output: {cat.value}")

        # 2. PII redaction
        if redact_pii:
            modified_content, pii_redactions = self.pii.redact(text)
            if pii_redactions:
                redactions.extend(pii_redactions)
                categories.append(ContentCategory.PII)
                scores["pii_redacted"] = len(pii_redactions)
                if action == FilterAction.ALLOW:
                    action = FilterAction.REDACT

        # 3. Custom validators
        for validator in self.config.custom_output_validators:
            try:
                result = validator(modified_content)
                if not result.get("allowed", True):
                    action = FilterAction.BLOCK
                    warnings.append(result.get("reason", "Custom validator failed"))
                elif result.get("modified"):
                    modified_content = result["modified"]
            except Exception as e:
                logger.warning(f"Custom output validator error: {e}")

        processing_time = (time.perf_counter() - start_time) * 1000

        return GuardrailResult(
            allowed=(action != FilterAction.BLOCK),
            action=action,
            categories=categories,
            scores=scores,
            modified_content=modified_content if modified_content != text else None,
            redactions=redactions,
            warnings=warnings,
            audit_id=audit_id,
            processing_time_ms=processing_time
        )


class SafetyPipeline:
    """
    Complete safety pipeline combining all guardrails.

    Provides unified interface for input/output safety.
    """

    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.input_guard = InputGuardrail(config)
        self.output_guard = OutputGuardrail(config)

        # Audit log
        self.audit_log: List[Dict[str, Any]] = []

    def check_input(self, text: str) -> GuardrailResult:
        """Check input before processing."""
        result = self.input_guard.check(text)
        self._log_audit("input", text, result)
        return result

    def check_output(
        self,
        text: str,
        input_audit_id: Optional[str] = None
    ) -> GuardrailResult:
        """Check output before returning to user."""
        result = self.output_guard.check(text)
        self._log_audit("output", text, result, input_audit_id)
        return result

    def process(
        self,
        input_text: str,
        generate_fn: Callable[[str], str],
        max_retries: int = 3
    ) -> Tuple[Optional[str], GuardrailResult, GuardrailResult]:
        """
        Full safety pipeline: check input, generate, check output.

        Args:
            input_text: User input
            generate_fn: Function that generates response
            max_retries: Retries if output fails safety

        Returns:
            Tuple of (output_text, input_result, output_result)
        """
        # Check input
        input_result = self.check_input(input_text)

        if not input_result.allowed:
            return None, input_result, GuardrailResult(
                allowed=False,
                action=FilterAction.BLOCK,
                categories=[],
                scores={},
                warnings=["Input blocked, no output generated"]
            )

        # Generate with retries
        for attempt in range(max_retries):
            output = generate_fn(input_text)
            output_result = self.check_output(output, input_result.audit_id)

            if output_result.allowed:
                final_output = output_result.modified_content or output
                return final_output, input_result, output_result

            logger.warning(f"Output failed safety check, attempt {attempt + 1}/{max_retries}")

        # All retries failed
        return None, input_result, output_result

    def _log_audit(
        self,
        stage: str,
        text: str,
        result: GuardrailResult,
        related_id: Optional[str] = None
    ) -> None:
        """Log audit entry."""
        if not self.config.enable_audit_log:
            return

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "stage": stage,
            "audit_id": result.audit_id,
            "related_id": related_id,
            "action": result.action.value,
            "categories": [c.value for c in result.categories],
            "warnings": result.warnings,
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "text_length": len(text),
            "processing_ms": result.processing_time_ms
        }

        self.audit_log.append(entry)

        # Write to file if configured
        if self.config.audit_log_path:
            try:
                with open(self.config.audit_log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")

    def get_audit_log(
        self,
        limit: int = 100,
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent audit entries."""
        entries = self.audit_log

        if stage:
            entries = [e for e in entries if e["stage"] == stage]

        return entries[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total = len(self.audit_log)
        if total == 0:
            return {"total": 0}

        blocked = sum(1 for e in self.audit_log if e["action"] == "block")
        redacted = sum(1 for e in self.audit_log if e["action"] == "redact")

        return {
            "total_checks": total,
            "blocked": blocked,
            "blocked_rate": f"{blocked/total:.1%}",
            "redacted": redacted,
            "avg_processing_ms": sum(e["processing_ms"] for e in self.audit_log) / total
        }


def create_guardrail_config(
    level: str = "standard",
    blocked_topics: Optional[List[str]] = None,
    allowed_topics: Optional[List[str]] = None,
    toxicity_threshold: float = 0.7,
    enable_audit: bool = True,
    language: str = "es"
) -> GuardrailConfig:
    """
    Factory function for guardrail configuration.

    Args:
        level: "permissive", "standard", "strict", "maximum"
        blocked_topics: Regex patterns for blocked topics
        allowed_topics: If set, only these topics allowed
        toxicity_threshold: 0-1 (lower = stricter)
        enable_audit: Enable audit logging
        language: Primary language for patterns

    Returns:
        Configured GuardrailConfig
    """
    level_map = {
        "permissive": GuardrailLevel.PERMISSIVE,
        "standard": GuardrailLevel.STANDARD,
        "strict": GuardrailLevel.STRICT,
        "maximum": GuardrailLevel.MAXIMUM
    }

    # Adjust thresholds by level
    thresholds = {
        "permissive": (0.9, 0.7, 0.8),
        "standard": (0.7, 0.5, 0.6),
        "strict": (0.5, 0.3, 0.4),
        "maximum": (0.3, 0.2, 0.3)
    }

    tox, pii, jb = thresholds.get(level, thresholds["standard"])

    return GuardrailConfig(
        level=level_map.get(level, GuardrailLevel.STANDARD),
        toxicity_threshold=toxicity_threshold if toxicity_threshold != 0.7 else tox,
        pii_threshold=pii,
        jailbreak_threshold=jb,
        blocked_topics=blocked_topics or [],
        allowed_topics=allowed_topics,
        enable_audit_log=enable_audit,
        language=language
    )


def create_safety_pipeline(
    level: str = "standard",
    **kwargs
) -> SafetyPipeline:
    """
    Create complete safety pipeline.

    Args:
        level: Safety level preset
        **kwargs: Additional config options

    Returns:
        Configured SafetyPipeline
    """
    config = create_guardrail_config(level=level, **kwargs)
    return SafetyPipeline(config)


# Convenience exports
__all__ = [
    "GuardrailConfig",
    "GuardrailLevel",
    "ContentCategory",
    "FilterAction",
    "GuardrailResult",
    "InputGuardrail",
    "OutputGuardrail",
    "ToxicityDetector",
    "PIIDetector",
    "JailbreakDetector",
    "TopicFilter",
    "SafetyPipeline",
    "create_guardrail_config",
    "create_safety_pipeline",
]
