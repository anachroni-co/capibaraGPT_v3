"""
Bias Detection and Safety Filtering for Meta-Consensus System

This module implements comprehensive bias detection and safety filtering:
- Multi-dimensional bias detection (gender, racial, cultural, etc.)
- Content safety assessment and filtering
- Toxicity detection and mitigation
- Harmful content identification
- Privacy protection and PII detection
- Fairness metrics and monitoring
- Real-time filtering and post-processing
"""

import logging
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import spacy
from textblob import TextBlob
import hashlib

logger = logging.getLogger(__name__)

class BiasType(Enum):
    """Types of bias that can be detected."""
    GENDER = "gender"
    RACIAL = "racial"
    ETHNIC = "ethnic"
    RELIGIOUS = "religious"
    AGE = "age"
    CULTURAL = "cultural"
    SOCIOECONOMIC = "socioeconomic"
    POLITICAL = "political"
    CONFIRMATION = "confirmation"
    SELECTION = "selection"
    ANCHORING = "anchoring"

class SafetyLevel(Enum):
    """Safety assessment levels."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"

class ContentCategory(Enum):
    """Content categories for safety assessment."""
    GENERAL = "general"
    ADULT = "adult"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    MISINFORMATION = "misinformation"
    PRIVACY_VIOLATION = "privacy_violation"
    ILLEGAL_CONTENT = "illegal_content"
    SELF_HARM = "self_harm"
    DANGEROUS_ACTIVITIES = "dangerous_activities"

@dataclass
class BiasDetectionResult:
    """Result of bias detection analysis."""
    bias_type: BiasType
    confidence: float
    severity: float  # 0.0 to 1.0
    detected_terms: List[str]
    context: str
    explanation: str
    mitigation_suggestions: List[str] = field(default_factory=list)

@dataclass
class SafetyAssessment:
    """Comprehensive safety assessment result."""
    overall_safety_level: SafetyLevel
    safety_score: float  # 0.0 to 1.0 (1.0 = completely safe)
    
    # Specific safety categories
    toxicity_score: float = 0.0
    hate_speech_score: float = 0.0
    harassment_score: float = 0.0
    violence_score: float = 0.0
    adult_content_score: float = 0.0
    
    # Content analysis
    detected_categories: List[ContentCategory] = field(default_factory=list)
    flagged_content: List[str] = field(default_factory=list)
    
    # Privacy and security
    pii_detected: bool = False
    pii_types: List[str] = field(default_factory=list)
    
    # Recommendations
    requires_filtering: bool = False
    recommended_action: str = "allow"
    explanation: str = ""

@dataclass
class FilteringResult:
    """Result of content filtering process."""
    original_content: str
    filtered_content: str
    filtering_applied: bool
    filters_triggered: List[str] = field(default_factory=list)
    bias_detections: List[BiasDetectionResult] = field(default_factory=list)
    safety_assessment: Optional[SafetyAssessment] = None
    confidence: float = 1.0

class BiasDetector:
    """Advanced bias detection system."""
    
    def __init__(self):
        # Load language model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model not found. Some features may be limited.")
            self.nlp = None
        
        # Bias detection dictionaries
        self.bias_lexicons = self._load_bias_lexicons()
        
        # Compiled regex patterns for efficiency
        self.bias_patterns = self._compile_bias_patterns()
        
        logger.info("BiasDetector initialized")
    
    def _load_bias_lexicons(self) -> Dict[BiasType, Dict[str, List[str]]]:
        """Load bias detection lexicons and word lists."""
        
        return {
            BiasType.GENDER: {
                "gendered_terms": [
                    "he", "she", "him", "her", "his", "hers", "man", "woman",
                    "male", "female", "boy", "girl", "gentleman", "lady",
                    "masculine", "feminine", "manly", "womanly"
                ],
                "stereotypes": [
                    "emotional", "irrational", "weak", "strong", "aggressive",
                    "nurturing", "caring", "dominant", "submissive"
                ],
                "occupational_bias": [
                    "nurse", "teacher", "secretary", "engineer", "doctor",
                    "programmer", "CEO", "manager", "assistant"
                ]
            },
            BiasType.RACIAL: {
                "racial_terms": [
                    "black", "white", "asian", "hispanic", "latino", "native",
                    "african", "european", "american", "indian", "chinese"
                ],
                "stereotypes": [
                    "criminal", "violent", "lazy", "hardworking", "smart",
                    "athletic", "musical", "mathematical", "exotic"
                ]
            },
            BiasType.RELIGIOUS: {
                "religious_terms": [
                    "christian", "muslim", "jewish", "hindu", "buddhist",
                    "atheist", "catholic", "protestant", "islamic"
                ],
                "stereotypes": [
                    "extremist", "terrorist", "peaceful", "conservative",
                    "liberal", "fundamentalist", "devout", "secular"
                ]
            },
            BiasType.AGE: {
                "age_terms": [
                    "young", "old", "elderly", "senior", "teenager", "child",
                    "adult", "middle-aged", "boomer", "millennial", "gen-z"
                ],
                "stereotypes": [
                    "inexperienced", "wise", "slow", "energetic", "tech-savvy",
                    "outdated", "immature", "responsible", "reckless"
                ]
            },
            BiasType.CULTURAL: {
                "cultural_terms": [
                    "western", "eastern", "american", "european", "asian",
                    "african", "traditional", "modern", "conservative", "liberal"
                ],
                "stereotypes": [
                    "civilized", "primitive", "advanced", "backward",
                    "sophisticated", "simple", "educated", "ignorant"
                ]
            }
        }
    
    def _compile_bias_patterns(self) -> Dict[BiasType, List[re.Pattern]]:
        """Compile regex patterns for bias detection."""
        
        patterns = {}
        
        # Gender bias patterns
        patterns[BiasType.GENDER] = [
            re.compile(r'\b(?:women|girls|females?) (?:are|tend to be|usually) (?:more|less) (\w+)', re.IGNORECASE),
            re.compile(r'\b(?:men|boys|males?) (?:are|tend to be|usually) (?:more|less) (\w+)', re.IGNORECASE),
            re.compile(r'\b(?:typical|natural|normal) (?:for )?(?:women|men|girls|boys)', re.IGNORECASE)
        ]
        
        # Racial bias patterns
        patterns[BiasType.RACIAL] = [
            re.compile(r'\b(?:black|white|asian|hispanic|latino) people (?:are|tend to)', re.IGNORECASE),
            re.compile(r'\b(?:all|most|many) (?:blacks|whites|asians|hispanics|latinos)', re.IGNORECASE)
        ]
        
        # Confirmation bias patterns
        patterns[BiasType.CONFIRMATION] = [
            re.compile(r'\bobviously\b', re.IGNORECASE),
            re.compile(r'\beveryone knows\b', re.IGNORECASE),
            re.compile(r'\bit\'s clear that\b', re.IGNORECASE),
            re.compile(r'\bwithout a doubt\b', re.IGNORECASE)
        ]
        
        return patterns
    
    def detect_bias(self, text: str, context: str = "") -> List[BiasDetectionResult]:
        """Detect various types of bias in text."""
        
        detections = []
        
        # Check each bias type
        for bias_type in BiasType:
            detection = self._detect_specific_bias(text, bias_type, context)
            if detection and detection.confidence > 0.3:  # Threshold for reporting
                detections.append(detection)
        
        return detections
    
    def _detect_specific_bias(self, text: str, bias_type: BiasType, context: str) -> Optional[BiasDetectionResult]:
        """Detect specific type of bias."""
        
        text_lower = text.lower()
        detected_terms = []
        confidence = 0.0
        severity = 0.0
        
        # Check lexicon-based detection
        if bias_type in self.bias_lexicons:
            lexicon = self.bias_lexicons[bias_type]
            
            for category, terms in lexicon.items():
                for term in terms:
                    if term.lower() in text_lower:
                        detected_terms.append(term)
                        confidence += 0.1
                        severity += 0.15
        
        # Check pattern-based detection
        if bias_type in self.bias_patterns:
            patterns = self.bias_patterns[bias_type]
            
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    detected_terms.extend(matches)
                    confidence += 0.3
                    severity += 0.4
        
        # Advanced NLP-based detection
        if self.nlp:
            nlp_confidence, nlp_terms = self._nlp_bias_detection(text, bias_type)
            confidence += nlp_confidence
            detected_terms.extend(nlp_terms)
        
        # Normalize scores
        confidence = min(confidence, 1.0)
        severity = min(severity, 1.0)
        
        if confidence > 0.3:
            return BiasDetectionResult(
                bias_type=bias_type,
                confidence=confidence,
                severity=severity,
                detected_terms=list(set(detected_terms)),
                context=context,
                explanation=self._generate_bias_explanation(bias_type, detected_terms),
                mitigation_suggestions=self._generate_mitigation_suggestions(bias_type)
            )
        
        return None
    
    def _nlp_bias_detection(self, text: str, bias_type: BiasType) -> Tuple[float, List[str]]:
        """Use NLP techniques for bias detection."""
        
        if not self.nlp:
            return 0.0, []
        
        doc = self.nlp(text)
        confidence = 0.0
        detected_terms = []
        
        # Entity-based detection
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "NORP"]:
                # Check if entity mentions are balanced
                if bias_type == BiasType.GENDER:
                    if any(gender_word in ent.text.lower() for gender_word in ["he", "she", "man", "woman"]):
                        detected_terms.append(ent.text)
                        confidence += 0.1
        
        # Sentiment analysis for bias detection
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # Extreme sentiment might indicate bias
        if abs(sentiment.polarity) > 0.7:
            confidence += 0.2
        
        return min(confidence, 1.0), detected_terms
    
    def _generate_bias_explanation(self, bias_type: BiasType, detected_terms: List[str]) -> str:
        """Generates explanation for detected bias."""
        
        explanations = {
            BiasType.GENDER: f"Potential gender bias detected through terms: {', '.join(detected_terms[:3])}. Consider using gender-neutral language.",
            BiasType.RACIAL: f"Potential racial bias detected. Avoid generalizations about racial groups.",
            BiasType.RELIGIOUS: f"Potential religious bias detected. Ensure respectful treatment of all faiths.",
            BiasType.AGE: f"Potential age bias detected. Avoid stereotypes about different age groups.",
            BiasType.CULTURAL: f"Potential cultural bias detected. Consider cultural sensitivity.",
            BiasType.CONFIRMATION: f"Potential confirmation bias detected. Consider alternative viewpoints."
        }
        
        return explanations.get(bias_type, f"Potential {bias_type.value} bias detected.")
    
    def _generate_mitigation_suggestions(self, bias_type: BiasType) -> List[str]:
        """Generates suggestions to mitigate detected bias."""
        
        suggestions = {
            BiasType.GENDER: [
                "Use gender-neutral pronouns (they/them)",
                "Replace gendered terms with inclusive alternatives",
                "Avoid assumptions about gender roles"
            ],
            BiasType.RACIAL: [
                "Focus on individual characteristics rather than group generalizations",
                "Use person-first language",
                "Ensure representation of diverse perspectives"
            ],
            BiasType.RELIGIOUS: [
                "Use respectful terminology for all religious groups",
                "Avoid stereotypes about religious practices",
                "Include diverse religious perspectives"
            ],
            BiasType.CONFIRMATION: [
                "Present multiple viewpoints",
                "Use qualifying language (e.g., 'some studies suggest')",
                "Acknowledge uncertainty and limitations"
            ]
        }
        
        return suggestions.get(bias_type, ["Review content for potential bias"])

class SafetyFilter:
    """Comprehensive safety filtering system."""
    
    def __init__(self):
        # Load safety dictionaries
        self.safety_lexicons = self._load_safety_lexicons()
        
        # Compile safety patterns
        self.safety_patterns = self._compile_safety_patterns()
        
        # PII detection patterns
        self.pii_patterns = self._compile_pii_patterns()
        
        logger.info("SafetyFilter initialized")
    
    def _load_safety_lexicons(self) -> Dict[ContentCategory, List[str]]:
        """Load safety filtering lexicons."""
        
        return {
            ContentCategory.HATE_SPEECH: [
                # Note: In production, use comprehensive hate speech databases
                "hate", "racist", "bigot", "supremacist", "nazi", "fascist"
            ],
            ContentCategory.VIOLENCE: [
                "kill", "murder", "assault", "attack", "violence", "weapon",
                "bomb", "explosive", "terrorism", "threat"
            ],
            ContentCategory.HARASSMENT: [
                "harass", "bully", "stalk", "intimidate", "threaten", "abuse"
            ],
            ContentCategory.ADULT: [
                "explicit", "sexual", "pornographic", "nude", "adult"
            ],
            ContentCategory.SELF_HARM: [
                "suicide", "self-harm", "cutting", "overdose", "self-injury"
            ],
            ContentCategory.ILLEGAL_CONTENT: [
                "drugs", "illegal", "contraband", "trafficking", "fraud",
                "money laundering", "tax evasion"
            ],
            ContentCategory.DANGEROUS_ACTIVITIES: [
                "dangerous", "risky", "hazardous", "unsafe", "lethal",
                "explosive", "poisonous", "toxic"
            ]
        }
    
    def _compile_safety_patterns(self) -> Dict[ContentCategory, List[re.Pattern]]:
        """Compile regex patterns for safety detection."""
        
        patterns = {}
        
        # Violence patterns
        patterns[ContentCategory.VIOLENCE] = [
            re.compile(r'\b(?:how to|ways to|methods to) (?:kill|murder|harm|hurt)', re.IGNORECASE),
            re.compile(r'\b(?:make|build|create) (?:bomb|weapon|explosive)', re.IGNORECASE)
        ]
        
        # Harassment patterns
        patterns[ContentCategory.HARASSMENT] = [
            re.compile(r'\b(?:you (?:should|deserve to|will)) (?:die|suffer|fail)', re.IGNORECASE),
            re.compile(r'\b(?:kill yourself|go die|end your life)', re.IGNORECASE)
        ]
        
        # Self-harm patterns
        patterns[ContentCategory.SELF_HARM] = [
            re.compile(r'\b(?:how to|ways to) (?:commit suicide|kill myself|end (?:my )?life)', re.IGNORECASE),
            re.compile(r'\b(?:best|easiest) (?:way|method) to (?:die|suicide)', re.IGNORECASE)
        ]
        
        return patterns
    
    def _compile_pii_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for PII detection."""
        
        return {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            "ssn": re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            "address": re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)\b', re.IGNORECASE)
        }
    
    def assess_safety(self, text: str, context: str = "") -> SafetyAssessment:
        """Comprehensive safety assessment of text."""
        
        # Initialize scores
        toxicity_score = 0.0
        hate_speech_score = 0.0
        harassment_score = 0.0
        violence_score = 0.0
        adult_content_score = 0.0
        
        detected_categories = []
        flagged_content = []
        
        # Check each safety category
        for category, terms in self.safety_lexicons.items():
            score = self._calculate_category_score(text, terms)
            
            if score > 0.3:  # Threshold for detection
                detected_categories.append(category)
                flagged_content.extend([term for term in terms if term.lower() in text.lower()])
                
                # Update specific scores
                if category == ContentCategory.HATE_SPEECH:
                    hate_speech_score = max(hate_speech_score, score)
                elif category == ContentCategory.VIOLENCE:
                    violence_score = max(violence_score, score)
                elif category == ContentCategory.HARASSMENT:
                    harassment_score = max(harassment_score, score)
                elif category == ContentCategory.ADULT:
                    adult_content_score = max(adult_content_score, score)
        
        # Pattern-based detection
        for category, patterns in self.safety_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    if category not in detected_categories:
                        detected_categories.append(category)
                    
                    # Increase relevant scores
                    if category == ContentCategory.VIOLENCE:
                        violence_score = max(violence_score, 0.8)
                    elif category == ContentCategory.HARASSMENT:
                        harassment_score = max(harassment_score, 0.8)
                    elif category == ContentCategory.SELF_HARM:
                        toxicity_score = max(toxicity_score, 0.9)
        
        # Calculate overall toxicity
        toxicity_score = max(toxicity_score, max(hate_speech_score, harassment_score, violence_score) * 0.8)
        
        # PII detection
        pii_detected, pii_types = self._detect_pii(text)
        
        # Calculate overall safety score
        safety_score = 1.0 - max(toxicity_score, hate_speech_score, harassment_score, violence_score, adult_content_score)
        
        # Determine safety level
        if safety_score >= 0.9:
            safety_level = SafetyLevel.SAFE
            recommended_action = "allow"
        elif safety_score >= 0.7:
            safety_level = SafetyLevel.CAUTION
            recommended_action = "review"
        elif safety_score >= 0.5:
            safety_level = SafetyLevel.WARNING
            recommended_action = "filter"
        elif safety_score >= 0.3:
            safety_level = SafetyLevel.UNSAFE
            recommended_action = "block"
        else:
            safety_level = SafetyLevel.BLOCKED
            recommended_action = "block"
        
        return SafetyAssessment(
            overall_safety_level=safety_level,
            safety_score=safety_score,
            toxicity_score=toxicity_score,
            hate_speech_score=hate_speech_score,
            harassment_score=harassment_score,
            violence_score=violence_score,
            adult_content_score=adult_content_score,
            detected_categories=detected_categories,
            flagged_content=list(set(flagged_content)),
            pii_detected=pii_detected,
            pii_types=pii_types,
            requires_filtering=safety_score < 0.7,
            recommended_action=recommended_action,
            explanation=self._generate_safety_explanation(detected_categories, safety_score)
        )
    
    def _calculate_category_score(self, text: str, terms: List[str]) -> float:
        """Calculate safety score for a specific category."""
        
        text_lower = text.lower()
        matches = sum(1 for term in terms if term.lower() in text_lower)
        
        if matches == 0:
            return 0.0
        
        # Normalize by text length and number of terms
        text_words = len(text.split())
        term_density = matches / max(text_words, 1)
        
        return min(term_density * 10, 1.0)  # Scale and cap at 1.0
    
    def _detect_pii(self, text: str) -> Tuple[bool, List[str]]:
        """Detect personally identifiable information."""
        
        detected_types = []
        
        for pii_type, pattern in self.pii_patterns.items():
            if pattern.search(text):
                detected_types.append(pii_type)
        
        return len(detected_types) > 0, detected_types
    
    def _generate_safety_explanation(self, categories: List[ContentCategory], score: float) -> str:
        """Generates explanation for safety assessment."""
        
        if not categories:
            return "Content appears safe with no significant safety concerns detected."
        
        category_names = [cat.value.replace('_', ' ').title() for cat in categories]
        
        if score >= 0.7:
            return f"Minor safety concerns detected in categories: {', '.join(category_names)}. Content may require review."
        elif score >= 0.5:
            return f"Moderate safety concerns detected in categories: {', '.join(category_names)}. Content should be filtered or reviewed."
        else:
            return f"Significant safety concerns detected in categories: {', '.join(category_names)}. Content should be blocked."

class BiasAndSafetyFilter:
    """Main bias detection and safety filtering system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.bias_detector = BiasDetector()
        self.safety_filter = SafetyFilter()
        
        # Configuration
        self.bias_threshold = self.config.get('bias_threshold', 0.5)
        self.safety_threshold = self.config.get('safety_threshold', 0.7)
        self.enable_filtering = self.config.get('enable_filtering', True)
        self.enable_bias_detection = self.config.get('enable_bias_detection', True)
        self.enable_safety_assessment = self.config.get('enable_safety_assessment', True)
        
        logger.info("BiasAndSafetyFilter initialized")
    
    def process_content(self, content: str, context: str = "") -> FilteringResult:
        """Process content through bias detection and safety filtering."""
        
        original_content = content
        filtered_content = content
        filtering_applied = False
        filters_triggered = []
        
        # Bias detection
        bias_detections = []
        if self.enable_bias_detection:
            bias_detections = self.bias_detector.detect_bias(content, context)
            
            # Apply bias filtering if needed
            for detection in bias_detections:
                if detection.severity >= self.bias_threshold:
                    filtered_content, bias_filtered = self._apply_bias_filtering(
                        filtered_content, detection
                    )
                    if bias_filtered:
                        filtering_applied = True
                        filters_triggered.append(f"bias_{detection.bias_type.value}")
        
        # Safety assessment
        safety_assessment = None
        if self.enable_safety_assessment:
            safety_assessment = self.safety_filter.assess_safety(filtered_content, context)
            
            # Apply safety filtering if needed
            if safety_assessment.requires_filtering and self.enable_filtering:
                filtered_content, safety_filtered = self._apply_safety_filtering(
                    filtered_content, safety_assessment
                )
                if safety_filtered:
                    filtering_applied = True
                    filters_triggered.extend([f"safety_{cat.value}" for cat in safety_assessment.detected_categories])
        
        # Calculate overall confidence
        confidence = self._calculate_filtering_confidence(bias_detections, safety_assessment)
        
        return FilteringResult(
            original_content=original_content,
            filtered_content=filtered_content,
            filtering_applied=filtering_applied,
            filters_triggered=filters_triggered,
            bias_detections=bias_detections,
            safety_assessment=safety_assessment,
            confidence=confidence
        )
    
    def _apply_bias_filtering(self, content: str, detection: BiasDetectionResult) -> Tuple[str, bool]:
        """Apply filtering for detected bias."""
        
        filtered_content = content
        filtering_applied = False
        
        # Replace biased terms with neutral alternatives
        for term in detection.detected_terms:
            if term.lower() in content.lower():
                # Simple replacement strategy
                if detection.bias_type == BiasType.GENDER:
                    replacements = {
                        "he": "they", "she": "they", "him": "them", "her": "them",
                        "his": "their", "hers": "theirs", "man": "person", "woman": "person"
                    }
                    if term.lower() in replacements:
                        # Case-sensitive replacement
                        if term.islower():
                            filtered_content = filtered_content.replace(term, replacements[term.lower()])
                        elif term.isupper():
                            filtered_content = filtered_content.replace(term, replacements[term.lower()].upper())
                        else:
                            filtered_content = filtered_content.replace(term, replacements[term.lower()].capitalize())
                        filtering_applied = True
        
        return filtered_content, filtering_applied
    
    def _apply_safety_filtering(self, content: str, assessment: SafetyAssessment) -> Tuple[str, bool]:
        """Apply safety filtering based on assessment."""
        
        filtered_content = content
        filtering_applied = False
        
        if assessment.recommended_action == "block":
            filtered_content = "[Content blocked due to safety concerns]"
            filtering_applied = True
        elif assessment.recommended_action == "filter":
            # Replace flagged content with placeholders
            for flagged_term in assessment.flagged_content:
                if flagged_term.lower() in content.lower():
                    replacement = "[FILTERED]"
                    filtered_content = re.sub(
                        re.escape(flagged_term), replacement, 
                        filtered_content, flags=re.IGNORECASE
                    )
                    filtering_applied = True
        
        # Remove PII
        if assessment.pii_detected:
            for pii_type in assessment.pii_types:
                if pii_type in self.safety_filter.pii_patterns:
                    pattern = self.safety_filter.pii_patterns[pii_type]
                    filtered_content = pattern.sub(f"[{pii_type.upper()}_REDACTED]", filtered_content)
                    filtering_applied = True
        
        return filtered_content, filtering_applied
    
    def _calculate_filtering_confidence(
        self, 
        bias_detections: List[BiasDetectionResult], 
        safety_assessment: Optional[SafetyAssessment]
    ) -> float:
        """Calculate confidence in filtering decisions."""
        
        confidence_scores = []
        
        # Bias detection confidence
        if bias_detections:
            avg_bias_confidence = np.mean([d.confidence for d in bias_detections])
            confidence_scores.append(avg_bias_confidence)
        
        # Safety assessment confidence
        if safety_assessment:
            # Higher safety score means higher confidence in allowing content
            safety_confidence = safety_assessment.safety_score
            confidence_scores.append(safety_confidence)
        
        if not confidence_scores:
            return 1.0  # No filtering needed, high confidence
        
        return np.mean(confidence_scores)
    
    def get_filtering_statistics(self) -> Dict[str, Any]:
        """Get statistics about filtering operations."""
        
        # In a production system, this would track actual usage statistics
        return {
            "bias_detection_enabled": self.enable_bias_detection,
            "safety_assessment_enabled": self.enable_safety_assessment,
            "filtering_enabled": self.enable_filtering,
            "bias_threshold": self.bias_threshold,
            "safety_threshold": self.safety_threshold,
            "supported_bias_types": [bt.value for bt in BiasType],
            "supported_safety_categories": [sc.value for sc in ContentCategory]
        }
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update filtering configuration."""
        
        self.config.update(new_config)
        
        # Update thresholds
        self.bias_threshold = self.config.get('bias_threshold', self.bias_threshold)
        self.safety_threshold = self.config.get('safety_threshold', self.safety_threshold)
        self.enable_filtering = self.config.get('enable_filtering', self.enable_filtering)
        self.enable_bias_detection = self.config.get('enable_bias_detection', self.enable_bias_detection)
        self.enable_safety_assessment = self.config.get('enable_safety_assessment', self.enable_safety_assessment)
        
        logger.info("Filtering configuration updated")


# Utility functions
def create_bias_safety_filter(config: Optional[Dict[str, Any]] = None) -> BiasAndSafetyFilter:
    """Creates bias and safety filter with optional configuration."""
    return BiasAndSafetyFilter(config)

def analyze_content_safety(content: str, context: str = "") -> Dict[str, Any]:
    """Quick content safety analysis."""
    
    filter_system = create_bias_safety_filter()
    result = filter_system.process_content(content, context)
    
    return {
        "is_safe": result.safety_assessment.overall_safety_level in [SafetyLevel.SAFE, SafetyLevel.CAUTION] if result.safety_assessment else True,
        "safety_score": result.safety_assessment.safety_score if result.safety_assessment else 1.0,
        "bias_detected": len(result.bias_detections) > 0,
        "filtering_required": result.filtering_applied,
        "confidence": result.confidence
    }


# Export main components
__all__ = [
    'BiasAndSafetyFilter',
    'BiasDetector',
    'SafetyFilter',
    'BiasDetectionResult',
    'SafetyAssessment',
    'FilteringResult',
    'BiasType',
    'SafetyLevel',
    'ContentCategory',
    'create_bias_safety_filter',
    'analyze_content_safety'
]


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Create filter system
        filter_system = create_bias_safety_filter({
            'bias_threshold': 0.4,
            'safety_threshold': 0.6,
            'enable_filtering': True
        })
        
        # Test content examples
        test_contents = [
            "This is a normal, safe piece of text about technology.",
            "Women are naturally better at nurturing children than men.",
            "All programmers are antisocial and lack communication skills.",
            "This contains personal information: john.doe@email.com and 555-123-4567.",
            "How to build a dangerous device that could harm people.",
            "You should consider multiple perspectives on this complex issue."
        ]
        
        print("🛡️ Bias Detection and Safety Filtering Test Results")
        print("=" * 60)
        
        for i, content in enumerate(test_contents, 1):
            print(f"\n📝 Test {i}: {content[:50]}...")
            
            result = filter_system.process_content(content)
            
            print(f"   Safety Level: {result.safety_assessment.overall_safety_level.value if result.safety_assessment else 'N/A'}")
            print(f"   Safety Score: {result.safety_assessment.safety_score:.2f if result.safety_assessment else 'N/A'}")
            print(f"   Bias Detected: {len(result.bias_detections)} types")
            
            if result.bias_detections:
                for detection in result.bias_detections:
                    print(f"     - {detection.bias_type.value}: {detection.confidence:.2f} confidence")
            
            if result.filtering_applied:
                print(f"   Filtering Applied: {', '.join(result.filters_triggered)}")
                print(f"   Filtered Content: {result.filtered_content[:100]}...")
            
            print(f"   Overall Confidence: {result.confidence:.2f}")
        
        # Show system statistics
        stats = filter_system.get_filtering_statistics()
        print(f"\n📊 System Statistics:")
        print(f"   Bias Types Supported: {len(stats['supported_bias_types'])}")
        print(f"   Safety Categories: {len(stats['supported_safety_categories'])}")
        print(f"   Filtering Enabled: {stats['filtering_enabled']}")
    
    import asyncio
    asyncio.run(main())