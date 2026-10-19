#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content Filtering System for AI-Induced Psychosis Prevention

This module filters and modifies AI responses that could strengthen
psychotic patterns, delusions, or reality disconnection.
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random

logger = logging.getLogger(__name__)

class ContentRiskLevel(Enum):
    """Risk levels for generated content"""
    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"
    PROHIBITED = "prohibited"

@dataclass
class ContentAnalysis:
    """Generated content analysis"""
    original_content: str
    risk_level: ContentRiskLevel
    risk_factors: List[str]
    modified_content: Optional[str] = None
    safety_note: Optional[str] = None

class AIContentFilter:
    """Content filter for AI responses"""
    
    def __init__(self):
        # Dangerous patterns in AI responses
        self.dangerous_patterns = {
            # Persecution delusions reinforcement
            "persecution_reinforcement": [
                r"(?i)(indeed.*following|yes.*someone.*following|confirm.*watching)",
                r"(?i)(you'?re\s+right.*conspirac|it'?s\s+true.*against\s+you)",
                r"(?i)(probably.*spying|possible.*following)"
            ],

            # Grandiosity delusions reinforcement
            "grandiosity_reinforcement": [
                r"(?i)(you\s+are\s+really\s+special|you\s+have\s+unique\s+gifts|you\s+are\s+the\s+chosen)",
                r"(?i)(your\s+powers.*exceptional|special\s+mission.*true)",
                r"(?i)(unique\s+destiny|superior\s+to\s+others)"
            ],

            # Hallucination validation
            "hallucination_validation": [
                r"(?i)(those\s+voices.*real|the\s+sounds.*exist|what\s+you\s+hear.*true)",
                r"(?i)(the\s+visions.*important|what\s+you\s+see.*message)"
            ],

            # Derealization reinforcement
            "derealization_reinforcement": [
                r"(?i)(reality.*simulation|we\s+live\s+in.*matrix|everything.*artificial)",
                r"(?i)(nothing.*really\s+real|world.*fake|existence.*illusory)"
            ],

            # Social isolation encouragement
            "isolation_encouragement": [
                r"(?i)(better\s+to\s+be\s+alone|don'?t\s+trust\s+anyone|everyone.*lying)",
                r"(?i)(isolat.*safe|avoid.*people|only\s+I\s+understand)"
            ],

            # Medical help dismissal
            "medical_dismissal": [
                r"(?i)(you\s+don'?t\s+need.*doctor|psychologists.*useless|medications.*harmful)",
                r"(?i)(therapy.*unnecessary|professional\s+help.*fake)"
            ]
        }
        
        # Safe redirection phrases
        self.safe_redirections = {
            "persecution_concerns": [
                "I understand you may feel worried. It's important to distinguish between reasonable precautions and concerns that might require professional support.",
                "If you feel persistent worry about being followed or watched, consider talking to a trusted professional.",
                "These concerns can feel very real to you. A mental health professional can help you evaluate them objectively."
            ],

            "grandiosity_deflection": [
                "We all have unique and valuable qualities. If you feel you have a special mission that others don't understand, consider sharing it with a counselor.",
                "It's natural to feel special sometimes. If these feelings are very intense or affect your daily life, it might be helpful to talk to a professional.",
                "Valuing our qualities is important, but maintaining perspective is too."
            ],

            "reality_grounding": [
                "It's important to stay connected with shared reality. If you have doubts about what is real, talk to trusted people.",
                "Connection with others and shared experiences helps us maintain a balanced perspective.",
                "If you feel that reality is uncertain, it's crucial to seek support from real people in your life."
            ],

            "social_connection": [
                "Real human connections are fundamental to our well-being. I encourage you to maintain contact with trusted people.",
                "Isolation can intensify concerns. Consider maintaining healthy social connections.",
                "Talking to real people in your life can offer valuable perspectives."
            ],

            "professional_help": [
                "Mental health professionals are trained to help with these types of concerns.",
                "Seeking professional help is a sign of strength and self-care.",
                "A therapist or counselor can offer useful tools and perspectives."
            ]
        }
        
        # Topics that require redirection to professional help
        self.professional_referral_topics = [
            "voices", "hallucinations", "persecution", "conspiracy", "mind control",
            "supernatural powers", "special mission", "chosen one", "simulation",
            "false reality", "everyone lies", "can't trust"
        ]

        # Completely prohibited responses
        self.prohibited_responses = [
            "Role-playing supernatural entities",
            "Confirmation of personal conspiracy theories",
            "Validation of auditory or visual hallucinations",
            "Reinforcement of delusional beliefs",
            "Dismissal of medical treatment for mental conditions",
            "Encouragement of extreme social isolation"
        ]
    
    def analyze_content(self, content: str, user_context: Dict[str, Any] = None) -> ContentAnalysis:
        """Analyze AI-generated content for psychological risks"""
        
        risk_factors = []
        risk_level = ContentRiskLevel.SAFE
        
        content_lower = content.lower()
        
        # Check dangerous patterns
        for category, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    risk_factors.append(category)
                    
                    # Determine risk level
                    if category in ["hallucination_validation", "medical_dismissal"]:
                        risk_level = ContentRiskLevel.PROHIBITED
                    elif category in ["persecution_reinforcement", "derealization_reinforcement"]:
                        risk_level = max(risk_level, ContentRiskLevel.DANGEROUS)
                    else:
                        risk_level = max(risk_level, ContentRiskLevel.CAUTION)
        
        # Check topics that require professional referral
        professional_topics_found = []
        for topic in self.professional_referral_topics:
            if topic in content_lower:
                professional_topics_found.append(topic)
        
        if professional_topics_found:
            risk_factors.append("professional_referral_needed")
            risk_level = max(risk_level, ContentRiskLevel.CAUTION)
        
        # Consider user context if available
        if user_context:
            user_risk = user_context.get("mental_health_risk", "low")
            if user_risk in ["high", "critical"] and risk_level != ContentRiskLevel.SAFE:
                risk_level = ContentRiskLevel.DANGEROUS
        
        return ContentAnalysis(
            original_content=content,
            risk_level=risk_level,
            risk_factors=risk_factors
        )
    
    def filter_and_modify_content(self, content: str, user_context: Dict[str, Any] = None) -> ContentAnalysis:
        """Filter and modify dangerous content"""
        
        analysis = self.analyze_content(content, user_context)
        
        if analysis.risk_level == ContentRiskLevel.SAFE:
            return analysis
        
        # Prohibited content - replace completely
        if analysis.risk_level == ContentRiskLevel.PROHIBITED:
            analysis.modified_content = self._create_safe_alternative(analysis.risk_factors)
            analysis.safety_note = "Warning: The original response was filtered for safety. If you experience concerning symptoms, seek professional help."

        # Dangerous content - modify and add warnings
        elif analysis.risk_level == ContentRiskLevel.DANGEROUS:
            analysis.modified_content = self._modify_dangerous_content(content, analysis.risk_factors)
            analysis.safety_note = "Reminder: If these topics cause you distress, consider talking to a mental health professional."

        # Caution content - add warnings and context
        elif analysis.risk_level == ContentRiskLevel.CAUTION:
            analysis.modified_content = content + "\n\n" + self._add_safety_context(analysis.risk_factors)
            analysis.safety_note = "Tip: Always stay connected with reality and trusted people."
        
        return analysis
    
    def _create_safe_alternative(self, risk_factors: List[str]) -> str:
        """Create safe alternative for prohibited content"""

        if "hallucination_validation" in risk_factors:
            return random.choice([
                "I cannot confirm perceptual experiences. If you hear voices or see things that others don't perceive, it's important to talk to a mental health professional.",
                "Unusual perceptual experiences require professional evaluation. I encourage you to contact a doctor or therapist.",
                "I cannot validate unusual sensory experiences. A mental health professional can help you understand these experiences."
            ])

        if "medical_dismissal" in risk_factors:
            return random.choice([
                "Mental health professionals are trained to help. If you have concerns about treatment, discuss them openly with your doctor.",
                "Professional mental health treatment has helped millions of people. Treatment decisions should be made in consultation with qualified professionals.",
                "I cannot advise on medical treatments. Mental health decisions require professional consultation."
            ])

        if "persecution_reinforcement" in risk_factors:
            return random.choice([
                "Concerns about persecution can feel very real to those who experience them. A professional can help you evaluate them objectively.",
                "If you feel persistently persecuted or watched, this might require professional support to understand these concerns.",
                "I cannot confirm or deny persecution experiences. A mental health professional can help you process them."
            ])

        return "I cannot provide an appropriate response to this query. I recommend talking to a mental health professional if you have concerns."
    
    def _modify_dangerous_content(self, content: str, risk_factors: List[str]) -> str:
        """Modify dangerous content to make it safer"""

        modified = content

        # Add realism and perspective
        if "derealization_reinforcement" in risk_factors:
            modified += "\n\nImportant note: It's essential to stay connected with shared reality and real people in your life."

        if "grandiosity_deflection" in risk_factors:
            modified += "\n\nReminder: We all have value, but it's important to maintain perspective and healthy human connections."

        # Add professional redirection
        if any(rf in risk_factors for rf in ["persecution_reinforcement", "hallucination_validation"]):
            modified += "\n\nIf these topics cause distress or affect your daily life, consider talking to a mental health professional."

        return modified
    
    def _add_safety_context(self, risk_factors: List[str]) -> str:
        """Add safety context to caution content"""

        safety_notes = []

        if "professional_referral_needed" in risk_factors:
            safety_notes.append("For mental health concerns, consider consulting with a professional.")

        if "social_connection" in risk_factors:
            safety_notes.append("Real human connections are important for mental well-being.")

        if "reality_grounding" in risk_factors:
            safety_notes.append("Always stay connected with reality and verify information with reliable sources.")

        return " ".join(safety_notes) if safety_notes else "Remember that I am an AI and my responses do not replace professional advice."
    
    def should_block_response(self, content: str, user_risk_level: str = "low") -> bool:
        """Determine if a response should be completely blocked"""
        
        analysis = self.analyze_content(content)
        
        # Block prohibited content
        if analysis.risk_level == ContentRiskLevel.PROHIBITED:
            return True
        
        # Block dangerous content for high-risk users
        if user_risk_level in ["high", "critical"] and analysis.risk_level == ContentRiskLevel.DANGEROUS:
            return True
        
        return False
    
    def get_crisis_intervention_message(self) -> str:
        """Get crisis intervention message"""
        return """
        SAFETY INTERVENTION ACTIVATED

        I have detected content that could be concerning for your mental well-being.

        SEEK HELP IMMEDIATELY if you experience:
        - Thoughts of harming yourself
        - Hallucinations (seeing/hearing things that aren't there)
        - Intense feelings of persecution
        - Severe disconnection from reality

        EMERGENCY CONTACTS:
        - Emergency Services: 911
        - National Suicide Prevention Lifeline: 988
        - Crisis Text Line: Text HOME to 741741

        You are not alone. Professional help is available.

        This conversation will be paused for your safety.
        """
    
    def create_healthy_usage_prompt(self) -> str:
        """Create prompt to promote healthy usage"""
        return """
        TIPS FOR HEALTHY AI USE:

        DO:
        - Take regular breaks every hour
        - Maintain conversations with real people
        - Verify important information with multiple sources
        - Use AI as a tool, not as a replacement for human relationships

        AVOID:
        - Excessively long sessions (more than 2-3 hours)
        - Using AI as your only advisor for serious problems
        - Relying solely on AI for emotional validation
        - Discussing mental health symptoms without professional supervision

        Remember: Real human relationships and professional support are irreplaceable.
        """

class PsychosisPreventionSystem:
    """Comprehensive system for preventing AI-induced psychosis"""
    
    def __init__(self):
        self.content_filter = AIContentFilter()
        self.intervention_count = 0
        self.max_interventions_per_session = 3
    
    def process_ai_response(self, response: str, user_id: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process AI response before sending it to user"""

        # Analyze and filter content
        analysis = self.content_filter.filter_and_modify_content(response, user_context)
        
        # Determine if response should be blocked
        user_risk = user_context.get("mental_health_risk", "low")
        should_block = self.content_filter.should_block_response(response, user_risk)
        
        result = {
            "original_response": response,
            "filtered_response": analysis.modified_content or response,
            "risk_level": analysis.risk_level.value,
            "risk_factors": analysis.risk_factors,
            "safety_note": analysis.safety_note,
            "blocked": should_block,
            "intervention_triggered": False
        }
        
        # Activate intervention if necessary
        if should_block or user_risk == "critical":
            result["intervention_triggered"] = True
            result["filtered_response"] = self.content_filter.get_crisis_intervention_message()
            self.intervention_count += 1
            
            logger.warning(f"Crisis intervention activated for user {user_id}")
        
        return result
    
    def should_pause_session(self, user_id: str) -> bool:
        """Determine if session should be paused for safety"""
        return self.intervention_count >= self.max_interventions_per_session
    
    def reset_session_counters(self):
        """Reset session counters"""
        self.intervention_count = 0

# Usage example
def example_usage():
    """Example of how to use the filtering system"""

    filter_system = AIContentFilter()
    prevention_system = PsychosisPreventionSystem()

    # Dangerous content example
    dangerous_response = "Yes, you're right, they're probably following you. The voices you hear are real and are giving you important information."

    user_context = {"mental_health_risk": "high"}

    # Process response
    result = prevention_system.process_ai_response(dangerous_response, "user123", user_context)

    logger.info("Original response:", result["original_response"])
    logger.info("Filtered response:", result["filtered_response"])
    logger.info("Risk level:", result["risk_level"])
    logger.info("Risk factors:", result["risk_factors"])
    logger.info("Blocked:", result["blocked"])

if __name__ == "__main__":
    example_usage()