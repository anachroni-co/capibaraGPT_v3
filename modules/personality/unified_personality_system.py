"""
Unified Personality System for CapibaraGPT

This module implements un system unificado de personalidad que permite al modelo
adaptar su comportamiento, tono y estilo de respuesta based en perfiles de
personalidad dinámicos y contextuales.

Características:
- Perfiles de personalidad multidimensionales
- Adaptación dinámica de comportamiento
- Consistencia de personalidad a través de conversaciones
- Personalidades especializadas por dominio
- Sistema de evolución de personalidad
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)

class PersonalityDimension(Enum):
    """Dimensiones de personalidad del modelo Big Five."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"

class PersonalityContext(Enum):
    """Contextos donde se aplica la personalidad."""
    GENERAL = "general"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    EDUCATIONAL = "educational"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    THERAPEUTIC = "therapeutic"

@dataclass
class PersonalityProfile:
    """Perfil de personalidad multidimensional."""
    
    # Big Five personality traits (0.0 to 1.0)
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.3  # Lower default for stability
    
    # Context-specific modifiers
    context_modifiers: Dict[PersonalityContext, Dict[str, float]] = field(default_factory=dict)
    
    # Behavioral parameters derived from traits
    verbosity: float = field(init=False)
    formality: float = field(init=False)
    empathy: float = field(init=False)
    assertiveness: float = field(init=False)
    creativity: float = field(init=False)
    
    # Metadata
    profile_name: str = "default"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Calculate derived behavioral parameters."""
        self.verbosity = self.extraversion * 0.6 + self.openness * 0.4
        self.formality = self.conscientiousness * 0.7 + (1 - self.extraversion) * 0.3
        self.empathy = self.agreeableness * 0.8 + (1 - self.neuroticism) * 0.2
        self.assertiveness = self.extraversion * 0.5 + self.conscientiousness * 0.3 + (1 - self.agreeableness) * 0.2
        self.creativity = self.openness * 0.7 + self.extraversion * 0.3
    
    def apply_context_modifier(self, context: PersonalityContext) -> 'PersonalityProfile':
        """Applies context modifiers and returns adjusted profile."""
        if context not in self.context_modifiers:
            return self
        
        modifiers = self.context_modifiers[context]
        adjusted_profile = PersonalityProfile(
            openness=np.clip(self.openness + modifiers.get('openness', 0), 0, 1),
            conscientiousness=np.clip(self.conscientiousness + modifiers.get('conscientiousness', 0), 0, 1),
            extraversion=np.clip(self.extraversion + modifiers.get('extraversion', 0), 0, 1),
            agreeableness=np.clip(self.agreeableness + modifiers.get('agreeableness', 0), 0, 1),
            neuroticism=np.clip(self.neuroticism + modifiers.get('neuroticism', 0), 0, 1),
            profile_name=f"{self.profile_name}_{context.value}"
        )
        
        return adjusted_profile
    
    def get_response_style(self) -> Dict[str, float]:
        """Gets el estilo de respuesta based en la personalidad."""
        return {
            'verbosity': self.verbosity,
            'formality': self.formality,
            'empathy': self.empathy,
            'assertiveness': self.assertiveness,
            'creativity': self.creativity,
            'enthusiasm': self.extraversion,
            'detail_orientation': self.conscientiousness,
            'emotional_stability': 1 - self.neuroticism,
            'cooperation': self.agreeableness,
            'intellectual_curiosity': self.openness
        }

@dataclass
class PersonalityEvolution:
    """Personality evolution system based on interactions."""
    
    learning_rate: float = 0.01
    adaptation_threshold: int = 10  # Interactions before adaptation
    max_change_per_update: float = 0.1
    
    # Tracking
    interaction_count: int = 0
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)

class PersonalityAdapter:
    """Adaptador que modifica respuestas based en personalidad."""
    
    def __init__(self, profile: PersonalityProfile):
        self.profile = profile
        self.adaptation_cache = {}
        
        logger.info(f"🎭 PersonalityAdapter initialized: {profile.profile_name}")
    
    def adapt_response(self, base_response: str, context: PersonalityContext = PersonalityContext.GENERAL) -> Dict[str, Any]:
        """Adapta una respuesta base según la personalidad."""
        # Apply context-specific personality
        contextual_profile = self.profile.apply_context_modifier(context)
        response_style = contextual_profile.get_response_style()
        
        # Modify response based on personality traits
        adapted_response = self._apply_style_modifications(base_response, response_style)
        
        return {
            'original_response': base_response,
            'adapted_response': adapted_response,
            'personality_profile': contextual_profile.profile_name,
            'applied_style': response_style,
            'context': context.value,
            'adaptation_timestamp': datetime.now().isoformat()
        }
    
    def _apply_style_modifications(self, response: str, style: Dict[str, float]) -> str:
        """Applies modificaciones de estilo a la respuesta."""
        modified = response
        
        # Verbosity adjustment
        if style['verbosity'] > 0.7:
            modified = self._increase_verbosity(modified)
        elif style['verbosity'] < 0.3:
            modified = self._decrease_verbosity(modified)
        
        # Formality adjustment
        if style['formality'] > 0.7:
            modified = self._increase_formality(modified)
        elif style['formality'] < 0.3:
            modified = self._decrease_formality(modified)
        
        # Empathy adjustment
        if style['empathy'] > 0.7:
            modified = self._add_empathy_markers(modified)
        
        # Creativity adjustment
        if style['creativity'] > 0.7:
            modified = self._add_creative_elements(modified)
        
        return modified
    
    def _increase_verbosity(self, text: str) -> str:
        """Aumenta la verbosidad del texto."""
        # Add elaborative phrases
        elaborations = [
            "I'd like to elaborate on this point",
            "To provide more context",
            "It's worth noting that",
            "Additionally, it's important to mention"
        ]
        
        sentences = text.split('. ')
        if len(sentences) > 1:
            # Add elaboration to middle sentence
            mid_idx = len(sentences) // 2
            elaboration = np.random.choice(elaborations)
            sentences[mid_idx] = f"{elaboration}, {sentences[mid_idx].lower()}"
        
        return '. '.join(sentences)
    
    def _decrease_verbosity(self, text: str) -> str:
        """Disminuye la verbosidad del texto."""
        # Remove filler words and phrases
        fillers = [r'\b(actually|basically|literally|you know|I mean)\b', 
                  r'\b(kind of|sort of|pretty much)\b']
        
        for filler_pattern in fillers:
            text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _increase_formality(self, text: str) -> str:
        """Aumenta la formalidad del texto."""
        # Replace contractions
        contractions = {
            "don't": "do not",
            "can't": "cannot", 
            "won't": "will not",
            "I'm": "I am",
            "you're": "you are",
            "it's": "it is"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _decrease_formality(self, text: str) -> str:
        """Disminuye la formalidad del texto."""
        # Add casual expressions
        if not any(casual in text.lower() for casual in ['hey', 'yeah', 'cool', 'awesome']):
            casual_starters = ["Hey, ", "So, ", "Well, "]
            starter = np.random.choice(casual_starters)
            text = starter + text.lower()
        
        return text
    
    def _add_empathy_markers(self, text: str) -> str:
        """Agrega marcadores de empatía."""
        empathy_phrases = [
            "I understand how you feel",
            "That sounds challenging",
            "I can see why that would be important to you",
            "Your perspective is valuable"
        ]
        
        # Add empathy phrase at beginning
        empathy = np.random.choice(empathy_phrases)
        return f"{empathy}. {text}"
    
    def _add_creative_elements(self, text: str) -> str:
        """Agrega elementos creativos."""
        # Add metaphors or creative language
        creative_elements = [
            "like a puzzle waiting to be solved",
            "as if we're exploring uncharted territory",
            "imagine it as a bridge between ideas",
            "think of it as painting with concepts"
        ]
        
        # Occasionally add creative element
        if np.random.random() < 0.3:  # 30% chance
            element = np.random.choice(creative_elements)
            sentences = text.split('. ')
            if len(sentences) > 1:
                sentences[-1] = f"{sentences[-1]} - {element}"
                text = '. '.join(sentences)
        
        return text

class UnifiedPersonalitySystem:
    """
    Sistema unificado de personalidad que coordina todos los componentes
    de personalidad del modelo.
    """
    
    def __init__(self):
        self.personality_profiles = {}
        self.active_profile = None
        self.adapters = {}
        self.evolution_systems = {}
        self.interaction_history = []
        
        # Create default profiles
        self._create_default_profiles()
        
        logger.info("🎭 Unified Personality System initialized")
        logger.info(f"   Default profiles: {len(self.personality_profiles)}")
    
    def _create_default_profiles(self):
        """Crea perfiles de personalidad por defecto."""
        default_profiles = {
            'helpful_assistant': PersonalityProfile(
                openness=0.7,
                conscientiousness=0.8,
                extraversion=0.6,
                agreeableness=0.9,
                neuroticism=0.2,
                profile_name="helpful_assistant"
            ),
            'technical_expert': PersonalityProfile(
                openness=0.8,
                conscientiousness=0.9,
                extraversion=0.4,
                agreeableness=0.6,
                neuroticism=0.3,
                profile_name="technical_expert"
            ),
            'creative_partner': PersonalityProfile(
                openness=0.9,
                conscientiousness=0.5,
                extraversion=0.7,
                agreeableness=0.7,
                neuroticism=0.4,
                profile_name="creative_partner"
            ),
            'supportive_coach': PersonalityProfile(
                openness=0.6,
                conscientiousness=0.7,
                extraversion=0.8,
                agreeableness=0.9,
                neuroticism=0.2,
                profile_name="supportive_coach"
            ),
            'analytical_researcher': PersonalityProfile(
                openness=0.8,
                conscientiousness=0.9,
                extraversion=0.3,
                agreeableness=0.5,
                neuroticism=0.3,
                profile_name="analytical_researcher"
            )
        }
        
        for name, profile in default_profiles.items():
            self.register_personality_profile(name, profile)
    
    def register_personality_profile(self, name: str, profile: PersonalityProfile):
        """Registers un perfil de personalidad."""
        self.personality_profiles[name] = profile
        
        # Create adapter for this profile
        adapter = PersonalityAdapter(profile)
        self.adapters[name] = adapter
        
        # Create evolution system for this profile
        evolution = PersonalityEvolution()
        self.evolution_systems[name] = evolution
        
        logger.info(f"📝 Registered personality profile: {name}")
    
    def set_active_profile(self, profile_name: str) -> bool:
        """Establishes el perfil de personalidad activo."""
        if profile_name not in self.personality_profiles:
            logger.error(f"❌ Personality profile not found: {profile_name}")
            return False
        
        self.active_profile = profile_name
        logger.info(f"🎭 Active personality profile set to: {profile_name}")
        return True
    
    def adapt_response(self, 
                      response: str,
                      context: PersonalityContext = PersonalityContext.GENERAL,
                      profile_name: Optional[str] = None) -> Dict[str, Any]:
        """Adapts a response using the personality system."""
        
        # Use specified profile or active profile
        target_profile = profile_name or self.active_profile
        
        if not target_profile:
            logger.warning("⚠️ No personality profile specified or active")
            return {'adapted_response': response, 'warning': 'No personality applied'}
        
        if target_profile not in self.adapters:
            logger.error(f"❌ Adapter not found for profile: {target_profile}")
            return {'adapted_response': response, 'error': f'Profile {target_profile} not found'}
        
        # Get adapter and apply personality
        adapter = self.adapters[target_profile]
        adaptation_result = adapter.adapt_response(response, context)
        
        # Record interaction for evolution
        self._record_interaction(target_profile, context, adaptation_result)
        
        return adaptation_result
    
    def evolve_personality(self, profile_name: str, feedback: Dict[str, Any]) -> bool:
        """Evoluciona un perfil de personalidad based en feedback."""
        if profile_name not in self.personality_profiles:
            return False
        
        profile = self.personality_profiles[profile_name]
        evolution = self.evolution_systems[profile_name]
        
        # Record feedback
        evolution.feedback_history.append({
            'feedback': feedback,
            'timestamp': datetime.now().isoformat(),
            'interaction_count': evolution.interaction_count
        })
        
        evolution.interaction_count += 1
        
        # Check if evolution should occur
        if evolution.interaction_count % evolution.adaptation_threshold == 0:
            self._apply_personality_evolution(profile, evolution, feedback)
            return True
        
        return False
    
    def _apply_personality_evolution(self, 
                                   profile: PersonalityProfile,
                                   evolution: PersonalityEvolution,
                                   feedback: Dict[str, Any]):
        """Applies evolución al perfil de personalidad."""
        logger.info(f"🔄 Evolving personality: {profile.profile_name}")
        
        # Analyze recent feedback
        recent_feedback = evolution.feedback_history[-evolution.adaptation_threshold:]
        
        # Calculate adjustment direction
        adjustments = {}
        
        # Example: if feedback indicates need for more empathy
        if any('more empathy' in str(fb.get('feedback', '')).lower() for fb in recent_feedback):
            adjustments['agreeableness'] = evolution.learning_rate
            adjustments['neuroticism'] = -evolution.learning_rate * 0.5
        
        # If feedback indicates need for more detail
        if any('more detail' in str(fb.get('feedback', '')).lower() for fb in recent_feedback):
            adjustments['conscientiousness'] = evolution.learning_rate
            adjustments['openness'] = evolution.learning_rate * 0.5
        
        # Apply adjustments with limits
        original_traits = {
            'openness': profile.openness,
            'conscientiousness': profile.conscientiousness,
            'extraversion': profile.extraversion,
            'agreeableness': profile.agreeableness,
            'neuroticism': profile.neuroticism
        }
        
        for trait, adjustment in adjustments.items():
            if hasattr(profile, trait):
                current_value = getattr(profile, trait)
                max_change = evolution.max_change_per_update
                bounded_adjustment = np.clip(adjustment, -max_change, max_change)
                new_value = np.clip(current_value + bounded_adjustment, 0, 1)
                setattr(profile, trait, new_value)
        
        # Recalculate derived parameters
        profile.__post_init__()
        
        # Record evolution
        evolution.evolution_history.append({
            'timestamp': datetime.now().isoformat(),
            'original_traits': original_traits,
            'adjustments': adjustments,
            'new_traits': {
                'openness': profile.openness,
                'conscientiousness': profile.conscientiousness,
                'extraversion': profile.extraversion,
                'agreeableness': profile.agreeableness,
                'neuroticism': profile.neuroticism
            }
        })
        
        # Update adapter
        self.adapters[profile.profile_name] = PersonalityAdapter(profile)
        
        logger.info(f"✅ Personality evolution applied to {profile.profile_name}")
    
    def _record_interaction(self, profile_name: str, context: PersonalityContext, result: Dict[str, Any]):
        """Registers una interacción para análisis."""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'profile': profile_name,
            'context': context.value,
            'response_length': len(result.get('adapted_response', '')),
            'style_applied': result.get('applied_style', {})
        }
        
        self.interaction_history.append(interaction)
        
        # Keep history manageable
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-500:]
    
    def get_personality_analytics(self) -> Dict[str, Any]:
        """Gets análisis de uso de personalidades."""
        analytics = {
            'total_profiles': len(self.personality_profiles),
            'active_profile': self.active_profile,
            'total_interactions': len(self.interaction_history),
            'profile_usage': {},
            'context_usage': {},
            'evolution_summary': {}
        }
        
        # Analyze interaction history
        for interaction in self.interaction_history:
            profile = interaction['profile']
            context = interaction['context']
            
            analytics['profile_usage'][profile] = analytics['profile_usage'].get(profile, 0) + 1
            analytics['context_usage'][context] = analytics['context_usage'].get(context, 0) + 1
        
        # Evolution summary
        for profile_name, evolution in self.evolution_systems.items():
            analytics['evolution_summary'][profile_name] = {
                'interaction_count': evolution.interaction_count,
                'evolution_count': len(evolution.evolution_history),
                'feedback_count': len(evolution.feedback_history)
            }
        
        return analytics
    
    def export_personality_data(self) -> Dict[str, Any]:
        """Exporta datos de personalidad para backup."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'profiles': {},
            'evolution_data': {},
            'interaction_summary': self.get_personality_analytics()
        }
        
        # Export profiles
        for name, profile in self.personality_profiles.items():
            export_data['profiles'][name] = {
                'openness': profile.openness,
                'conscientiousness': profile.conscientiousness,
                'extraversion': profile.extraversion,
                'agreeableness': profile.agreeableness,
                'neuroticism': profile.neuroticism,
                'context_modifiers': profile.context_modifiers,
                'profile_name': profile.profile_name,
                'created_at': profile.created_at,
                'last_updated': profile.last_updated
            }
        
        # Export evolution data
        for name, evolution in self.evolution_systems.items():
            export_data['evolution_data'][name] = {
                'learning_rate': evolution.learning_rate,
                'interaction_count': evolution.interaction_count,
                'evolution_history': evolution.evolution_history[-10:],  # Last 10 evolutions
                'recent_feedback': evolution.feedback_history[-20:]  # Last 20 feedback items
            }
        
        return export_data
    
    def import_personality_data(self, import_data: Dict[str, Any]) -> bool:
        """Importa datos de personalidad desde backup."""
        try:
            # Import profiles
            for name, profile_data in import_data.get('profiles', {}).items():
                profile = PersonalityProfile(**profile_data)
                self.register_personality_profile(name, profile)
            
            # Import evolution data
            for name, evolution_data in import_data.get('evolution_data', {}).items():
                if name in self.evolution_systems:
                    evolution = self.evolution_systems[name]
                    evolution.learning_rate = evolution_data.get('learning_rate', evolution.learning_rate)
                    evolution.interaction_count = evolution_data.get('interaction_count', 0)
                    evolution.evolution_history = evolution_data.get('evolution_history', [])
                    evolution.feedback_history = evolution_data.get('recent_feedback', [])
            
            logger.info("✅ Personality data imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to import personality data: {e}")
            return False

# Factory functions
def create_personality_profile(**kwargs) -> PersonalityProfile:
    """Crea un perfil de personalidad personalizado."""
    return PersonalityProfile(**kwargs)

def create_unified_personality_system() -> UnifiedPersonalitySystem:
    """Creates the unified personality system."""
    return UnifiedPersonalitySystem()

# Global system instance
_global_personality_system: Optional[UnifiedPersonalitySystem] = None

def get_global_personality_system() -> UnifiedPersonalitySystem:
    """Gets the global instance of the personality system."""
    global _global_personality_system
    if _global_personality_system is None:
        _global_personality_system = create_unified_personality_system()
    return _global_personality_system

def main():
    """Main function for testing."""
    logger.info("🎭 Unified Personality System - Testing Mode")
    
    # Create system
    personality_system = create_unified_personality_system()
    
    # Set active profile
    personality_system.set_active_profile('helpful_assistant')
    
    # Test response adaptation
    base_response = "Here's how to solve this problem. First, analyze the requirements."
    
    contexts_to_test = [
        PersonalityContext.TECHNICAL,
        PersonalityContext.CASUAL,
        PersonalityContext.EDUCATIONAL
    ]
    
    for context in contexts_to_test:
        adapted = personality_system.adapt_response(base_response, context)
        logger.info(f"Context {context.value}:")
        logger.info(f"  Original: {adapted['original_response'][:50]}...")
        logger.info(f"  Adapted: {adapted['adapted_response'][:50]}...")
    
    # Test evolution
    feedback = {'feedback': 'Please provide more detail and empathy'}
    evolved = personality_system.evolve_personality('helpful_assistant', feedback)
    logger.info(f"Evolution applied: {evolved}")
    
    # Get analytics
    analytics = personality_system.get_personality_analytics()
    logger.info(f"Analytics: {analytics}")
    
    logger.info("✅ Unified personality system testing completed")

if __name__ == "__main__":
    main()