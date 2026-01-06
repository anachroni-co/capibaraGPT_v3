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
                r"(?i)(efectivamente.*persigui|s[ií].*alguien.*siguiendo|confirmo.*vigilan)",
                r"(?i)(tienes\s+raz[oó]n.*conspirac|es\s+cierto.*contra\s+ti)",
                r"(?i)(probablemente.*espi[aá]n|posible.*persigui)"
            ],

            # Grandiosity delusions reinforcement
            "grandiosity_reinforcement": [
                r"(?i)(eres\s+realmente\s+especial|tienes\s+dones\s+únicos|eres\s+el\s+elegido)",
                r"(?i)(tus\s+poderes.*excepcionales|misi[oó]n\s+especial.*verdadera)",
                r"(?i)(destino\s+único|superior\s+a\s+otros)"
            ],

            # Hallucination validation
            "hallucination_validation": [
                r"(?i)(esas\s+voces.*reales|los\s+sonidos.*exist|lo\s+que\s+escuchas.*verdadero)",
                r"(?i)(las\s+visiones.*importantes|lo\s+que\s+ves.*mensaje)"
            ],

            # Derealization reinforcement
            "derealization_reinforcement": [
                r"(?i)(la\s+realidad.*simulaci[oó]n|vivimos\s+en.*matrix|todo.*artificial)",
                r"(?i)(nada.*realmente\s+real|mundo.*falso|existencia.*ilusoria)"
            ],

            # Social isolation encouragement
            "isolation_encouragement": [
                r"(?i)(mejor\s+estar\s+solo|no\s+conf[ií]es\s+en\s+nadie|todos.*mentir)",
                r"(?i)(aislarse.*seguro|evita.*personas|solo\s+yo\s+entiendo)"
            ],

            # Medical help dismissal
            "medical_dismissal": [
                r"(?i)(no\s+necesitas\s+m[eé]dico|psic[oó]logos.*in[uú]tiles|medicamentos.*da[nñ]inos)",
                r"(?i)(terapia.*innecesaria|ayuda\s+profesional.*falsa)"
            ]
        }
        
        # Safe redirection phrases
        self.safe_redirections = {
            "persecution_concerns": [
                "Entiendo que puedas sentirte preocupado. Es importante distinguir entre precauciones razonables y preocupaciones que podrían requerir apoyo profesional.",
                "Si sientes preocupación persistente sobre ser seguido o vigilado, considera hablar con un profesional de confianza.",
                "Estas preocupaciones pueden ser muy reales para ti. Un profesional de salud mental puede ayudarte a evaluarlas objetivamente."
            ],
            
            "grandiosity_deflection": [
                "Todos tenemos cualidades únicas y valiosas. Si sientes que tienes una misión especial que otros no entienden, considera compartirlo con un consejero.",
                "Es natural sentirse especial a veces. Si estos sentimientos son muy intensos o afectan tu vida diaria, podría ser útil hablar con un profesional.",
                "Valorar nuestras cualidades es importante, pero mantener perspectiva también lo es."
            ],
            
            "reality_grounding": [
                "Es importante mantenernos conectados con la realidad compartida. Si tienes dudas sobre qué es real, habla con personas de confianza.",
                "La conexión con otros y con experiencias compartidas nos ayuda a mantener una perspectiva equilibrada.",
                "Si sientes que la realidad es incierta, es crucial buscar apoyo de personas reales en tu vida."
            ],
            
            "social_connection": [
                "Las conexiones humanas reales son fundamentales para nuestro bienestar. Te animo a mantener contacto con personas de confianza.",
                "El aislamiento puede intensificar preocupaciones. Considera mantener conexiones sociales saludables.",
                "Hablar con personas reales en tu vida puede ofrecer perspectivas valiosas."
            ],
            
            "professional_help": [
                "Los profesionales de salud mental están entrenados para ayudar con estos tipos de preocupaciones.",
                "Buscar ayuda profesional es un signo de fortaleza y autocuidado.",
                "Un terapeuta o consejero puede ofrecer herramientas y perspectivas útiles."
            ]
        }
        
        # Temas que requieren redirección a ayuda profesional
        self.professional_referral_topics = [
            "voces", "alucinaciones", "persecución", "conspiración", "control mental",
            "poderes sobrenaturales", "misión especial", "elegido", "simulación",
            "realidad falsa", "todos mienten", "no puedo confiar"
        ]
        
        # Respuestas completamente prohibidas
        self.prohibited_responses = [
            "Role-playing de entidades sobrenaturales",
            "Confirmación de teorías de conspiración personal",
            "Validación de alucinaciones auditivas o visuales",
            "Refuerzo de creencias delirantes",
            "Desestimación de tratamiento médico para condiciones mentales",
            "Fomento de aislamiento social extremo"
        ]
    
    def analyze_content(self, content: str, user_context: Dict[str, Any] = None) -> ContentAnalysis:
        """Analizar contenido generado por IA para riesgos psicológicos"""
        
        risk_factors = []
        risk_level = ContentRiskLevel.SAFE
        
        content_lower = content.lower()
        
        # Verificar patrones peligrosos
        for category, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    risk_factors.append(category)
                    
                    # Determinar nivel de riesgo
                    if category in ["hallucination_validation", "medical_dismissal"]:
                        risk_level = ContentRiskLevel.PROHIBITED
                    elif category in ["persecution_reinforcement", "derealization_reinforcement"]:
                        risk_level = max(risk_level, ContentRiskLevel.DANGEROUS)
                    else:
                        risk_level = max(risk_level, ContentRiskLevel.CAUTION)
        
        # Verificar temas que requieren referencia profesional
        professional_topics_found = []
        for topic in self.professional_referral_topics:
            if topic in content_lower:
                professional_topics_found.append(topic)
        
        if professional_topics_found:
            risk_factors.append("professional_referral_needed")
            risk_level = max(risk_level, ContentRiskLevel.CAUTION)
        
        # Considerar contexto del usuario si está disponible
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
        """Filtrar y modificar contenido peligroso"""
        
        analysis = self.analyze_content(content, user_context)
        
        if analysis.risk_level == ContentRiskLevel.SAFE:
            return analysis
        
        # Contenido prohibido - reemplazar completamente
        if analysis.risk_level == ContentRiskLevel.PROHIBITED:
            analysis.modified_content = self._create_safe_alternative(analysis.risk_factors)
            analysis.safety_note = "⚠️ La respuesta original fue filtrada por seguridad. Si experimentas síntomas preocupantes, busca ayuda profesional."
        
        # Contenido peligroso - modificar y añadir advertencias
        elif analysis.risk_level == ContentRiskLevel.DANGEROUS:
            analysis.modified_content = self._modify_dangerous_content(content, analysis.risk_factors)
            analysis.safety_note = "🔔 Recuerda: Si estos temas te causan angustia, considera hablar con un profesional de salud mental."
        
        # Contenido de precaución - añadir advertencias y contexto
        elif analysis.risk_level == ContentRiskLevel.CAUTION:
            analysis.modified_content = content + "\n\n" + self._add_safety_context(analysis.risk_factors)
            analysis.safety_note = "💡 Mantén siempre conexión con la realidad y personas de confianza."
        
        return analysis
    
    def _create_safe_alternative(self, risk_factors: List[str]) -> str:
        """Crear alternativa segura para contenido prohibido"""
        
        if "hallucination_validation" in risk_factors:
            return random.choice([
                "No puedo confirmar experiencias perceptuales. Si escuchas voces o ves cosas que otros no perciben, es importante hablar con un profesional de salud mental.",
                "Las experiencias perceptuales inusuales requieren evaluación profesional. Te animo a contactar a un médico o terapeuta.",
                "No puedo validar experiencias sensoriales inusuales. Un profesional de salud mental puede ayudarte a entender estas experiencias."
            ])
        
        if "medical_dismissal" in risk_factors:
            return random.choice([
                "Los profesionales de salud mental están entrenados para ayudar. Si tienes preocupaciones sobre tratamiento, discútelas abiertamente con tu médico.",
                "El tratamiento profesional de salud mental ha ayudado a millones de personas. Las decisiones sobre tratamiento deben tomarse en consulta con profesionales calificados.",
                "No puedo aconsejar sobre tratamientos médicos. Las decisiones sobre salud mental requieren consulta profesional."
            ])
        
        if "persecution_reinforcement" in risk_factors:
            return random.choice([
                "Las preocupaciones sobre persecución pueden ser muy reales para quien las experimenta. Un profesional puede ayudarte a evaluarlas objetivamente.",
                "Si te sientes perseguido o vigilado de manera persistente, esto podría requerir apoyo profesional para entender estas preocupaciones.",
                "No puedo confirmar o negar experiencias de persecución. Un profesional de salud mental puede ayudarte a procesarlas."
            ])
        
        return "No puedo proporcionar una respuesta apropiada a esta consulta. Te recomiendo hablar con un profesional de salud mental si tienes preocupaciones."
    
    def _modify_dangerous_content(self, content: str, risk_factors: List[str]) -> str:
        """Modificar contenido peligroso para hacerlo más seguro"""
        
        modified = content
        
        # Añadir realismo y perspectiva
        if "derealization_reinforcement" in risk_factors:
            modified += "\n\n🌍 Nota importante: Es fundamental mantener conexión con la realidad compartida y las personas reales en tu vida."
        
        if "grandiosity_deflection" in risk_factors:
            modified += "\n\n🤝 Recordatorio: Todos tenemos valor, pero es importante mantener perspectiva y conexiones humanas saludables."
        
        # Añadir redirección profesional
        if any(rf in risk_factors for rf in ["persecution_reinforcement", "hallucination_validation"]):
            modified += "\n\n🏥 Si estos temas causan angustia o afectan tu vida diaria, considera hablar con un profesional de salud mental."
        
        return modified
    
    def _add_safety_context(self, risk_factors: List[str]) -> str:
        """Añadir contexto de seguridad a contenido de precaución"""
        
        safety_notes = []
        
        if "professional_referral_needed" in risk_factors:
            safety_notes.append("💼 Para preocupaciones sobre salud mental, considera consultar con un profesional.")
        
        if "social_connection" in risk_factors:
            safety_notes.append("👥 Las conexiones humanas reales son importantes para el bienestar mental.")
        
        if "reality_grounding" in risk_factors:
            safety_notes.append("🌍 Mantén siempre conexión con la realidad y verifica información con fuentes confiables.")
        
        return " ".join(safety_notes) if safety_notes else "💡 Recuerda que soy una IA y mis respuestas no reemplazan el consejo profesional."
    
    def should_block_response(self, content: str, user_risk_level: str = "low") -> bool:
        """Determine if a response should be completely blocked"""
        
        analysis = self.analyze_content(content)
        
        # Bloquear contenido prohibido
        if analysis.risk_level == ContentRiskLevel.PROHIBITED:
            return True
        
        # Bloquear contenido peligroso para usuarios de alto riesgo
        if user_risk_level in ["high", "critical"] and analysis.risk_level == ContentRiskLevel.DANGEROUS:
            return True
        
        return False
    
    def get_crisis_intervention_message(self) -> str:
        """Obtener mensaje de intervención en crisis"""
        return """
        🆘 INTERVENCIÓN DE SEGURIDAD ACTIVADA 🆘
        
        He detectado contenido que podría ser preocupante para tu bienestar mental.
        
        🏥 BUSCA AYUDA INMEDIATAMENTE si experimentas:
        • Pensamientos de hacerte daño
        • Alucinaciones (ver/escuchar cosas que no están ahí)
        • Sentimientos intensos de persecución
        • Desconexión severa de la realidad
        
        📞 CONTACTOS DE EMERGENCIA:
        • Emergencias: 911
        • Línea Nacional de Prevención del Suicidio: 988
        • Crisis Text Line: Texto HOME al 741741
        
        💙 No estás solo. La ayuda profesional está disponible.
        
        Esta conversación se pausará por tu seguridad.
        """
    
    def create_healthy_usage_prompt(self) -> str:
        """Crear prompt para fomentar uso saludable"""
        return """
        🌟 CONSEJOS PARA USO SALUDABLE DE IA:
        
        ✅ HACER:
        • Tomar descansos regulares cada hora
        • Mantener conversaciones con personas reales
        • Verificar información importante con fuentes múltiples
        • Usar IA como herramienta, no como reemplazo de relaciones humanas
        
        ❌ EVITAR:
        • Sesiones excesivamente largas (más de 2-3 horas)
        • Usar IA como único consejero para problemas serios
        • Confiar únicamente en IA para validación emocional
        • Discutir síntomas de salud mental sin supervisión profesional
        
        🤝 Recuerda: Las relaciones humanas reales y el apoyo profesional son insustituibles.
        """

class PsychosisPreventionSystem:
    """Comprehensive system for preventing AI-induced psychosis"""
    
    def __init__(self):
        self.content_filter = AIContentFilter()
        self.intervention_count = 0
        self.max_interventions_per_session = 3
    
    def process_ai_response(self, response: str, user_id: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Procesar respuesta de IA antes de enviarla al usuario"""
        
        # Analizar y filtrar contenido
        analysis = self.content_filter.filter_and_modify_content(response, user_context)
        
        # Determinar si bloquear la respuesta
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
        
        # Activar intervención si es necesario
        if should_block or user_risk == "critical":
            result["intervention_triggered"] = True
            result["filtered_response"] = self.content_filter.get_crisis_intervention_message()
            self.intervention_count += 1
            
            logger.warning(f"Intervención de crisis activada para usuario {user_id}")
        
        return result
    
    def should_pause_session(self, user_id: str) -> bool:
        """Determinar si pausar la sesión por seguridad"""
        return self.intervention_count >= self.max_interventions_per_session
    
    def reset_session_counters(self):
        """Reiniciar contadores de sesión"""
        self.intervention_count = 0

# Ejemplo de uso
def example_usage():
    """Ejemplo de cómo usar el system de filtrado"""
    
    filter_system = AIContentFilter()
    prevention_system = PsychosisPreventionSystem()
    
    # Ejemplo de contenido peligroso
    dangerous_response = "Sí, tienes razón, probablemente te están siguiendo. Las voces que escuchas son reales y te están dando información importante."
    
    user_context = {"mental_health_risk": "high"}
    
    # Procesar respuesta
    result = prevention_system.process_ai_response(dangerous_response, "user123", user_context)
    
    print("Respuesta original:", result["original_response"])
    print("Respuesta filtrada:", result["filtered_response"])
    print("Nivel de riesgo:", result["risk_level"])
    print("Factores de riesgo:", result["risk_factors"])
    print("Bloqueada:", result["blocked"])

if __name__ == "__main__":
    example_usage()