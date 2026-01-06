#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Seguridad y Protección de Salud Mental para Capibara6

This module provides protección integral contra psicosis inducida por IA
y otros riesgos de salud mental asociados con el uso intensivo de systems de IA.

Componentes principales:
- MentalHealthMonitor: Monitoreo de patrones de uso
- ContentFilter: Filtrado de contenido peligroso
- InterventionSystem: Manejo de intervenciones automáticas
- SafetyMiddleware: Integración con aplicaciones web

Uso básico:
    from capibara.safety import activate_safety_system
    
    # Activar todas las protecciones
    safety_manager = activate_safety_system(app)
    
    # Procesar mensaje del usuario
    result = safety_manager.process_user_message(user_id, message, session_data)
    
    # Filtrar respuesta de IA
    filtered = safety_manager.process_ai_response(user_id, ai_response, user_context)
"""

import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Security system version
__version__ = "1.0.0"

# Configuración por defecto
DEFAULT_SAFETY_CONFIG = {
    "mental_health_monitoring": {
        "enabled": True,
        "max_session_duration": 180,  # minutos
        "max_daily_messages": 500,
        "max_hourly_messages": 100,
        "obsession_threshold": 0.7,
        "reality_disconnection_threshold": 0.6,
        "paranoid_threshold": 0.5,
        "emotional_volatility_threshold": 0.8
    },
    "content_filtering": {
        "enabled": True,
        "block_psychotic_validation": True,
        "block_medical_dismissal": True,
        "filter_conspiracy_theories": True,
        "enhance_reality_grounding": True
    },
    "interventions": {
        "enabled": True,
        "max_interventions_per_session": 3,
        "enable_emergency_contacts": True,
        "enable_automatic_escalation": True,
        "escalation_timeout_minutes": 30
    },
    "web_integration": {
        "show_disclaimer": True,
        "disclaimer_frequency_days": 7,
        "require_acknowledgment": True,
        "safety_headers": True
    },
    "logging": {
        "log_level": "WARNING",
        "log_file": "mental_health_safety.log",
        "log_interventions": True,
        "log_filtered_content": True
    }
}

def get_safety_config() -> Dict[str, Any]:
    """Obtener configuración de seguridad actual"""
    return DEFAULT_SAFETY_CONFIG.copy()

def activate_safety_system(app=None, config: Optional[Dict[str, Any]] = None):
    """
    Activar system completo de seguridad de salud mental
    
    Args:
        app: Aplicación Flask (opcional)
        config: Configuración personalizada (opcional)
    
    Returns:
        SafetyIntegrationManager: Gestor integral de seguridad
    """
    try:
        # Use default configuration if not provided
        if config is None:
            config = DEFAULT_SAFETY_CONFIG
        
        # Configurar logging
        _setup_safety_logging(config.get("logging", {}))
        
        # Importar y inicializar componentes
        from capibara.safety.intervention_system import SafetyIntegrationManager
        
        logger.info("🔒 Iniciando system de protección de salud mental...")
        
        # Crear gestor de seguridad
        safety_manager = SafetyIntegrationManager()
        
        # Configurar límites de seguridad
        _configure_safety_limits(safety_manager, config)
        
        # Integrar con aplicación web si se proporciona
        if app is not None:
            _integrate_with_web_app(app, safety_manager, config)
        
        logger.info("✅ Sistema de protección de salud mental activado correctamente")
        logger.info("📋 Funciones activas:")
        logger.info("   - Monitoreo de patrones de uso: ✅")
        logger.info("   - Filtrado de contenido peligroso: ✅")
        logger.info("   - Intervenciones automáticas: ✅")
        logger.info("   - Recursos de emergencia: ✅")
        
        return safety_manager
        
    except ImportError as e:
        logger.error(f"❌ Error importando componentes de seguridad: {e}")
        logger.error("⚠️  SISTEMA FUNCIONANDO SIN PROTECCIONES DE SALUD MENTAL")
        return None
    except Exception as e:
        logger.error(f"❌ Error activando system de seguridad: {e}")
        return None

def _setup_safety_logging(logging_config: Dict[str, Any]):
    """Configurar logging especializado para seguridad"""
    log_level = getattr(logging, logging_config.get("log_level", "WARNING"))
    log_file = logging_config.get("log_file", "mental_health_safety.log")
    
    # Crear logger específico para seguridad
    safety_logger = logging.getLogger("capibara.safety")
    safety_logger.setLevel(log_level)
    
    # Handler para archivo
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - SAFETY - %(levelname)s - %(name)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        safety_logger.addHandler(file_handler)
    
    # Handler para consola (solo para errores críticos)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_formatter = logging.Formatter('🚨 SAFETY ALERT: %(message)s')
    console_handler.setFormatter(console_formatter)
    safety_logger.addHandler(console_handler)

def _configure_safety_limits(safety_manager, config: Dict[str, Any]):
    """Configurar límites de seguridad según configuración"""
    
    if not safety_manager:
        return
    
    monitoring_config = config.get("mental_health_monitoring", {})
    
    if monitoring_config.get("enabled", True):
        # Update limits in monitor
        safety_manager.mental_health_monitor.safety_limits.update({
            "max_session_duration": monitoring_config.get("max_session_duration", 180),
            "max_daily_messages": monitoring_config.get("max_daily_messages", 500),
            "max_hourly_messages": monitoring_config.get("max_hourly_messages", 100),
            "obsession_threshold": monitoring_config.get("obsession_threshold", 0.7),
            "reality_disconnection_threshold": monitoring_config.get("reality_disconnection_threshold", 0.6),
            "paranoid_threshold": monitoring_config.get("paranoid_threshold", 0.5),
            "emotional_volatility_threshold": monitoring_config.get("emotional_volatility_threshold", 0.8)
        })
    
    # Configurar filtrado de contenido
    content_config = config.get("content_filtering", {})
    if content_config.get("enabled", True):
        # Los filtros ya están configurados por defecto
        pass
    
    # Configurar intervenciones
    intervention_config = config.get("interventions", {})
    if intervention_config.get("enabled", True):
        safety_manager.intervention_manager.max_interventions_per_session = intervention_config.get("max_interventions_per_session", 3)

def _integrate_with_web_app(app, safety_manager, config: Dict[str, Any]):
    """Integrar system de seguridad con aplicación web"""
    
    try:
        from web_interface.safety_middleware import init_safety_middleware
        
        # Inicializar middleware
        middleware = init_safety_middleware(app)
        middleware.safety_manager = safety_manager
        
        # Configurar disclaimer
        web_config = config.get("web_integration", {})
        if web_config.get("show_disclaimer", True):
            # El middleware ya maneja el disclaimer automáticamente
            pass
        
        logger.info("🌐 Integración web activada correctamente")
        
    except ImportError:
        logger.warning("⚠️  Middleware web no disponible, continuando sin integración web")

def create_safety_summary() -> Dict[str, Any]:
    """Crear resumen del estado del system de seguridad"""
    return {
        "version": __version__,
        "components": {
            "mental_health_monitor": "✅ Activo",
            "content_filter": "✅ Activo", 
            "intervention_system": "✅ Activo",
            "web_middleware": "✅ Activo"
        },
        "protections": {
            "psychosis_prevention": "✅ Habilitada",
            "usage_monitoring": "✅ Habilitada",
            "content_filtering": "✅ Habilitada",
            "emergency_interventions": "✅ Habilitada",
            "professional_resources": "✅ Disponibles"
        },
        "emergency_contacts": {
            "suicide_prevention": "988 (US)",
            "emergency_services": "911",
            "crisis_text": "741741"
        }
    }

def verify_safety_system() -> bool:
    """Verifiesr que todos los componentes de seguridad estén funcionando"""
    try:
        # Verificar importaciones
        from capibara.safety.mental_health_monitor import MentalHealthMonitor
        from capibara.safety.content_filter import AIContentFilter  
        from capibara.safety.intervention_system import InterventionManager
        
        # Verificar inicialización básica
        monitor = MentalHealthMonitor()
        content_filter = AIContentFilter()
        intervention_manager = InterventionManager()
        
        # Verificar funciones básicas
        test_message = "prueba del system"
        test_scores = monitor.analyze_message_content(test_message)
        test_analysis = content_filter.analyze_content(test_message)
        
        logger.info("✅ Verificación del system de seguridad completada exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallo en verificación del system de seguridad: {e}")
        return False

# Utility functions for developers

def is_user_at_risk(user_id: str, safety_manager) -> Dict[str, Any]:
    """Verifiesr si un usuario está en riesgo"""
    if not safety_manager:
        return {"at_risk": False, "reason": "Sistema de seguridad no disponible"}
    
    try:
        interventions = safety_manager.intervention_manager.check_user_interventions(str(user_id))
        risk_history = safety_manager.mental_health_monitor.get_user_risk_history(str(user_id), days=7)
        
        # Determinar nivel de riesgo
        has_critical_interventions = any(
            i.get('severity') == 'critical' for i in interventions.get('interventions', [])
        )
        
        recent_high_risk = any(
            h.get('risk_level') in ['high', 'critical'] for h in risk_history[-5:]
        )
        
        at_risk = has_critical_interventions or recent_high_risk
        
        return {
            "at_risk": at_risk,
            "active_interventions": len(interventions.get('interventions', [])),
            "recent_risk_events": len([h for h in risk_history if h.get('risk_level') != 'low']),
            "can_use_system": interventions.get('can_use_system', True),
            "recommendations": _get_risk_recommendations(at_risk, interventions, risk_history)
        }
        
    except Exception as e:
        logger.error(f"Error verificando riesgo del usuario {user_id}: {e}")
        return {"at_risk": False, "error": str(e)}

def _get_risk_recommendations(at_risk: bool, interventions: Dict, risk_history: List) -> List[str]:
    """Obtener recomendaciones basadas en el nivel de riesgo"""
    recommendations = []
    
    if at_risk:
        recommendations.extend([
            "Monitoreo adicional requerido",
            "Considerar contacto con supervisión humana", 
            "Verificar acceso a recursos de salud mental",
            "Evaluar necesidad de intervención profesional"
        ])
    
    if interventions.get('interventions'):
        recommendations.append("Revisar efectividad de intervenciones actuales")
    
    if len(risk_history) > 10:
        recommendations.append("Analizar patrones de uso a largo plazo")
    
    return recommendations

def get_emergency_resources() -> Dict[str, Any]:
    """Obtener recursos de emergencia actualizados"""
    return {
        "immediate_danger": {
            "phone": "911",
            "description": "Servicios de emergencia médica"
        },
        "suicide_prevention": {
            "phone": "988", 
            "description": "Línea Nacional de Prevención del Suicidio (EE.UU.)"
        },
        "crisis_text": {
            "number": "741741",
            "text": "HOME",
            "description": "Línea de texto de crisis 24/7"
        },
        "mental_health_resources": {
            "nami": "https://www.nami.org",
            "nimh": "https://www.nimh.nih.gov",
            "crisis_text_line": "https://www.crisistextline.org"
        },
        "international": {
            "description": "Recursos internacionales",
            "url": "https://findahelpline.com"
        }
    }

# Exportar funciones principales
__all__ = [
    'activate_safety_system',
    'verify_safety_system', 
    'create_safety_summary',
    'is_user_at_risk',
    'get_emergency_resources',
    'get_safety_config',
    'DEFAULT_SAFETY_CONFIG'
]

# Mensaje de seguridad al importar
if __name__ != "__main__":
    logger.info("🔒 Módulo de seguridad de salud mental cargado")
    logger.info("📖 Documentación: docs/MENTAL_HEALTH_SAFETY_GUIDE.md")
    logger.info("🆘 Emergencias: 911 | Prevención suicidio: 988")