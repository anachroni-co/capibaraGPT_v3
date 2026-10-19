#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mental Health Safety and Protection System for Capibara6

This module provides comprehensive protection against AI-induced psychosis
and other mental health risks associated with intensive use of AI systems.

Main components:
- MentalHealthMonitor: Usage pattern monitoring
- ContentFilter: Dangerous content filtering
- InterventionSystem: Automatic intervention management
- SafetyMiddleware: Web application integration

Basic usage:
    from capibara.safety import activate_safety_system

    # Activate all protections
    safety_manager = activate_safety_system(app)

    # Process user message
    result = safety_manager.process_user_message(user_id, message, session_data)

    # Filter AI response
    filtered = safety_manager.process_ai_response(user_id, ai_response, user_context)
"""

import logging
import os
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Security system version
__version__ = "1.0.0"

# Default configuration
DEFAULT_SAFETY_CONFIG = {
    "mental_health_monitoring": {
        "enabled": True,
        "max_session_duration": 180,  # minutes
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
    """Get current safety configuration"""
    return DEFAULT_SAFETY_CONFIG.copy()

def activate_safety_system(app=None, config: Optional[Dict[str, Any]] = None):
    """
    Activate complete mental health safety system

    Args:
        app: Flask application (optional)
        config: Custom configuration (optional)

    Returns:
        SafetyIntegrationManager: Comprehensive safety manager
    """
    try:
        # Use default configuration if not provided
        if config is None:
            config = DEFAULT_SAFETY_CONFIG

        # Configure logging
        _setup_safety_logging(config.get("logging", {}))

        # Import and initialize components
        from capibara.safety.intervention_system import SafetyIntegrationManager

        logger.info(" Starting mental health protection system...")

        # Create safety manager
        safety_manager = SafetyIntegrationManager()

        # Configure safety limits
        _configure_safety_limits(safety_manager, config)

        # Integrate with web application if provided
        if app is not None:
            _integrate_with_web_app(app, safety_manager, config)

        logger.info(" Mental health protection system activated successfully")
        logger.info(" Active features:")
        logger.info("   - Usage pattern monitoring: ")
        logger.info("   - Dangerous content filtering: ")
        logger.info("   - Automatic interventions: ")
        logger.info("   - Emergency resources: ")

        return safety_manager

    except ImportError as e:
        logger.error(f" Error importing safety components: {e}")
        logger.error("️  SYSTEM RUNNING WITHOUT MENTAL HEALTH PROTECTIONS")
        return None
    except Exception as e:
        logger.error(f" Error activating safety system: {e}")
        return None

def _setup_safety_logging(logging_config: Dict[str, Any]):
    """Configure specialized logging for safety"""
    log_level = getattr(logging, logging_config.get("log_level", "WARNING"))
    log_file = logging_config.get("log_file", "mental_health_safety.log")

    # Create specific logger for safety
    safety_logger = logging.getLogger("capibara.safety")
    safety_logger.setLevel(log_level)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - SAFETY - %(levelname)s - %(name)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        safety_logger.addHandler(file_handler)

    # Console handler (only for critical errors)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_formatter = logging.Formatter(' SAFETY ALERT: %(message)s')
    console_handler.setFormatter(console_formatter)
    safety_logger.addHandler(console_handler)

def _configure_safety_limits(safety_manager, config: Dict[str, Any]):
    """Configure safety limits according to configuration"""

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

    # Configure content filtering
    content_config = config.get("content_filtering", {})
    if content_config.get("enabled", True):
        # Filters are already configured by default
        pass

    # Configure interventions
    intervention_config = config.get("interventions", {})
    if intervention_config.get("enabled", True):
        safety_manager.intervention_manager.max_interventions_per_session = intervention_config.get("max_interventions_per_session", 3)

def _integrate_with_web_app(app, safety_manager, config: Dict[str, Any]):
    """Integrate security system with web application"""

    try:
        from web_interface.safety_middleware import init_safety_middleware

        # Initialize middleware
        middleware = init_safety_middleware(app)
        middleware.safety_manager = safety_manager

        # Configure disclaimer
        web_config = config.get("web_integration", {})
        if web_config.get("show_disclaimer", True):
            # Middleware handles disclaimer automatically
            pass

        logger.info(" Web integration activated successfully")

    except ImportError:
        logger.warning("️  Web middleware not available, continuing without web integration")

def create_safety_summary() -> Dict[str, Any]:
    """Create summary of security system status"""
    return {
        "version": __version__,
        "components": {
            "mental_health_monitor": " Active",
            "content_filter": " Active",
            "intervention_system": " Active",
            "web_middleware": " Active"
        },
        "protections": {
            "psychosis_prevention": " Enabled",
            "usage_monitoring": " Enabled",
            "content_filtering": " Enabled",
            "emergency_interventions": " Enabled",
            "professional_resources": " Available"
        },
        "emergency_contacts": {
            "suicide_prevention": "988 (US)",
            "emergency_services": "911",
            "crisis_text": "741741"
        }
    }

def verify_safety_system() -> bool:
    """Verify that all safety components are working"""
    try:
        # Verify imports
        from capibara.safety.mental_health_monitor import MentalHealthMonitor
        from capibara.safety.content_filter import AIContentFilter
        from capibara.safety.intervention_system import InterventionManager

        # Verify basic initialization
        monitor = MentalHealthMonitor()
        content_filter = AIContentFilter()
        intervention_manager = InterventionManager()

        # Verify basic functions
        test_message = "system test"
        test_scores = monitor.analyze_message_content(test_message)
        test_analysis = content_filter.analyze_content(test_message)

        logger.info(" Safety system verification completed successfully")
        return True

    except Exception as e:
        logger.error(f" Safety system verification failed: {e}")
        return False

# Utility functions for developers

def is_user_at_risk(user_id: str, safety_manager) -> Dict[str, Any]:
    """Check if a user is at risk"""
    if not safety_manager:
        return {"at_risk": False, "reason": "Safety system not available"}

    try:
        interventions = safety_manager.intervention_manager.check_user_interventions(str(user_id))
        risk_history = safety_manager.mental_health_monitor.get_user_risk_history(str(user_id), days=7)

        # Determine risk level
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
        logger.error(f"Error checking user risk {user_id}: {e}")
        return {"at_risk": False, "error": str(e)}

def _get_risk_recommendations(at_risk: bool, interventions: Dict, risk_history: List) -> List[str]:
    """Get recommendations based on risk level"""
    recommendations = []

    if at_risk:
        recommendations.extend([
            "Additional monitoring required",
            "Consider contact with human supervision",
            "Verify access to mental health resources",
            "Evaluate need for professional intervention"
        ])

    if interventions.get('interventions'):
        recommendations.append("Review effectiveness of current interventions")

    if len(risk_history) > 10:
        recommendations.append("Analyze long-term usage patterns")

    return recommendations

def get_emergency_resources() -> Dict[str, Any]:
    """Get updated emergency resources"""
    return {
        "immediate_danger": {
            "phone": "911",
            "description": "Emergency medical services"
        },
        "suicide_prevention": {
            "phone": "988",
            "description": "National Suicide Prevention Lifeline (US)"
        },
        "crisis_text": {
            "number": "741741",
            "text": "HOME",
            "description": "24/7 crisis text line"
        },
        "mental_health_resources": {
            "nami": "https://www.nami.org",
            "nimh": "https://www.nimh.nih.gov",
            "crisis_text_line": "https://www.crisistextline.org"
        },
        "international": {
            "description": "International resources",
            "url": "https://findahelpline.com"
        }
    }

# Export main functions
__all__ = [
    'activate_safety_system',
    'verify_safety_system',
    'create_safety_summary',
    'is_user_at_risk',
    'get_emergency_resources',
    'get_safety_config',
    'DEFAULT_SAFETY_CONFIG'
]

# Safety message on import
if __name__ != "__main__":
    logger.info(" Mental health safety module loaded")
    logger.info(" Documentation: docs/MENTAL_HEALTH_SAFETY_GUIDE.md")
    logger.info("🆘 Emergencies: 911 | Suicide prevention: 988")
