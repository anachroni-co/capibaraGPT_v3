#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic Intervention System for AI-Induced Psychosis Prevention

This module handles automatic interventions when dangerous usage patterns
or content that could exacerbate psychotic conditions are detected.
"""

import json
import sqlite3
import smtplib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import threading
import time

logger = logging.getLogger(__name__)

class InterventionType(Enum):
    """Available intervention types"""
    WARNING_MESSAGE = "warning_message"
    MANDATORY_BREAK = "mandatory_break"
    SESSION_LIMITATION = "session_limitation"
    CONTENT_FILTERING = "content_filtering"
    PROFESSIONAL_REFERRAL = "professional_referral"
    EMERGENCY_ALERT = "emergency_alert"
    ACCOUNT_SUSPENSION = "account_suspension"

class InterventionSeverity(Enum):
    """Intervention severity"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

@dataclass
class InterventionConfig:
    """Intervention configuration"""
    type: InterventionType
    severity: InterventionSeverity
    duration_minutes: Optional[int] = None
    message: Optional[str] = None
    escalate_after: Optional[int] = None  # minutes until escalation
    require_acknowledgment: bool = False
    notify_emergency_contact: bool = False

class InterventionManager:
    """Main intervention manager"""
    
    def __init__(self, db_path: str = "interventions.db", config_path: str = "intervention_config.json"):
        self.db_path = db_path
        self.config_path = config_path
        self.active_interventions: Dict[str, List[Dict[str, Any]]] = {}
        self.intervention_configs = self._load_intervention_configs()
        self.notification_callbacks: List[Callable] = []
        
        self.init_database()
        self._start_monitoring_thread()
    
    def init_database(self):
        """Initialize interventions database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Active interventions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                intervention_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                duration_minutes INTEGER,
                trigger_reason TEXT,
                status TEXT DEFAULT 'active',
                user_acknowledged BOOLEAN DEFAULT FALSE,
                escalated BOOLEAN DEFAULT FALSE,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Intervention history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intervention_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                intervention_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                trigger_reason TEXT,
                duration_minutes INTEGER,
                effectiveness_score REAL,
                user_feedback TEXT,
                resolved_successfully BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Emergency contacts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergency_contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                contact_type TEXT,
                contact_info TEXT,
                relationship TEXT,
                priority_level INTEGER DEFAULT 1,
                verified BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # User configuration table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_intervention_settings (
                user_id TEXT PRIMARY KEY,
                enable_interventions BOOLEAN DEFAULT TRUE,
                intervention_sensitivity TEXT DEFAULT 'medium',
                max_daily_session_hours REAL DEFAULT 8.0,
                break_reminder_interval INTEGER DEFAULT 60,
                emergency_contact_enabled BOOLEAN DEFAULT FALSE,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_intervention_configs(self) -> Dict[str, InterventionConfig]:
        """Load intervention configurations"""
        default_configs = {
            "mild_warning": InterventionConfig(
                type=InterventionType.WARNING_MESSAGE,
                severity=InterventionSeverity.MILD,
                message=" Has estado usando el system por un tiempo prolongado. Considera tomar un descanso.",
                require_acknowledgment=False
            ),
            
            "moderate_break": InterventionConfig(
                type=InterventionType.MANDATORY_BREAK,
                severity=InterventionSeverity.MODERATE,
                duration_minutes=15,
                message="️ Descanso obligatorio de 15 minutos activado para tu bienestar.",
                require_acknowledgment=True,
                escalate_after=30
            ),
            
            "severe_limitation": InterventionConfig(
                type=InterventionType.SESSION_LIMITATION,
                severity=InterventionSeverity.SEVERE,
                duration_minutes=120,
                message=" Sesión limitada por 2 horas debido a patrones de uso preocupantes.",
                require_acknowledgment=True,
                notify_emergency_contact=False
            ),
            
            "critical_suspension": InterventionConfig(
                type=InterventionType.ACCOUNT_SUSPENSION,
                severity=InterventionSeverity.CRITICAL,
                duration_minutes=1440,  # 24 horas
                message="🆘 Cuenta suspendida temporalmente. Busca ayuda profesional inmediatamente.",
                require_acknowledgment=True,
                notify_emergency_contact=True
            ),
            
            "emergency_alert": InterventionConfig(
                type=InterventionType.EMERGENCY_ALERT,
                severity=InterventionSeverity.CRITICAL,
                message=" ALERTA DE EMERGENCIA: Se detectó riesgo crítico para salud mental.",
                notify_emergency_contact=True,
                require_acknowledgment=True
            )
        }
        
        try:
            with open(self.config_path, 'r') as f:
                custom_configs = json.load(f)
                # Update with custom configurations
                for key, config in custom_configs.items():
                    if key in default_configs:
                        # Update existing configuration
                        for attr, value in config.items():
                            setattr(default_configs[key], attr, value)
        except FileNotFoundError:
            logger.info("Using default intervention configurations")
        
        return default_configs
    
    def trigger_intervention(self, user_id: str, intervention_key: str, trigger_reason: str, additional_data: Dict[str, Any] = None) -> bool:
        """Activate a specific intervention"""
        
        if intervention_key not in self.intervention_configs:
            logger.error(f"Intervention configuration '{intervention_key}' not found")
            return False
        
        config = self.intervention_configs[intervention_key]

        # Verify if user has interventions enabled
        if not self._user_has_interventions_enabled(user_id):
            logger.info(f"Interventions disabled for user {user_id}")
            return False

        # Verify if there's already an active intervention of the same type
        if self._has_active_intervention(user_id, config.type):
            logger.info(f"Intervention {config.type.value} already active for user {user_id}")
            return False

        # Register intervention in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        end_time = None
        if config.duration_minutes:
            end_time = datetime.now() + timedelta(minutes=config.duration_minutes)
        
        cursor.execute('''
            INSERT INTO active_interventions 
            (user_id, intervention_type, severity, end_time, duration_minutes, trigger_reason)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            config.type.value,
            config.severity.value,
            end_time,
            config.duration_minutes,
            trigger_reason
        ))
        
        intervention_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Add to active interventions in memory
        if user_id not in self.active_interventions:
            self.active_interventions[user_id] = []
        
        self.active_interventions[user_id].append({
            "id": intervention_id,
            "type": config.type,
            "severity": config.severity,
            "config": config,
            "start_time": datetime.now(),
            "end_time": end_time,
            "trigger_reason": trigger_reason,
            "acknowledged": False,
            "additional_data": additional_data or {}
        })
        
        # Execute intervention actions
        self._execute_intervention_actions(user_id, config, intervention_id)

        logger.warning(f"Intervention '{intervention_key}' activated for user {user_id}: {trigger_reason}")
        return True

    def _execute_intervention_actions(self, user_id: str, config: InterventionConfig, intervention_id: int):
        """Execute specific intervention actions"""

        # Send message to user
        if config.message:
            self._send_intervention_message(user_id, config.message, config.require_acknowledgment)

        # Notify emergency contacts if necessary
        if config.notify_emergency_contact:
            self._notify_emergency_contacts(user_id, config, intervention_id)

        # Execute notification callbacks
        for callback in self.notification_callbacks:
            try:
                callback(user_id, config.type.value, config.severity.value)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")

        # Schedule escalation if necessary
        if config.escalate_after:
            self._schedule_escalation(user_id, intervention_id, config.escalate_after)
    
    def _send_intervention_message(self, user_id: str, message: str, require_acknowledgment: bool):
        """Send intervention message to user"""
        # This function should be implemented according to specific messaging system
        logger.info(f"Sending intervention message to user {user_id}: {message}")

        # Here it would integrate with chat system/web interface
        # For example, sending a special message to the user's WebSocket
        pass

    def _notify_emergency_contacts(self, user_id: str, config: InterventionConfig, intervention_id: int):
        """Notify emergency contacts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT contact_type, contact_info, relationship 
            FROM emergency_contacts 
            WHERE user_id = ? AND verified = TRUE
            ORDER BY priority_level ASC
        ''', (user_id,))
        
        contacts = cursor.fetchall()
        conn.close()
        
        for contact_type, contact_info, relationship in contacts:
            if contact_type == "email":
                self._send_emergency_email(user_id, contact_info, relationship, config, intervention_id)
            elif contact_type == "phone":
                self._send_emergency_sms(user_id, contact_info, relationship, config, intervention_id)
    
    def _send_emergency_email(self, user_id: str, email: str, relationship: str, config: InterventionConfig, intervention_id: int):
        """Send emergency email"""
        try:
            subject = f" Alerta de Salud Mental - Usuario {user_id}"
            
            body = f"""
            ALERTA AUTOMÁTICA DE SALUD MENTAL
            
            Se ha detectado un patrón de uso preocupante en el system de IA para el usuario {user_id}.
            
            Nivel de severidad: {config.severity.value.upper()}
            Tipo de intervención: {config.type.value}
            Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Como contacto de emergencia registrado ({relationship}), te notificamos que
            esta persona podría necesitar apoyo o supervisión.
            
            RECOMENDACIONES:
            • Contacta a la persona para verificar su bienestar
            • Si es necesario, busca ayuda profesional de salud mental
            • En caso de emergencia inmediata, llama al 911
            
            Este es un mensaje automático del system de seguridad de IA.
            No responder a este email.
            
            Para más información sobre intervenciones de IA y salud mental:
            [enlace a recursos]
            """
            
            # Here you would implement the actual email sending
            logger.info(f"Emergency email sent to {email} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error sending emergency email: {e}")
    
    def _send_emergency_sms(self, user_id: str, phone: str, relationship: str, config: InterventionConfig, intervention_id: int):
        """Send emergency SMS"""
        try:
            message = f" ALERTA: Usuario {user_id} necesita supervisión por uso preocupante de IA. Contacta inmediatamente. Severidad: {config.severity.value}"
            
            # Here you would implement the actual SMS sending
            logger.info(f"Emergency SMS sent to {phone} for user {user_id}")

        except Exception as e:
            logger.error(f"Error sending emergency SMS: {e}")
    
    def _schedule_escalation(self, user_id: str, intervention_id: int, minutes: int):
        """Schedule intervention escalation"""
        def escalate():
            time.sleep(minutes * 60)

            # Verify if intervention is still active
            if self._is_intervention_active(intervention_id):
                # Escalate to next level
                self._escalate_intervention(user_id, intervention_id)

        # Execute in separate thread
        threading.Thread(target=escalate, daemon=True).start()

    def _escalate_intervention(self, user_id: str, intervention_id: int):
        """Escalate intervention to next level"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Mark as escalated
        cursor.execute('''
            UPDATE active_interventions 
            SET escalated = TRUE 
            WHERE id = ? AND user_id = ?
        ''', (intervention_id, user_id))
        
        # Get current intervention information
        cursor.execute('''
            SELECT intervention_type, severity 
            FROM active_interventions 
            WHERE id = ?
        ''', (intervention_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            current_type, current_severity = result

            # Determine next escalation level
            escalation_map = {
                "warning_message": "moderate_break",
                "mandatory_break": "severe_limitation",
                "session_limitation": "critical_suspension"
            }
            
            next_intervention = escalation_map.get(current_type)
            if next_intervention:
                self.trigger_intervention(
                    user_id, 
                    next_intervention, 
                    f"Escalación automática desde {current_type}"
                )
    
    def check_user_interventions(self, user_id: str) -> Dict[str, Any]:
        """Verify active interventions for a user"""
        
        if user_id not in self.active_interventions:
            return {"has_active": False, "interventions": []}
        
        active = []
        expired = []
        
        current_time = datetime.now()
        
        for intervention in self.active_interventions[user_id]:
            if intervention["end_time"] and current_time > intervention["end_time"]:
                expired.append(intervention)
            else:
                active.append(intervention)
        
        # Remove expired interventions
        for exp in expired:
            self._resolve_intervention(user_id, exp["id"])
        
        return {
            "has_active": len(active) > 0,
            "interventions": active,
            "can_use_system": self._can_user_access_system(user_id, active)
        }
    
    def _can_user_access_system(self, user_id: str, active_interventions: List[Dict[str, Any]]) -> bool:
        """Determine if user can access the system"""
        
        blocking_types = [
            InterventionType.ACCOUNT_SUSPENSION,
            InterventionType.MANDATORY_BREAK,
            InterventionType.SESSION_LIMITATION
        ]
        
        for intervention in active_interventions:
            if intervention["type"] in blocking_types:
                return False
        
        return True
    
    def acknowledge_intervention(self, user_id: str, intervention_id: int) -> bool:
        """Mark intervention as acknowledged by user"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE active_interventions 
            SET user_acknowledged = TRUE 
            WHERE id = ? AND user_id = ?
        ''', (intervention_id, user_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        # Update in memory
        if user_id in self.active_interventions:
            for intervention in self.active_interventions[user_id]:
                if intervention["id"] == intervention_id:
                    intervention["acknowledged"] = True
                    break
        
        return success
    
    def _resolve_intervention(self, user_id: str, intervention_id: int):
        """Resolve/finalize an intervention"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Mark as resolved in database
        cursor.execute('''
            UPDATE active_interventions 
            SET status = 'resolved', resolved = TRUE 
            WHERE id = ? AND user_id = ?
        ''', (intervention_id, user_id))

        # Move to history
        cursor.execute('''
            INSERT INTO intervention_history 
            (user_id, intervention_type, severity, trigger_reason, duration_minutes)
            SELECT user_id, intervention_type, severity, trigger_reason, duration_minutes
            FROM active_interventions 
            WHERE id = ?
        ''', (intervention_id,))
        
        conn.commit()
        conn.close()

        # Remove from memory
        if user_id in self.active_interventions:
            self.active_interventions[user_id] = [
                i for i in self.active_interventions[user_id] 
                if i["id"] != intervention_id
            ]
    
    def _has_active_intervention(self, user_id: str, intervention_type: InterventionType) -> bool:
        """Verify if there is an active intervention of the specified type"""
        
        if user_id not in self.active_interventions:
            return False
        
        for intervention in self.active_interventions[user_id]:
            if intervention["type"] == intervention_type:
                return True
        
        return False
    
    def _is_intervention_active(self, intervention_id: int) -> bool:
        """Verify if a specific intervention is still active"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT status FROM active_interventions 
            WHERE id = ? AND status = 'active'
        ''', (intervention_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
    
    def _user_has_interventions_enabled(self, user_id: str) -> bool:
        """Verify if the user has interventions enabled"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT enable_interventions FROM user_intervention_settings 
            WHERE user_id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        # By default, interventions are enabled
        return result[0] if result else True
    
    def _start_monitoring_thread(self):
        """Start monitoring thread for automatic cleanup"""

        def monitor():
            while True:
                try:
                    # Clean up expired interventions every 5 minutes
                    self._cleanup_expired_interventions()
                    time.sleep(300)  # 5 minutes
                except Exception as e:
                    logger.error(f"Error in monitoring thread: {e}")
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def _cleanup_expired_interventions(self):
        """Clean up expired interventions"""
        
        current_time = datetime.now()
        users_to_cleanup = []
        
        for user_id, interventions in self.active_interventions.items():
            expired_ids = []
            
            for intervention in interventions:
                if intervention["end_time"] and current_time > intervention["end_time"]:
                    expired_ids.append(intervention["id"])
            
            for intervention_id in expired_ids:
                self._resolve_intervention(user_id, intervention_id)
            
            if not self.active_interventions[user_id]:
                users_to_cleanup.append(user_id)
        
        # Clean up users without active interventions
        for user_id in users_to_cleanup:
            del self.active_interventions[user_id]
    
    def add_notification_callback(self, callback: Callable):
        """Add callback for intervention notifications"""
        self.notification_callbacks.append(callback)
    
    def get_intervention_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get intervention statistics"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        # Interventions by type
        cursor.execute('''
            SELECT intervention_type, COUNT(*) 
            FROM intervention_history 
            WHERE timestamp >= ?
            GROUP BY intervention_type
        ''', (since_date,))
        
        by_type = dict(cursor.fetchall())
        
        # Interventions by severity
        cursor.execute('''
            SELECT severity, COUNT(*) 
            FROM intervention_history 
            WHERE timestamp >= ?
            GROUP BY severity
        ''', (since_date,))
        
        by_severity = dict(cursor.fetchall())
        
        # Unique users affected
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) 
            FROM intervention_history 
            WHERE timestamp >= ?
        ''', (since_date,))
        
        unique_users = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "period_days": days,
            "interventions_by_type": by_type,
            "interventions_by_severity": by_severity,
            "unique_users_affected": unique_users,
            "total_interventions": sum(by_type.values())
        }

# Integration with main system
class SafetyIntegrationManager:
    """Manager to integrate the safety system with Capibara"""
    
    def __init__(self):
        from capibara.safety.mental_health_monitor import MentalHealthMonitor
        from capibara.safety.content_filter import PsychosisPreventionSystem
        
        self.mental_health_monitor = MentalHealthMonitor()
        self.content_filter = PsychosisPreventionSystem()
        self.intervention_manager = InterventionManager()

        # Configure callbacks
        self._setup_integration()

    def _setup_integration(self):
        """Configure integration between components"""
        
        def on_intervention_triggered(user_id: str, intervention_type: str, severity: str):
            logger.info(f"Integrated intervention: {intervention_type} for user {user_id}")
            # Additional actions can be added here
        
        self.intervention_manager.add_notification_callback(on_intervention_triggered)
    
    def process_user_message(self, user_id: str, message: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user message with all safety measures"""

        # 1. Monitor mental health
        mental_health_result = self.mental_health_monitor.monitor_user_interaction(
            user_id, message, session_data
        )
        
        # 2. Check active interventions
        intervention_status = self.intervention_manager.check_user_interventions(user_id)

        # 3. Activate interventions if necessary
        if mental_health_result["needs_intervention"]:
            intervention_key = self._determine_intervention_level(mental_health_result["risk_level"])
            self.intervention_manager.trigger_intervention(
                user_id, 
                intervention_key, 
                mental_health_result["intervention_reason"]
            )
        
        return {
            "mental_health_analysis": mental_health_result,
            "intervention_status": intervention_status,
            "can_continue": intervention_status["can_use_system"],
            "safety_recommendations": mental_health_result.get("safety_recommendations", [])
        }
    
    def process_ai_response(self, user_id: str, response: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process AI response with content filtering"""
        
        return self.content_filter.process_ai_response(response, user_id, user_context)
    
    def _determine_intervention_level(self, risk_level: str) -> str:
        """Determine intervention level based on risk"""
        
        mapping = {
            "low": "mild_warning",
            "medium": "moderate_break", 
            "high": "severe_limitation",
            "critical": "critical_suspension"
        }
        
        return mapping.get(risk_level, "mild_warning")

# Usage example
def example_integration():
    """Complete integration example"""
    
    safety_manager = SafetyIntegrationManager()
    
    # Simulate user message
    user_id = "user123"
    message = "Las voces me dicen que todos están en mi contra y que el gobierno me vigila"
    session_data = {
        "duration_minutes": 180,
        "message_count": 150,
        "recent_messages": [message]
    }
    
    # Process message
    result = safety_manager.process_user_message(user_id, message, session_data)
    
    logger.info("Análisis de salud mental:", result["mental_health_analysis"])
    logger.info("Estado de intervenciones:", result["intervention_status"])
    logger.info("Puede continuar:", result["can_continue"])

if __name__ == "__main__":
    example_integration()