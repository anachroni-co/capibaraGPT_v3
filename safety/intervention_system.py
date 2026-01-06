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
        
        # Tabla de contactos de emergencia
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
        """Cargar configuraciones de intervención"""
        default_configs = {
            "mild_warning": InterventionConfig(
                type=InterventionType.WARNING_MESSAGE,
                severity=InterventionSeverity.MILD,
                message="🔔 Has estado usando el system por un tiempo prolongado. Considera tomar un descanso.",
                require_acknowledgment=False
            ),
            
            "moderate_break": InterventionConfig(
                type=InterventionType.MANDATORY_BREAK,
                severity=InterventionSeverity.MODERATE,
                duration_minutes=15,
                message="⏸️ Descanso obligatorio de 15 minutos activado para tu bienestar.",
                require_acknowledgment=True,
                escalate_after=30
            ),
            
            "severe_limitation": InterventionConfig(
                type=InterventionType.SESSION_LIMITATION,
                severity=InterventionSeverity.SEVERE,
                duration_minutes=120,
                message="🚫 Sesión limitada por 2 horas debido a patrones de uso preocupantes.",
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
                message="🚨 ALERTA DE EMERGENCIA: Se detectó riesgo crítico para salud mental.",
                notify_emergency_contact=True,
                require_acknowledgment=True
            )
        }
        
        try:
            with open(self.config_path, 'r') as f:
                custom_configs = json.load(f)
                # Actualizar con configuraciones personalizadas
                for key, config in custom_configs.items():
                    if key in default_configs:
                        # Update existing configuration
                        for attr, value in config.items():
                            setattr(default_configs[key], attr, value)
        except FileNotFoundError:
            logger.info("Usando configuraciones de intervención por defecto")
        
        return default_configs
    
    def trigger_intervention(self, user_id: str, intervention_key: str, trigger_reason: str, additional_data: Dict[str, Any] = None) -> bool:
        """Activar una intervención específica"""
        
        if intervention_key not in self.intervention_configs:
            logger.error(f"Configuración de intervención '{intervention_key}' no encontrada")
            return False
        
        config = self.intervention_configs[intervention_key]
        
        # Verificar si el usuario tiene intervenciones habilitadas
        if not self._user_has_interventions_enabled(user_id):
            logger.info(f"Intervenciones deshabilitadas para usuario {user_id}")
            return False
        
        # Verificar si ya hay una intervención activa del mismo tipo
        if self._has_active_intervention(user_id, config.type):
            logger.info(f"Intervención {config.type.value} ya activa para usuario {user_id}")
            return False
        
        # Registrar intervención en base de datos
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
        
        # Agregar a intervenciones activas en memoria
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
        
        # Ejecutar acciones de intervención
        self._execute_intervention_actions(user_id, config, intervention_id)
        
        logger.warning(f"Intervención '{intervention_key}' activada para usuario {user_id}: {trigger_reason}")
        return True
    
    def _execute_intervention_actions(self, user_id: str, config: InterventionConfig, intervention_id: int):
        """Ejecutar acciones específicas de la intervención"""
        
        # Enviar mensaje al usuario
        if config.message:
            self._send_intervention_message(user_id, config.message, config.require_acknowledgment)
        
        # Notificar contactos de emergencia si es necesario
        if config.notify_emergency_contact:
            self._notify_emergency_contacts(user_id, config, intervention_id)
        
        # Ejecutar callbacks de notificación
        for callback in self.notification_callbacks:
            try:
                callback(user_id, config.type.value, config.severity.value)
            except Exception as e:
                logger.error(f"Error en callback de notificación: {e}")
        
        # Programar escalación si es necesario
        if config.escalate_after:
            self._schedule_escalation(user_id, intervention_id, config.escalate_after)
    
    def _send_intervention_message(self, user_id: str, message: str, require_acknowledgment: bool):
        """Enviar mensaje de intervención al usuario"""
        # This function should be implemented according to specific messaging system
        logger.info(f"Enviando mensaje de intervención a usuario {user_id}: {message}")
        
        # Here it would integrate with chat system/web interface
        # Por ejemplo, enviando un mensaje especial al WebSocket del usuario
        pass
    
    def _notify_emergency_contacts(self, user_id: str, config: InterventionConfig, intervention_id: int):
        """Notificar contactos de emergencia"""
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
        """Enviar email de emergencia"""
        try:
            subject = f"🚨 Alerta de Salud Mental - Usuario {user_id}"
            
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
            
            # Aquí implementarías el envío real del email
            logger.info(f"Email de emergencia enviado a {email} para usuario {user_id}")
            
        except Exception as e:
            logger.error(f"Error enviando email de emergencia: {e}")
    
    def _send_emergency_sms(self, user_id: str, phone: str, relationship: str, config: InterventionConfig, intervention_id: int):
        """Enviar SMS de emergencia"""
        try:
            message = f"🚨 ALERTA: Usuario {user_id} necesita supervisión por uso preocupante de IA. Contacta inmediatamente. Severidad: {config.severity.value}"
            
            # Aquí implementarías el envío real del SMS
            logger.info(f"SMS de emergencia enviado a {phone} para usuario {user_id}")
            
        except Exception as e:
            logger.error(f"Error enviando SMS de emergencia: {e}")
    
    def _schedule_escalation(self, user_id: str, intervention_id: int, minutes: int):
        """Programar escalación de intervención"""
        def escalate():
            time.sleep(minutes * 60)
            
            # Verificar si la intervención sigue activa
            if self._is_intervention_active(intervention_id):
                # Escalar a siguiente nivel
                self._escalate_intervention(user_id, intervention_id)
        
        # Ejecutar en hilo separado
        threading.Thread(target=escalate, daemon=True).start()
    
    def _escalate_intervention(self, user_id: str, intervention_id: int):
        """Escalar intervención a siguiente nivel"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Marcar como escalada
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
            
            # Determinar siguiente nivel de escalación
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
        """Verifiesr intervenciones activas para un usuario"""
        
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
        
        # Remover intervenciones expiradas
        for exp in expired:
            self._resolve_intervention(user_id, exp["id"])
        
        return {
            "has_active": len(active) > 0,
            "interventions": active,
            "can_use_system": self._can_user_access_system(user_id, active)
        }
    
    def _can_user_access_system(self, user_id: str, active_interventions: List[Dict[str, Any]]) -> bool:
        """Determinar si el usuario puede acceder al system"""
        
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
        """Marcar intervención como reconocida por el usuario"""
        
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
        
        # Actualizar en memoria
        if user_id in self.active_interventions:
            for intervention in self.active_interventions[user_id]:
                if intervention["id"] == intervention_id:
                    intervention["acknowledged"] = True
                    break
        
        return success
    
    def _resolve_intervention(self, user_id: str, intervention_id: int):
        """Resolver/finalizar una intervención"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Marcar como resuelta en base de datos
        cursor.execute('''
            UPDATE active_interventions 
            SET status = 'resolved', resolved = TRUE 
            WHERE id = ? AND user_id = ?
        ''', (intervention_id, user_id))
        
        # Mover a historial
        cursor.execute('''
            INSERT INTO intervention_history 
            (user_id, intervention_type, severity, trigger_reason, duration_minutes)
            SELECT user_id, intervention_type, severity, trigger_reason, duration_minutes
            FROM active_interventions 
            WHERE id = ?
        ''', (intervention_id,))
        
        conn.commit()
        conn.close()
        
        # Remover de memoria
        if user_id in self.active_interventions:
            self.active_interventions[user_id] = [
                i for i in self.active_interventions[user_id] 
                if i["id"] != intervention_id
            ]
    
    def _has_active_intervention(self, user_id: str, intervention_type: InterventionType) -> bool:
        """Verifiesr si hay intervención activa del tipo especificado"""
        
        if user_id not in self.active_interventions:
            return False
        
        for intervention in self.active_interventions[user_id]:
            if intervention["type"] == intervention_type:
                return True
        
        return False
    
    def _is_intervention_active(self, intervention_id: int) -> bool:
        """Verifiesr si una intervención específica sigue activa"""
        
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
        """Verifiesr si el usuario tiene intervenciones habilitadas"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT enable_interventions FROM user_intervention_settings 
            WHERE user_id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        # Por defecto, las intervenciones están habilitadas
        return result[0] if result else True
    
    def _start_monitoring_thread(self):
        """Iniciar hilo de monitoreo para limpieza automática"""
        
        def monitor():
            while True:
                try:
                    # Limpiar intervenciones expiradas cada 5 minutos
                    self._cleanup_expired_interventions()
                    time.sleep(300)  # 5 minutos
                except Exception as e:
                    logger.error(f"Error en hilo de monitoreo: {e}")
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def _cleanup_expired_interventions(self):
        """Limpiar intervenciones expiradas"""
        
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
        
        # Limpiar usuarios sin intervenciones activas
        for user_id in users_to_cleanup:
            del self.active_interventions[user_id]
    
    def add_notification_callback(self, callback: Callable):
        """Agregar callback para notificaciones de intervención"""
        self.notification_callbacks.append(callback)
    
    def get_intervention_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Obtener statistics de intervenciones"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        # Intervenciones por tipo
        cursor.execute('''
            SELECT intervention_type, COUNT(*) 
            FROM intervention_history 
            WHERE timestamp >= ?
            GROUP BY intervention_type
        ''', (since_date,))
        
        by_type = dict(cursor.fetchall())
        
        # Intervenciones por severidad
        cursor.execute('''
            SELECT severity, COUNT(*) 
            FROM intervention_history 
            WHERE timestamp >= ?
            GROUP BY severity
        ''', (since_date,))
        
        by_severity = dict(cursor.fetchall())
        
        # Usuarios únicos afectados
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
    """Manager para integrar el system de seguridad con Capibara"""
    
    def __init__(self):
        from capibara.safety.mental_health_monitor import MentalHealthMonitor
        from capibara.safety.content_filter import PsychosisPreventionSystem
        
        self.mental_health_monitor = MentalHealthMonitor()
        self.content_filter = PsychosisPreventionSystem()
        self.intervention_manager = InterventionManager()
        
        # Configurar callbacks
        self._setup_integration()
    
    def _setup_integration(self):
        """Configurar integración entre componentes"""
        
        def on_intervention_triggered(user_id: str, intervention_type: str, severity: str):
            logger.info(f"Intervención integrada: {intervention_type} para usuario {user_id}")
            # Aquí se pueden agregar acciones adicionales
        
        self.intervention_manager.add_notification_callback(on_intervention_triggered)
    
    def process_user_message(self, user_id: str, message: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesar mensaje del usuario con todas las medidas de seguridad"""
        
        # 1. Monitorear salud mental
        mental_health_result = self.mental_health_monitor.monitor_user_interaction(
            user_id, message, session_data
        )
        
        # 2. Verificar intervenciones activas
        intervention_status = self.intervention_manager.check_user_interventions(user_id)
        
        # 3. Activar intervenciones si es necesario
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
        """Procesar respuesta de IA con filtrado de contenido"""
        
        return self.content_filter.process_ai_response(response, user_id, user_context)
    
    def _determine_intervention_level(self, risk_level: str) -> str:
        """Determinar nivel de intervención based en riesgo"""
        
        mapping = {
            "low": "mild_warning",
            "medium": "moderate_break", 
            "high": "severe_limitation",
            "critical": "critical_suspension"
        }
        
        return mapping.get(risk_level, "mild_warning")

# Ejemplo de uso
def example_integration():
    """Ejemplo de integración completa"""
    
    safety_manager = SafetyIntegrationManager()
    
    # Simular mensaje del usuario
    user_id = "user123"
    message = "Las voces me dicen que todos están en mi contra y que el gobierno me vigila"
    session_data = {
        "duration_minutes": 180,
        "message_count": 150,
        "recent_messages": [message]
    }
    
    # Procesar mensaje
    result = safety_manager.process_user_message(user_id, message, session_data)
    
    print("Análisis de salud mental:", result["mental_health_analysis"])
    print("Estado de intervenciones:", result["intervention_status"])
    print("Puede continuar:", result["can_continue"])

if __name__ == "__main__":
    example_integration()