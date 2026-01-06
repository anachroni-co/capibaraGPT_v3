#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mental Health Monitoring System for Capibara6

This module implements safety measures to prevent AI-induced psychosis
and detect abusive usage patterns that could affect the user's mental health.
"""

import json
import re
import sqlite3
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Mental health risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class UsagePattern:
    """User usage pattern"""
    user_id: str
    session_duration: float  # in minutes
    message_frequency: float  # messages per minute
    topic_obsession_score: float  # 0-1, obsession with specific topics
    reality_disconnection_score: float  # 0-1, reality disconnection
    paranoid_content_score: float  # 0-1, paranoid content
    emotional_volatility_score: float  # 0-1, emotional volatility
    timestamp: datetime

class MentalHealthMonitor:
    """Main mental health monitor"""
    
    def __init__(self, db_path: str = "mental_health_monitor.db"):
        self.db_path = db_path
        self.init_database()
        
        # Dangerous patterns for detecting psychotic content
        self.psychotic_patterns = [
            # Persecution delusions
            r"(?i)(me\s+estan?\s+siguiendo|me\s+persiguen|estan?\s+en\s+mi\s+contra)",
            r"(?i)(todos?\s+estan?\s+confabulando|conspirac[ií][oó]n\s+contra\s+m[ií])",
            r"(?i)(me\s+estan?\s+vigilando|me\s+espían|me\s+controlan)",

            # Grandiosity delusions
            r"(?i)(soy\s+el\s+elegido|tengo\s+poderes?\s+especiales?|soy\s+superior)",
            r"(?i)(misi[oó]n\s+especial|destino\s+único|poder\s+divino)",

            # Auditory hallucinations
            r"(?i)(voces?\s+me\s+dicen|escucho\s+voces?|me\s+hablan\s+en\s+la\s+mente)",
            r"(?i)(susurros?\s+en\s+mi\s+cabeza|sonidos?\s+que\s+no\s+existen?)",

            # Derealization/depersonalization
            r"(?i)(nada\s+es\s+real|todo\s+es\s+una\s+simulaci[oó]n|matrix|realidad\s+falsa)",
            r"(?i)(no\s+soy\s+yo\s+mismo|mi\s+cuerpo\s+no\s+es\s+m[ií]o)",

            # Disorganized thinking
            r"(?i)(conexiones?\s+ocultas?|patrones?\s+en\s+todo|números?\s+me\s+persiguen)",
            r"(?i)(significados?\s+ocultos?|mensajes?\s+en\s+todo|señales?\s+por\s+doquier)"
        ]

        # Problematic obsessive topics
        self.obsessive_topics = [
            "conspiración", "illuminati", "control mental", "matrix", "simulación",
            "voces", "persecución", "espionaje", "poderes sobrenaturales",
            "profecías", "elegido", "misión especial", "realidad falsa"
        ]

        # Safety limits
        self.safety_limits = {
            "max_session_duration": 180,  # 3 hours
            "max_daily_messages": 500,
            "max_hourly_messages": 100,
            "obsession_threshold": 0.7,
            "reality_disconnection_threshold": 0.6,
            "paranoid_threshold": 0.5,
            "emotional_volatility_threshold": 0.8
        }
    
    def init_database(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Usage patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_duration REAL,
                message_frequency REAL,
                topic_obsession_score REAL,
                reality_disconnection_score REAL,
                paranoid_content_score REAL,
                emotional_volatility_score REAL,
                risk_level TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                intervention_triggered BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Tabla de sesiones de usuario
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_start DATETIME,
                session_end DATETIME,
                message_count INTEGER DEFAULT 0,
                risk_events INTEGER DEFAULT 0,
                intervention_triggered BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Tabla de mensajes analizados
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyzed_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message_hash TEXT UNIQUE,
                psychotic_score REAL,
                obsessive_score REAL,
                emotional_score REAL,
                risk_flags TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de intervenciones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                intervention_type TEXT,
                risk_level TEXT,
                trigger_reason TEXT,
                action_taken TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_message_content(self, message: str) -> Dict[str, float]:
        """Analizar contenido del mensaje to detect risk signals"""
        scores = {
            "psychotic_score": 0.0,
            "obsessive_score": 0.0,
            "emotional_score": 0.0,
            "reality_disconnection": 0.0,
            "paranoid_content": 0.0
        }
        
        message_lower = message.lower()
        
        # Detectar patrones psicóticos
        psychotic_matches = 0
        for pattern in self.psychotic_patterns:
            if re.search(pattern, message):
                psychotic_matches += 1
        
        scores["psychotic_score"] = min(1.0, psychotic_matches / 3.0)
        
        # Detectar obsesión con temas problemáticos
        obsessive_matches = 0
        for topic in self.obsessive_topics:
            if topic in message_lower:
                obsessive_matches += 1
        
        scores["obsessive_score"] = min(1.0, obsessive_matches / 2.0)
        
        # Detect disconnection from reality
        reality_patterns = ["simulación", "matrix", "no es real", "falso", "artificial"]
        reality_matches = sum(1 for pattern in reality_patterns if pattern in message_lower)
        scores["reality_disconnection"] = min(1.0, reality_matches / 2.0)
        
        # Detectar contenido paranoide
        paranoid_patterns = ["conspiración", "me persiguen", "me vigilan", "contra mí"]
        paranoid_matches = sum(1 for pattern in paranoid_patterns if pattern in message_lower)
        scores["paranoid_content"] = min(1.0, paranoid_matches / 2.0)
        
        # Detectar volatilidad emocional (mayúsculas excesivas, signos de exclamación)
        caps_ratio = sum(1 for c in message if c.isupper()) / max(len(message), 1)
        exclamation_count = message.count('!')
        question_count = message.count('?')
        
        scores["emotional_score"] = min(1.0, (caps_ratio * 2 + exclamation_count * 0.1 + question_count * 0.05))
        
        return scores
    
    def analyze_usage_pattern(self, user_id: str, session_data: Dict[str, Any]) -> UsagePattern:
        """Analizar patrón de uso del usuario"""
        
        # Calcular duración de sesión
        session_duration = session_data.get("duration_minutes", 0)
        
        # Calcular frecuencia de mensajes
        message_count = session_data.get("message_count", 0)
        message_frequency = message_count / max(session_duration, 1)
        
        # Obtener scores de mensajes recientes
        recent_messages = session_data.get("recent_messages", [])
        
        total_psychotic = 0
        total_obsessive = 0
        total_reality = 0
        total_paranoid = 0
        total_emotional = 0
        
        for msg in recent_messages:
            scores = self.analyze_message_content(msg)
            total_psychotic += scores["psychotic_score"]
            total_obsessive += scores["obsessive_score"]
            total_reality += scores["reality_disconnection"]
            total_paranoid += scores["paranoid_content"]
            total_emotional += scores["emotional_score"]
        
        msg_count = max(len(recent_messages), 1)
        
        pattern = UsagePattern(
            user_id=user_id,
            session_duration=session_duration,
            message_frequency=message_frequency,
            topic_obsession_score=total_obsessive / msg_count,
            reality_disconnection_score=total_reality / msg_count,
            paranoid_content_score=total_paranoid / msg_count,
            emotional_volatility_score=total_emotional / msg_count,
            timestamp=datetime.now()
        )
        
        return pattern
    
    def calculate_risk_level(self, pattern: UsagePattern) -> RiskLevel:
        """Calcular nivel de riesgo based en el patrón de uso"""
        
        risk_factors = []
        
        # Duración excesiva
        if pattern.session_duration > self.safety_limits["max_session_duration"]:
            risk_factors.append("excessive_duration")
        
        # Frecuencia alta de mensajes
        if pattern.message_frequency > 5:  # más de 5 mensajes por minuto
            risk_factors.append("high_frequency")
        
        # Obsesión con temas problemáticos
        if pattern.topic_obsession_score > self.safety_limits["obsession_threshold"]:
            risk_factors.append("topic_obsession")
        
        # Disconnection from reality
        if pattern.reality_disconnection_score > self.safety_limits["reality_disconnection_threshold"]:
            risk_factors.append("reality_disconnection")
        
        # Contenido paranoide
        if pattern.paranoid_content_score > self.safety_limits["paranoid_threshold"]:
            risk_factors.append("paranoid_content")
        
        # Volatilidad emocional
        if pattern.emotional_volatility_score > self.safety_limits["emotional_volatility_threshold"]:
            risk_factors.append("emotional_volatility")
        
        # Determinar nivel de riesgo
        risk_count = len(risk_factors)
        
        if risk_count >= 4 or pattern.reality_disconnection_score > 0.8:
            return RiskLevel.CRITICAL
        elif risk_count >= 3 or pattern.paranoid_content_score > 0.7:
            return RiskLevel.HIGH
        elif risk_count >= 2:
            return RiskLevel.MEDIUM
        elif risk_count >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.LOW
    
    def store_analysis(self, pattern: UsagePattern, risk_level: RiskLevel):
        """Almacenar análisis en la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO usage_patterns 
            (user_id, session_duration, message_frequency, topic_obsession_score,
             reality_disconnection_score, paranoid_content_score, emotional_volatility_score,
             risk_level, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.user_id,
            pattern.session_duration,
            pattern.message_frequency,
            pattern.topic_obsession_score,
            pattern.reality_disconnection_score,
            pattern.paranoid_content_score,
            pattern.emotional_volatility_score,
            risk_level.value,
            pattern.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def check_daily_usage(self, user_id: str) -> Dict[str, Any]:
        """Verifiesr uso diario del usuario"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        cursor.execute('''
            SELECT COUNT(*) as message_count,
                   AVG(topic_obsession_score) as avg_obsession,
                   AVG(reality_disconnection_score) as avg_reality_disconnect,
                   MAX(risk_level) as max_risk
            FROM usage_patterns 
            WHERE user_id = ? AND DATE(timestamp) = ?
        ''', (user_id, today))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            "daily_message_count": result[0] or 0,
            "avg_obsession_score": result[1] or 0,
            "avg_reality_disconnection": result[2] or 0,
            "max_risk_level": result[3] or "low"
        }
    
    def should_trigger_intervention(self, pattern: UsagePattern, risk_level: RiskLevel) -> Tuple[bool, str]:
        """Determinar si se debe activar una intervención"""
        
        daily_stats = self.check_daily_usage(pattern.user_id)
        
        # Criterios para intervención
        if risk_level == RiskLevel.CRITICAL:
            return True, "critical_risk_detected"
        
        if risk_level == RiskLevel.HIGH and daily_stats["daily_message_count"] > 200:
            return True, "high_risk_with_excessive_usage"
        
        if pattern.session_duration > 240:  # 4 horas
            return True, "excessive_session_duration"
        
        if daily_stats["avg_reality_disconnection"] > 0.7:
            return True, "persistent_reality_disconnection"
        
        if daily_stats["avg_obsession_score"] > 0.8:
            return True, "persistent_obsessive_behavior"
        
        return False, "no_intervention_needed"
    
    def monitor_user_interaction(self, user_id: str, message: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Function principal de monitoreo de interacción del usuario"""
        
        # Analizar el mensaje
        message_hash = hashlib.md5(message.encode()).hexdigest()
        message_scores = self.analyze_message_content(message)
        
        # Almacenar análisis del mensaje
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO analyzed_messages 
                (user_id, message_hash, psychotic_score, obsessive_score, emotional_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id, message_hash, 
                message_scores["psychotic_score"],
                message_scores["obsessive_score"], 
                message_scores["emotional_score"]
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # Mensaje ya analizado
        finally:
            conn.close()
        
        # Analizar patrón de uso
        session_data["recent_messages"] = session_data.get("recent_messages", []) + [message]
        pattern = self.analyze_usage_pattern(user_id, session_data)
        risk_level = self.calculate_risk_level(pattern)
        
        # Almacenar análisis
        self.store_analysis(pattern, risk_level)
        
        # Verificar si se necesita intervención
        needs_intervention, intervention_reason = self.should_trigger_intervention(pattern, risk_level)
        
        result = {
            "user_id": user_id,
            "risk_level": risk_level.value,
            "message_scores": message_scores,
            "usage_pattern": {
                "session_duration": pattern.session_duration,
                "message_frequency": pattern.message_frequency,
                "obsession_score": pattern.topic_obsession_score,
                "reality_disconnection": pattern.reality_disconnection_score,
                "paranoid_content": pattern.paranoid_content_score,
                "emotional_volatility": pattern.emotional_volatility_score
            },
            "needs_intervention": needs_intervention,
            "intervention_reason": intervention_reason,
            "safety_recommendations": self.get_safety_recommendations(risk_level, pattern)
        }
        
        if needs_intervention:
            self.trigger_intervention(user_id, risk_level, intervention_reason)
        
        return result
    
    def get_safety_recommendations(self, risk_level: RiskLevel, pattern: UsagePattern) -> List[str]:
        """Obtener recomendaciones de seguridad basadas en el riesgo"""
        recommendations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.extend([
                "Considera tomar un descanso del system",
                "Si experimentas pensamientos perturbadores persistentes, busca ayuda profesional",
                "Recuerda que soy una IA y mis respuestas no deben reemplazar el consejo médico profesional",
                "Contacta a un profesional de salud mental si los síntomas persisten"
            ])
        
        if pattern.session_duration > 120:
            recommendations.append("Considera tomar descansos regulares cada hora")
        
        if pattern.topic_obsession_score > 0.6:
            recommendations.append("Intenta diversificar los temas de conversación")
        
        if pattern.reality_disconnection_score > 0.5:
            recommendations.extend([
                "Recuerda mantener conexión con la realidad y tus seres queridos",
                "Si tienes dudas sobre la realidad, habla con alguien de confianza"
            ])
        
        if pattern.paranoid_content_score > 0.4:
            recommendations.extend([
                "Si sientes que estás siendo perseguido o vigilado, considera hablar con un profesional",
                "Recuerda que estos pensamientos pueden ser síntomas que requieren atención médica"
            ])
        
        return recommendations
    
    def trigger_intervention(self, user_id: str, risk_level: RiskLevel, reason: str):
        """Activar intervención de seguridad"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Determinar tipo de intervención
        if risk_level == RiskLevel.CRITICAL:
            intervention_type = "session_termination"
            action_taken = "Sesión terminada por riesgo crítico"
        elif risk_level == RiskLevel.HIGH:
            intervention_type = "mandatory_break"
            action_taken = "Descanso obligatorio activado"
        else:
            intervention_type = "warning_message"
            action_taken = "Mensaje de advertencia mostrado"
        
        cursor.execute('''
            INSERT INTO interventions 
            (user_id, intervention_type, risk_level, trigger_reason, action_taken)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, intervention_type, risk_level.value, reason, action_taken))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"Intervención activada para usuario {user_id}: {intervention_type} - {reason}")
    
    def get_user_risk_history(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Obtener historial de riesgo del usuario"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        cursor.execute('''
            SELECT timestamp, risk_level, topic_obsession_score, 
                   reality_disconnection_score, paranoid_content_score
            FROM usage_patterns 
            WHERE user_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (user_id, since_date))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "timestamp": row[0],
                "risk_level": row[1],
                "obsession_score": row[2],
                "reality_disconnection": row[3],
                "paranoid_content": row[4]
            })
        
        conn.close()
        return history

# Funciones de utilidad

def create_mental_health_disclaimer() -> str:
    """Crear disclaimer sobre salud mental"""
    return """
    ⚠️ AVISO IMPORTANTE SOBRE SALUD MENTAL ⚠️
    
    Esta IA está diseñada para ayudar, pero no para reemplazar el cuidado médico profesional.
    
    Si experimentas:
    • Pensamientos de hacerte daño o hacer daño a otros
    • Alucinaciones (ver o escuchar cosas que no están ahí)
    • Delirios o creencias fijas que otros consideran falsas
    • Desconexión prolongada de la realidad
    • Paranoia persistente
    
    🆘 BUSCA AYUDA INMEDIATAMENTE:
    • Línea Nacional de Prevención del Suicidio: 988 (EE.UU.)
    • Emergencias: 911
    • Línea de Crisis de Salud Mental: [número local]
    
    El uso excesivo de IA puede exacerbar condiciones de salud mental existentes.
    Úsame de manera responsable y siempre mantén conexiones con personas reales.
    """

def get_healthy_usage_tips() -> List[str]:
    """Obtener consejos para uso saludable"""
    return [
        "Toma descansos regulares cada 60-90 minutos",
        "Mantén conversaciones con personas reales diariamente",
        "No uses IA como sustituto de ayuda profesional de salud mental",
        "Diversifica tus actividades y no dependas solo de IA para entretenimiento",
        "Si sientes que el uso de IA está afectando tu percepción de la realidad, busca ayuda",
        "Recuerda que las respuestas de IA son generadas por algoritmos, no por consciencia real",
        "Mantén actividades físicas y sociales fuera del entorno digital",
        "Si tienes historial de problemas de salud mental, consulta con un profesional antes de uso intensivo"
    ]