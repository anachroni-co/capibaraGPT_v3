#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuración de umbrales para el sistema de routing inteligente.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RoutingThresholds:
    """
    Configuración de umbrales para routing inteligente.
    """
    # Umbrales principales
    complexity_threshold: float = 0.7
    domain_confidence_threshold: float = 0.6
    
    # Umbrales de latencia
    max_latency_20b_ms: int = 2000
    max_latency_120b_ms: int = 10000
    
    # Umbrales de tokens
    max_tokens_20b: int = 8000
    max_tokens_120b: int = 32000
    
    # Umbrales de calidad
    min_quality_score: float = 0.7
    min_success_rate: float = 0.85
    
    # Umbrales de escalación automática
    auto_escalate_complexity: float = 0.9
    auto_escalate_domain_uncertainty: float = 0.3
    
    # Umbrales de fallback
    fallback_to_20b_threshold: float = 0.4
    emergency_fallback_threshold: float = 0.2


class ThresholdManager:
    """
    Gestor de umbrales con persistencia y validación.
    """
    
    def __init__(self, config_file: str = "backend/data/thresholds.json"):
        """
        Inicializa el gestor de umbrales.
        
        Args:
            config_file: Archivo de configuración JSON
        """
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Cargar configuración
        self.thresholds = self._load_thresholds()
        
        logger.info("ThresholdManager inicializado")
    
    def _load_thresholds(self) -> RoutingThresholds:
        """Carga umbrales desde archivo o usa valores por defecto."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validar y crear instancia
                thresholds = RoutingThresholds(**data)
                logger.info(f"Umbrales cargados desde {self.config_file}")
                return thresholds
            else:
                # Crear archivo con valores por defecto
                thresholds = RoutingThresholds()
                self._save_thresholds(thresholds)
                logger.info(f"Umbrales por defecto creados en {self.config_file}")
                return thresholds
                
        except Exception as e:
            logger.error(f"Error cargando umbrales: {e}")
            return RoutingThresholds()
    
    def _save_thresholds(self, thresholds: RoutingThresholds):
        """Guarda umbrales en archivo."""
        try:
            data = {
                'complexity_threshold': thresholds.complexity_threshold,
                'domain_confidence_threshold': thresholds.domain_confidence_threshold,
                'max_latency_20b_ms': thresholds.max_latency_20b_ms,
                'max_latency_120b_ms': thresholds.max_latency_120b_ms,
                'max_tokens_20b': thresholds.max_tokens_20b,
                'max_tokens_120b': thresholds.max_tokens_120b,
                'min_quality_score': thresholds.min_quality_score,
                'min_success_rate': thresholds.min_success_rate,
                'auto_escalate_complexity': thresholds.auto_escalate_complexity,
                'auto_escalate_domain_uncertainty': thresholds.auto_escalate_domain_uncertainty,
                'fallback_to_20b_threshold': thresholds.fallback_to_20b_threshold,
                'emergency_fallback_threshold': thresholds.emergency_fallback_threshold
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Umbrales guardados en {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error guardando umbrales: {e}")
    
    def get_thresholds(self) -> RoutingThresholds:
        """Retorna los umbrales actuales."""
        return self.thresholds
    
    def update_threshold(self, name: str, value: float) -> bool:
        """
        Actualiza un umbral específico.
        
        Args:
            name: Nombre del umbral
            value: Nuevo valor
            
        Returns:
            True si se actualizó exitosamente
        """
        try:
            # Validar nombre
            if not hasattr(self.thresholds, name):
                logger.error(f"Umbral '{name}' no existe")
                return False
            
            # Validar valor
            if not self._validate_threshold_value(name, value):
                logger.error(f"Valor {value} inválido para umbral '{name}'")
                return False
            
            # Actualizar
            setattr(self.thresholds, name, value)
            self._save_thresholds(self.thresholds)
            
            logger.info(f"Umbral '{name}' actualizado a {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando umbral '{name}': {e}")
            return False
    
    def _validate_threshold_value(self, name: str, value: float) -> bool:
        """Valida que un valor de umbral sea válido."""
        try:
            # Rangos válidos por tipo de umbral
            if name.endswith('_threshold'):
                return 0.0 <= value <= 1.0
            elif name.endswith('_ms'):
                return 0 <= value <= 60000  # Max 1 minuto
            elif name.endswith('_tokens'):
                return 1000 <= value <= 100000  # Rango razonable
            elif name.endswith('_rate') or name.endswith('_score'):
                return 0.0 <= value <= 1.0
            else:
                return True  # Otros tipos sin validación específica
                
        except Exception:
            return False
    
    def update_multiple_thresholds(self, updates: Dict[str, float]) -> Dict[str, bool]:
        """
        Actualiza múltiples umbrales.
        
        Args:
            updates: Diccionario {nombre_umbral: valor}
            
        Returns:
            Diccionario con resultados de cada actualización
        """
        results = {}
        
        for name, value in updates.items():
            results[name] = self.update_threshold(name, value)
        
        return results
    
    def reset_to_defaults(self):
        """Resetea todos los umbrales a valores por defecto."""
        try:
            self.thresholds = RoutingThresholds()
            self._save_thresholds(self.thresholds)
            logger.info("Umbrales reseteados a valores por defecto")
        except Exception as e:
            logger.error(f"Error reseteando umbrales: {e}")
    
    def get_threshold_info(self) -> Dict[str, Any]:
        """Retorna información detallada de todos los umbrales."""
        return {
            'complexity_threshold': {
                'value': self.thresholds.complexity_threshold,
                'description': 'Umbral de complejidad para escalar a 120B',
                'range': '0.0 - 1.0',
                'default': 0.7
            },
            'domain_confidence_threshold': {
                'value': self.thresholds.domain_confidence_threshold,
                'description': 'Umbral de confianza de dominio para escalar a 120B',
                'range': '0.0 - 1.0',
                'default': 0.6
            },
            'max_latency_20b_ms': {
                'value': self.thresholds.max_latency_20b_ms,
                'description': 'Latencia máxima para modelo 20B',
                'range': '0 - 60000 ms',
                'default': 2000
            },
            'max_latency_120b_ms': {
                'value': self.thresholds.max_latency_120b_ms,
                'description': 'Latencia máxima para modelo 120B',
                'range': '0 - 60000 ms',
                'default': 10000
            },
            'max_tokens_20b': {
                'value': self.thresholds.max_tokens_20b,
                'description': 'Máximo tokens para modelo 20B',
                'range': '1000 - 100000',
                'default': 8000
            },
            'max_tokens_120b': {
                'value': self.thresholds.max_tokens_120b,
                'description': 'Máximo tokens para modelo 120B',
                'range': '1000 - 100000',
                'default': 32000
            },
            'min_quality_score': {
                'value': self.thresholds.min_quality_score,
                'description': 'Puntuación mínima de calidad',
                'range': '0.0 - 1.0',
                'default': 0.7
            },
            'min_success_rate': {
                'value': self.thresholds.min_success_rate,
                'description': 'Tasa mínima de éxito',
                'range': '0.0 - 1.0',
                'default': 0.85
            },
            'auto_escalate_complexity': {
                'value': self.thresholds.auto_escalate_complexity,
                'description': 'Complejidad para escalación automática',
                'range': '0.0 - 1.0',
                'default': 0.9
            },
            'auto_escalate_domain_uncertainty': {
                'value': self.thresholds.auto_escalate_domain_uncertainty,
                'description': 'Incertidumbre de dominio para escalación automática',
                'range': '0.0 - 1.0',
                'default': 0.3
            },
            'fallback_to_20b_threshold': {
                'value': self.thresholds.fallback_to_20b_threshold,
                'description': 'Umbral para fallback a 20B',
                'range': '0.0 - 1.0',
                'default': 0.4
            },
            'emergency_fallback_threshold': {
                'value': self.thresholds.emergency_fallback_threshold,
                'description': 'Umbral de emergencia para fallback',
                'range': '0.0 - 1.0',
                'default': 0.2
            }
        }


class AdaptiveThresholds:
    """
    Sistema de umbrales adaptativos que se ajustan basado en performance.
    """
    
    def __init__(self, threshold_manager: ThresholdManager):
        """
        Inicializa el sistema adaptativo.
        
        Args:
            threshold_manager: Instancia de ThresholdManager
        """
        self.threshold_manager = threshold_manager
        self.performance_history = []
        self.adaptation_rate = 0.1  # Qué tan rápido se adapta
        
        logger.info("AdaptiveThresholds inicializado")
    
    def record_performance(self, complexity: float, domain_conf: float, 
                          success: bool, latency_ms: int, model_used: str):
        """
        Registra performance de una decisión de routing.
        
        Args:
            complexity: Complejidad de la query
            domain_conf: Confianza de dominio
            success: Si la respuesta fue exitosa
            latency_ms: Latencia en milisegundos
            model_used: Modelo usado ('20B' o '120B')
        """
        try:
            record = {
                'complexity': complexity,
                'domain_conf': domain_conf,
                'success': success,
                'latency_ms': latency_ms,
                'model_used': model_used,
                'timestamp': self._get_timestamp()
            }
            
            self.performance_history.append(record)
            
            # Mantener solo últimos 1000 registros
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # Adaptar umbrales si hay suficientes datos
            if len(self.performance_history) >= 100:
                self._adapt_thresholds()
                
        except Exception as e:
            logger.error(f"Error registrando performance: {e}")
    
    def _get_timestamp(self) -> float:
        """Obtiene timestamp actual."""
        import time
        return time.time()
    
    def _adapt_thresholds(self):
        """Adapta umbrales basado en performance histórica."""
        try:
            # Analizar performance reciente (últimos 100 registros)
            recent_history = self.performance_history[-100:]
            
            # Calcular métricas
            success_rate_20b = self._calculate_success_rate(recent_history, '20B')
            success_rate_120b = self._calculate_success_rate(recent_history, '120B')
            avg_latency_20b = self._calculate_avg_latency(recent_history, '20B')
            avg_latency_120b = self._calculate_avg_latency(recent_history, '120B')
            
            # Ajustar umbrales si es necesario
            self._adjust_complexity_threshold(success_rate_20b, success_rate_120b)
            self._adjust_domain_confidence_threshold(success_rate_20b, success_rate_120b)
            
            logger.info(f"Umbrales adaptados - 20B: {success_rate_20b:.2f}, "
                       f"120B: {success_rate_120b:.2f}")
            
        except Exception as e:
            logger.error(f"Error adaptando umbrales: {e}")
    
    def _calculate_success_rate(self, history: list, model: str) -> float:
        """Calcula tasa de éxito para un modelo."""
        model_records = [r for r in history if r['model_used'] == model]
        if not model_records:
            return 0.5  # Valor por defecto
        
        successes = sum(1 for r in model_records if r['success'])
        return successes / len(model_records)
    
    def _calculate_avg_latency(self, history: list, model: str) -> float:
        """Calcula latencia promedio para un modelo."""
        model_records = [r for r in history if r['model_used'] == model]
        if not model_records:
            return 1000.0  # Valor por defecto
        
        return sum(r['latency_ms'] for r in model_records) / len(model_records)
    
    def _adjust_complexity_threshold(self, success_rate_20b: float, success_rate_120b: float):
        """Ajusta umbral de complejidad."""
        try:
            current_threshold = self.threshold_manager.thresholds.complexity_threshold
            
            # Si 20B tiene mejor performance, aumentar umbral (usar más 20B)
            if success_rate_20b > success_rate_120b + 0.1:
                new_threshold = min(1.0, current_threshold + self.adaptation_rate)
            # Si 120B tiene mejor performance, disminuir umbral (usar más 120B)
            elif success_rate_120b > success_rate_20b + 0.1:
                new_threshold = max(0.0, current_threshold - self.adaptation_rate)
            else:
                return  # No ajustar
            
            self.threshold_manager.update_threshold('complexity_threshold', new_threshold)
            
        except Exception as e:
            logger.error(f"Error ajustando umbral de complejidad: {e}")
    
    def _adjust_domain_confidence_threshold(self, success_rate_20b: float, success_rate_120b: float):
        """Ajusta umbral de confianza de dominio."""
        try:
            current_threshold = self.threshold_manager.thresholds.domain_confidence_threshold
            
            # Lógica similar a complejidad
            if success_rate_20b > success_rate_120b + 0.1:
                new_threshold = min(1.0, current_threshold + self.adaptation_rate)
            elif success_rate_120b > success_rate_20b + 0.1:
                new_threshold = max(0.0, current_threshold - self.adaptation_rate)
            else:
                return
            
            self.threshold_manager.update_threshold('domain_confidence_threshold', new_threshold)
            
        except Exception as e:
            logger.error(f"Error ajustando umbral de confianza: {e}")


# Funciones de conveniencia
def create_threshold_manager(config_file: str = None) -> ThresholdManager:
    """Crea una instancia de ThresholdManager."""
    if config_file is None:
        config_file = "backend/data/thresholds.json"
    return ThresholdManager(config_file)


def create_adaptive_thresholds(threshold_manager: ThresholdManager = None) -> AdaptiveThresholds:
    """Crea una instancia de AdaptiveThresholds."""
    if threshold_manager is None:
        threshold_manager = create_threshold_manager()
    return AdaptiveThresholds(threshold_manager)


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    # Crear gestor de umbrales
    manager = create_threshold_manager()
    
    # Mostrar información
    info = manager.get_threshold_info()
    print("=== Umbrales Actuales ===")
    for name, details in info.items():
        print(f"{name}: {details['value']} ({details['description']})")
    
    # Test adaptativo
    adaptive = create_adaptive_thresholds(manager)
    
    # Simular algunos registros de performance
    for i in range(10):
        adaptive.record_performance(
            complexity=0.5 + i * 0.05,
            domain_conf=0.6 + i * 0.03,
            success=True,
            latency_ms=1000 + i * 100,
            model_used='20B' if i % 2 == 0 else '120B'
        )
    
    print("\n=== Umbrales Después de Adaptación ===")
    updated_info = manager.get_threshold_info()
    for name, details in updated_info.items():
        print(f"{name}: {details['value']}")
