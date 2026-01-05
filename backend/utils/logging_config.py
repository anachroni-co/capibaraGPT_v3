#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de logging centralizado para Capibara6.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class Capibara6Logger:
    """
    Sistema de logging centralizado para Capibara6.
    """
    
    def __init__(self, log_dir: str = "backend/logs", 
                 log_level: str = "INFO",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        Inicializa el sistema de logging.
        
        Args:
            log_dir: Directorio para archivos de log
            log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_file_size: Tamaño máximo de archivo de log en bytes
            backup_count: Número de archivos de backup
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_level = getattr(logging, log_level.upper())
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Configurar logging
        self._setup_logging()
        
        # Logger principal
        self.logger = logging.getLogger('capibara6')
        
        # Loggers específicos
        self._setup_component_loggers()
        
        self.logger.info("Sistema de logging Capibara6 inicializado")
    
    def _setup_logging(self):
        """Configura el sistema de logging."""
        # Formato de logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        
        # Handler para archivo principal
        main_log_file = self.log_dir / "capibara6.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        
        # Handler para errores
        error_log_file = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        # Configurar logger raíz
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
    
    def _setup_component_loggers(self):
        """Configura loggers específicos para componentes."""
        components = [
            'capibara6.router',
            'capibara6.cag',
            'capibara6.rag',
            'capibara6.ace',
            'capibara6.execution',
            'capibara6.agents',
            'capibara6.metadata'
        ]
        
        for component in components:
            logger = logging.getLogger(component)
            
            # Archivo específico para cada componente
            component_file = self.log_dir / f"{component.split('.')[-1]}.log"
            handler = logging.handlers.RotatingFileHandler(
                component_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            handler.setLevel(self.log_level)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Obtiene un logger específico.
        
        Args:
            name: Nombre del logger
            
        Returns:
            Logger configurado
        """
        return logging.getLogger(f'capibara6.{name}')
    
    def log_decision(self, component: str, decision: Dict[str, Any]):
        """
        Registra una decisión del sistema.
        
        Args:
            component: Componente que tomó la decisión
            decision: Datos de la decisión
        """
        try:
            logger = self.get_logger(component)
            
            # Agregar timestamp
            decision['timestamp'] = datetime.now().isoformat()
            
            # Log como JSON para fácil parsing
            logger.info(f"DECISION: {json.dumps(decision, ensure_ascii=False)}")
            
        except Exception as e:
            self.logger.error(f"Error logging decision: {e}")
    
    def log_performance(self, component: str, metrics: Dict[str, Any]):
        """
        Registra métricas de performance.
        
        Args:
            component: Componente que generó las métricas
            metrics: Métricas de performance
        """
        try:
            logger = self.get_logger(component)
            
            # Agregar timestamp
            metrics['timestamp'] = datetime.now().isoformat()
            
            # Log como JSON
            logger.info(f"PERFORMANCE: {json.dumps(metrics, ensure_ascii=False)}")
            
        except Exception as e:
            self.logger.error(f"Error logging performance: {e}")
    
    def log_error(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """
        Registra un error con contexto.
        
        Args:
            component: Componente donde ocurrió el error
            error: Excepción
            context: Contexto adicional
        """
        try:
            logger = self.get_logger(component)
            
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {}
            }
            
            logger.error(f"ERROR: {json.dumps(error_data, ensure_ascii=False)}")
            
        except Exception as e:
            self.logger.error(f"Error logging error: {e}")
    
    def log_routing_decision(self, query_hash: str, complexity: float, 
                           domain_conf: float, decision: str, model_used: str,
                           latency_ms: int, success: bool):
        """
        Registra una decisión de routing.
        
        Args:
            query_hash: Hash de la query
            complexity: Complejidad calculada
            domain_conf: Confianza de dominio
            decision: Decisión tomada
            model_used: Modelo usado
            latency_ms: Latencia en milisegundos
            success: Si fue exitoso
        """
        self.log_decision('router', {
            'query_hash': query_hash,
            'complexity': complexity,
            'domain_confidence': domain_conf,
            'decision': decision,
            'model_used': model_used,
            'latency_ms': latency_ms,
            'success': success
        })
    
    def log_rag_performance(self, query_hash: str, search_type: str, 
                          latency_ms: int, results_count: int, relevance_score: float):
        """
        Registra performance de RAG.
        
        Args:
            query_hash: Hash de la query
            search_type: Tipo de búsqueda (mini/full)
            latency_ms: Latencia en milisegundos
            results_count: Número de resultados
            relevance_score: Puntuación de relevancia
        """
        self.log_performance('rag', {
            'query_hash': query_hash,
            'search_type': search_type,
            'latency_ms': latency_ms,
            'results_count': results_count,
            'relevance_score': relevance_score
        })
    
    def log_ace_reflection(self, query_hash: str, quality_score: float,
                          should_add_to_playbook: bool, insights: str):
        """
        Registra reflexión de ACE.
        
        Args:
            query_hash: Hash de la query
            quality_score: Puntuación de calidad
            should_add_to_playbook: Si debe agregarse al playbook
            insights: Insights generados
        """
        self.log_decision('ace', {
            'query_hash': query_hash,
            'quality_score': quality_score,
            'should_add_to_playbook': should_add_to_playbook,
            'insights': insights
        })
    
    def log_execution_result(self, code_hash: str, language: str, 
                           success: bool, execution_time_ms: int,
                           error_type: str = None, correction_attempts: int = 0):
        """
        Registra resultado de ejecución E2B.
        
        Args:
            code_hash: Hash del código
            language: Lenguaje de programación
            success: Si fue exitoso
            execution_time_ms: Tiempo de ejecución
            error_type: Tipo de error (si falló)
            correction_attempts: Intentos de corrección
        """
        self.log_performance('execution', {
            'code_hash': code_hash,
            'language': language,
            'success': success,
            'execution_time_ms': execution_time_ms,
            'error_type': error_type,
            'correction_attempts': correction_attempts
        })
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de los logs."""
        try:
            stats = {
                'log_directory': str(self.log_dir),
                'log_files': [],
                'total_size_mb': 0
            }
            
            # Analizar archivos de log
            for log_file in self.log_dir.glob("*.log"):
                file_stats = {
                    'name': log_file.name,
                    'size_mb': log_file.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                }
                stats['log_files'].append(file_stats)
                stats['total_size_mb'] += file_stats['size_mb']
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting log stats: {e}")
            return {}
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """
        Limpia logs antiguos.
        
        Args:
            days_to_keep: Días de logs a mantener
        """
        try:
            import time
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
            
            cleaned_count = 0
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    cleaned_count += 1
            
            self.logger.info(f"Limpiados {cleaned_count} archivos de log antiguos")
            
        except Exception as e:
            self.logger.error(f"Error limpiando logs: {e}")


# Instancia global del logger
_logger_instance: Optional[Capibara6Logger] = None


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger específico.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = Capibara6Logger()
    
    return _logger_instance.get_logger(name)


def get_main_logger() -> Capibara6Logger:
    """
    Obtiene la instancia principal del logger.
    
    Returns:
        Instancia de Capibara6Logger
    """
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = Capibara6Logger()
    
    return _logger_instance


def setup_logging(log_dir: str = "backend/logs", log_level: str = "INFO"):
    """
    Configura el sistema de logging.
    
    Args:
        log_dir: Directorio para logs
        log_level: Nivel de logging
    """
    global _logger_instance
    
    _logger_instance = Capibara6Logger(log_dir, log_level)
    return _logger_instance


if __name__ == "__main__":
    # Test del sistema de logging
    logger = setup_logging()
    
    # Test diferentes tipos de log
    router_logger = get_logger('router')
    router_logger.info("Test router logger")
    
    # Test logging de decisión
    logger.log_routing_decision(
        query_hash="abc123",
        complexity=0.8,
        domain_conf=0.6,
        decision="escalate",
        model_used="120B",
        latency_ms=1500,
        success=True
    )
    
    # Test stats
    stats = logger.get_log_stats()
    print("Log stats:", stats)
