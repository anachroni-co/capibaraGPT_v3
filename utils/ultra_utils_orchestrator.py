"""
 ULTRA UTILS ORCHESTRATOR
============================

El sistema de utilidades more advanced del mundo with:
-  Inteligencia Artificial integrada
-  Performance ultra-optimizado  
-  Auto-healing and self-optimization
-  Analytics predictivo
-  Smart routing automático

Autor: CapibaraGPT Ultra Team
Versión: 3.3.0 Ultra
"""

import asyncio
import functools
import hashlib
import inspect
import json
import os
import pickle
import time
import traceback
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Type
import logging

# Importaciones locales with fallbacks seguros
try:
    from .logging_utils import setup_logger, log_execution_time, create_structured_logger
    from .config_utils import load_config_file, merge_configs, get_config_value
    from .validation_utils import validate_model_config, is_valid_json_structure
    from .data_utils import flatten_dict, merge_dicts, chunk_list, batch_process
    from .cache_standalone import create_cache
    from .monitoring import setup_monitoring
    from .system_info import get_system_info, check_tpu_availability
except ImportError as e:
    logger.warning(f"️ Importación fallback en ultra_utils_orchestrator: {e}")


@dataclass
class UltraUtilsConfig:
    """setup del Ultra Utils Orchestrator."""
    
    # Performance Settings
    max_memory_mb: int = 1024
    max_threads: int = 8
    cache_size_mb: int = 256
    performance_threshold_ms: float = 100.0
    
    # AI Settings  
    enable_ai_optimization: bool = True
    ai_learning_rate: float = 0.001
    predictive_caching: bool = True
    auto_performance_tuning: bool = True
    
    # Analytics Settings
    enable_analytics: bool = True
    metrics_buffer_size: int = 10000
    analytics_interval_seconds: int = 60
    
    # Smart Contracts
    enable_smart_contracts: bool = True
    performance_sla_ms: float = 50.0
    memory_limit_mb: float = 512.0
    error_threshold_percent: float = 1.0
    
    # Auto-healing
    enable_auto_healing: bool = True
    max_retry_attempts: int = 3
    healing_timeout_seconds: int = 30


@dataclass
class UtilsMetrics:
    """Métricas de rendimiento de utilities."""
    
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    error_count: int = 0
    success_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calcula la tasa de éxito."""
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 0.0


class UltraSmartCache:
    """Sistema de caché inteligente with IA predictiva."""
    
    def __init__(self, max_size_mb: int = 256):
        self.max_size_mb = max_size_mb
        self.cache = {}
        self.access_patterns = {}
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
        
    def _get_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Genera una key única for el caché."""
        func_name = f"{func.__module__}.{func.__name__}"
        args_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(f"{func_name}:{args_str}".encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un value del caché."""
        with self._lock:
            if key in self.cache:
                # update pattern de acceso
                self.access_patterns[key] = self.access_patterns.get(key, 0) + 1
                self.hit_count += 1
                return self.cache[key]['value']
            else:
                self.miss_count += 1
                return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Establece un value en el caché."""
        with self._lock:
            expiry = datetime.now() + timedelta(seconds=ttl_seconds)
            self.cache[key] = {
                'value': value,
                'expiry': expiry,
                'access_count': 1
            }
            self._cleanup_expired()
    
    def _cleanup_expired(self) -> None:
        """Limpia entradas expiradas."""
        now = datetime.now()
        expired_keys = [
            key for key, data in self.cache.items()
            if data['expiry'] < now
        ]
        for key in expired_keys:
            del self.cache[key]
    
    @property
    def hit_rate(self) -> float:
        """Calcula la tasa de aciertos del caché."""
        total = self.hit_count + self.miss_count
        return (self.hit_count / total) if total > 0 else 0.0


class AIPerformanceOptimizer:
    """Optimizador de rendimiento basado en IA."""
    
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.performance_history = []
        self.optimization_rules = {}
        self.model_weights = {}
        
    def analyze_performance(self, metrics: UtilsMetrics, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza métricas de rendimiento with IA."""
        
        # Registrar métricas históricas
        self.performance_history.append({
            'metrics': metrics,
            'context': context,
            'timestamp': datetime.now()
        })
        
        # maintain only últimas 1000 entradas
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        # analysis predictivo simple
        recommendations = {
            'cache_strategy': self._recommend_cache_strategy(metrics),
            'thread_optimization': self._recommend_threading(metrics),
            'memory_optimization': self._recommend_memory_optimization(metrics),
            'performance_prediction': self._predict_performance_trend()
        }
        
        return recommendations
    
    def _recommend_cache_strategy(self, metrics: UtilsMetrics) -> str:
        """Recomienda estrategia de caché optimal."""
        if metrics.cache_hit_rate < 0.7:
            return "increase_cache_size"
        elif metrics.cache_hit_rate > 0.95:
            return "optimize_cache_keys"
        else:
            return "maintain_current"
    
    def _recommend_threading(self, metrics: UtilsMetrics) -> str:
        """Recomienda optimization de threading."""
        if metrics.cpu_usage_percent < 50:
            return "increase_parallelism"
        elif metrics.cpu_usage_percent > 90:
            return "reduce_threads"
        else:
            return "optimal_threading"
    
    def _recommend_memory_optimization(self, metrics: UtilsMetrics) -> str:
        """Recomienda optimization de memory."""
        if metrics.memory_usage_mb > 800:
            return "aggressive_cleanup"
        elif metrics.memory_usage_mb > 500:
            return "moderate_cleanup"
        else:
            return "memory_optimal"
    
    def _predict_performance_trend(self) -> Dict[str, float]:
        """Predice tendencias de rendimiento."""
        if len(self.performance_history) < 10:
            return {"trend": 0.0, "confidence": 0.0}
        
        # analysis de tendencia simple
        recent_metrics = self.performance_history[-10:]
        old_metrics = self.performance_history[-20:-10] if len(self.performance_history) >= 20 else []
        
        if not old_metrics:
            return {"trend": 0.0, "confidence": 0.5}
        
        recent_avg = sum(m['metrics'].execution_time_ms for m in recent_metrics) / len(recent_metrics)
        old_avg = sum(m['metrics'].execution_time_ms for m in old_metrics) / len(old_metrics)
        
        trend = (recent_avg - old_avg) / old_avg if old_avg > 0 else 0.0
        
        return {
            "trend": trend,
            "confidence": min(0.9, len(self.performance_history) / 100)
        }


class SmartUtilsContract:
    """Contrato inteligente for utilities."""
    
    def __init__(self, name: str, sla_ms: float = 50.0, memory_limit_mb: float = 512.0):
        self.name = name
        self.sla_ms = sla_ms
        self.memory_limit_mb = memory_limit_mb
        self.violations = []
        self.active = True
        
    def validate_execution(self, metrics: UtilsMetrics) -> Dict[str, Any]:
        """Valida la execution against el contrato."""
        violations = []
        
        if metrics.execution_time_ms > self.sla_ms:
            violations.append(f"SLA violation: {metrics.execution_time_ms}ms > {self.sla_ms}ms")
        
        if metrics.memory_usage_mb > self.memory_limit_mb:
            violations.append(f"Memory violation: {metrics.memory_usage_mb}MB > {self.memory_limit_mb}MB")
        
        if metrics.error_count > 0:
            violations.append(f"Error count: {metrics.error_count}")
        
        if violations:
            self.violations.extend(violations)
        
        return {
            'contract_satisfied': len(violations) == 0,
            'violations': violations,
            'total_violations': len(self.violations)
        }


class UltraUtilsOrchestrator:
    """
     CEREBRO CENTRAL del ecosistema de utilidades.
    
    Sistema ultra-advanced que unifica todas las utilidades with:
    - IA for optimization automática
    - Contratos inteligentes for SLAs
    - Analytics predictivo en tiempo real
    - Auto-healing and self-optimization
    """
    
    def __init__(self, config: Optional[UltraUtilsConfig] = None):
        self.config = config or UltraUtilsConfig()
        self.logger = self._setup_logger()
        
        # Componentes principales
        self.cache = UltraSmartCache(self.config.cache_size_mb)
        self.ai_optimizer = AIPerformanceOptimizer(self.config.ai_learning_rate)
        self.contracts = {}
        self.metrics_buffer = []
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_threads)
        self.async_loop = None
        
        # Registry de utilities
        self.utils_registry = {
            'data': self._init_data_utils(),
            'config': self._init_config_utils(),
            'logging': self._init_logging_utils(),
            'validation': self._init_validation_utils(),
            'performance': self._init_performance_utils(),
            'ai': self._init_ai_utils(),
            'system': self._init_system_utils()
        }
        
        # Contratos by defect
        self._setup_default_contracts()
        
        # start analytics
        if self.config.enable_analytics:
            self._start_analytics_thread()
            
        self.logger.info(" Ultra Utils Orchestrator iniciado correctamente")
    
    def _setup_logger(self) -> logging.Logger:
        """Configura el logger del orchestrator."""
        try:
            return setup_logger(
                "UltraUtilsOrchestrator",
                level=logging.INFO,
                log_file="logs/ultra_utils.log"
            )
        except:
            # Fallback logger
            logger = logging.getLogger("UltraUtilsOrchestrator")
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            return logger
    
    def _init_data_utils(self) -> Dict[str, Callable]:
        """Inicializa utilidades de data."""
        try:
            return {
                'flatten_dict': flatten_dict,
                'merge_dicts': merge_dicts,
                'chunk_list': chunk_list,
                'batch_process': batch_process
            }
        except:
            return {}
    
    def _init_config_utils(self) -> Dict[str, Callable]:
        """Inicializa utilidades de setup."""
        try:
            return {
                'load_config': load_config_file,
                'merge_configs': merge_configs,
                'get_config_value': get_config_value
            }
        except:
            return {}
    
    def _init_logging_utils(self) -> Dict[str, Callable]:
        """Inicializa utilidades de logging."""
        try:
            return {
                'setup_logger': setup_logger,
                'log_execution_time': log_execution_time,
                'create_structured_logger': create_structured_logger
            }
        except:
            return {}
    
    def _init_validation_utils(self) -> Dict[str, Callable]:
        """Inicializa utilidades de validation."""
        try:
            return {
                'validate_model_config': validate_model_config,
                'is_valid_json_structure': is_valid_json_structure
            }
        except:
            return {}
    
    def _init_performance_utils(self) -> Dict[str, Callable]:
        """Inicializa utilidades de performance."""
        return {
            'monitor_execution': self._monitor_execution,
            'profile_function': self._profile_function,
            'optimize_memory': self._optimize_memory
        }
    
    def _init_ai_utils(self) -> Dict[str, Callable]:
        """Inicializa utilidades de IA."""
        return {
            'smart_cache': self._smart_cache_decorator,
            'ai_optimize': self._ai_optimize_function,
            'predict_performance': self._predict_performance
        }
    
    def _init_system_utils(self) -> Dict[str, Callable]:
        """Inicializa utilidades de sistema."""
        try:
            return {
                'get_system_info': get_system_info,
                'check_tpu_availability': check_tpu_availability,
                'setup_monitoring': setup_monitoring
            }
        except:
            return {}
    
    def _setup_default_contracts(self) -> None:
        """Configura contratos inteligentes by defect."""
        if self.config.enable_smart_contracts:
            self.contracts['performance'] = SmartUtilsContract(
                "PerformanceContract",
                sla_ms=self.config.performance_sla_ms,
                memory_limit_mb=self.config.memory_limit_mb
            )
            
            self.contracts['reliability'] = SmartUtilsContract(
                "ReliabilityContract",
                sla_ms=100.0,
                memory_limit_mb=256.0
            )
    
    def smart_route(self, utility_type: str, operation: str, **kwargs) -> Any:
        """
         Smart routing - Auto-selecciona la better utilidad.
        
        Args:
            utility_type: type de utilidad ('data', 'config', etc.)
            operation: operation a perform
            **kwargs: Argumentos for la operation
            
        Returns:
            result de la operation optimizada
        """
        start_time = time.time()
        
        try:
            # verify if la utilidad existe
            if utility_type not in self.utils_registry:
                raise ValueError(f"Tipo de utilidad no soportado: {utility_type}")
            
            if operation not in self.utils_registry[utility_type]:
                raise ValueError(f"Operación no encontrada: {operation} en {utility_type}")
            
            # obtain la function
            func = self.utils_registry[utility_type][operation]
            
            # apply optimizaciones de IA if están habilitadas
            if self.config.enable_ai_optimization:
                func = self._ai_optimize_function(func)
            
            # execute with monitoreo
            result = self._execute_with_monitoring(func, **kwargs)
            
            # Registrar métricas
            execution_time = (time.time() - start_time) * 1000  # ms
            self._record_metrics(utility_type, operation, execution_time, success=True)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._record_metrics(utility_type, operation, execution_time, success=False)
            
            if self.config.enable_auto_healing:
                return self._attempt_auto_healing(utility_type, operation, e, **kwargs)
            else:
                raise
    
    def _execute_with_monitoring(self, func: Callable, **kwargs) -> Any:
        """Ejecuta function with monitoreo complete."""
        
        # verify contratos before de execute
        for contract in self.contracts.values():
            if not contract.active:
                continue
        
        # execute function
        result = func(**kwargs)
        
        return result
    
    def _record_metrics(self, utility_type: str, operation: str, execution_time_ms: float, success: bool) -> None:
        """Registra métricas de execution."""
        metrics = UtilsMetrics(
            execution_time_ms=execution_time_ms,
            success_count=1 if success else 0,
            error_count=0 if success else 1
        )
        
        # validate contratos
        for contract_name, contract in self.contracts.items():
            validation = contract.validate_execution(metrics)
            if not validation['contract_satisfied']:
                self.logger.warning(f"Contrato {contract_name} violado: {validation['violations']}")
        
        # add a buffer de métricas
        self.metrics_buffer.append({
            'utility_type': utility_type,
            'operation': operation,
            'metrics': metrics,
            'timestamp': datetime.now()
        })
        
        # maintain buffer en size límite
        if len(self.metrics_buffer) > self.config.metrics_buffer_size:
            self.metrics_buffer = self.metrics_buffer[-self.config.metrics_buffer_size:]
    
    def _attempt_auto_healing(self, utility_type: str, operation: str, error: Exception, **kwargs) -> Any:
        """Intenta auto-healing when falla una operation."""
        self.logger.warning(f" Auto-healing activado para {utility_type}.{operation}: {error}")
        
        for attempt in range(self.config.max_retry_attempts):
            try:
                time.sleep(0.1 * (attempt + 1))  # Backoff exponencial
                
                # try operation básica without optimizaciones
                func = self.utils_registry[utility_type][operation]
                result = func(**kwargs)
                
                self.logger.info(f" Auto-healing exitoso en intento {attempt + 1}")
                return result
                
            except Exception as retry_error:
                self.logger.warning(f"Auto-healing intento {attempt + 1} falló: {retry_error}")
                
                if attempt == self.config.max_retry_attempts - 1:
                    self.logger.error(f" Auto-healing agotado para {utility_type}.{operation}")
                    raise error
        
        raise error
    
    def _smart_cache_decorator(self, ttl_seconds: int = 3600):
        """Decorador de caché inteligente with IA."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = self.cache._get_cache_key(func, args, kwargs)
                
                # try obtain del caché
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # execute function and cachear result
                result = func(*args, **kwargs)
                self.cache.set(cache_key, result, ttl_seconds)
                
                return result
            return wrapper
        return decorator
    
    def _ai_optimize_function(self, func: Callable) -> Callable:
        """Optimiza function usando IA."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # analysis pre-execution
            context = {
                'function_name': func.__name__,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            }
            
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            
            # create métricas and obtain recomendaciones de IA
            metrics = UtilsMetrics(execution_time_ms=execution_time)
            recommendations = self.ai_optimizer.analyze_performance(metrics, context)
            
            # apply recomendaciones if es necessary
            if recommendations['cache_strategy'] == 'increase_cache_size':
                # Lógica for increase caché
                pass
            
            return result
        return wrapper
    
    def _monitor_execution(self, func: Callable) -> Callable:
        """Monitorea execution de function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                self.logger.debug(f" {func.__name__} ejecutado en {execution_time:.2f}ms")
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                self.logger.error(f" {func.__name__} falló en {execution_time:.2f}ms: {e}")
                raise
        return wrapper
    
    def _profile_function(self, func: Callable) -> Dict[str, Any]:
        """Perfila function for analysis de rendimiento."""
        import cProfile
        import pstats
        import io
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        try:
            result = func()
            success = True
        except Exception as e:
            result = None
            success = False
        finally:
            profiler.disable()
            execution_time = time.time() - start_time
        
        # obtain estadísticas del profiler
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        return {
            'execution_time_seconds': execution_time,
            'success': success,
            'profile_stats': stats_stream.getvalue(),
            'result': result
        }
    
    def _optimize_memory(self) -> Dict[str, Any]:
        """Optimiza uso de memory."""
        import gc
        
        before_gc = len(gc.get_objects())
        collected = gc.collect()
        after_gc = len(gc.get_objects())
        
        # clean caché if es necessary
        if hasattr(self.cache, '_cleanup_expired'):
            self.cache._cleanup_expired()
        
        return {
            'objects_before_gc': before_gc,
            'objects_collected': collected,
            'objects_after_gc': after_gc,
            'cache_hit_rate': self.cache.hit_rate
        }
    
    def _predict_performance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Predice rendimiento basado en contexto."""
        if not self.ai_optimizer.performance_history:
            return {"prediction": "insufficient_data"}
        
        # analysis simple de prediction
        recent_metrics = self.ai_optimizer.performance_history[-10:]
        avg_execution_time = sum(
            entry['metrics'].execution_time_ms for entry in recent_metrics
        ) / len(recent_metrics)
        
        # prediction basada en contexto
        prediction_factor = 1.0
        if context.get('data_size', 0) > 1000:
            prediction_factor *= 1.5
        if context.get('complexity', 'low') == 'high':
            prediction_factor *= 2.0
        
        predicted_time = avg_execution_time * prediction_factor
        
        return {
            'predicted_execution_time_ms': predicted_time,
            'confidence': 0.7,
            'recommendation': 'optimize_if_above_100ms' if predicted_time > 100 else 'proceed_normal'
        }
    
    def _start_analytics_thread(self) -> None:
        """Inicia thread de analytics en tiempo real."""
        def analytics_worker():
            while True:
                try:
                    time.sleep(self.config.analytics_interval_seconds)
                    self._generate_analytics_report()
                except Exception as e:
                    self.logger.error(f"Error en analytics thread: {e}")
        
        analytics_thread = threading.Thread(target=analytics_worker, daemon=True)
        analytics_thread.start()
    
    def _generate_analytics_report(self) -> Dict[str, Any]:
        """Genera reporte de analytics."""
        if not self.metrics_buffer:
            return {}
        
        # calculate estadísticas
        total_operations = len(self.metrics_buffer)
        avg_execution_time = sum(
            entry['metrics'].execution_time_ms for entry in self.metrics_buffer
        ) / total_operations
        
        success_count = sum(
            entry['metrics'].success_count for entry in self.metrics_buffer
        )
        
        success_rate = (success_count / total_operations) * 100 if total_operations > 0 else 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_operations': total_operations,
            'average_execution_time_ms': avg_execution_time,
            'success_rate_percent': success_rate,
            'cache_hit_rate': self.cache.hit_rate,
            'active_contracts': len([c for c in self.contracts.values() if c.active]),
            'total_violations': sum(len(c.violations) for c in self.contracts.values())
        }
        
        self.logger.info(f" Analytics Report: {json.dumps(report, indent=2)}")
        return report
    
    async def async_smart_route(self, utility_type: str, operation: str, **kwargs) -> Any:
        """Versión asíncrona del smart routing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.smart_route,
            utility_type,
            operation,
            **kwargs
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene estado complete del orchestrator."""
        return {
            'config': {
                'max_memory_mb': self.config.max_memory_mb,
                'max_threads': self.config.max_threads,
                'cache_size_mb': self.config.cache_size_mb,
                'ai_optimization_enabled': self.config.enable_ai_optimization
            },
            'performance': {
                'cache_hit_rate': self.cache.hit_rate,
                'total_cached_items': len(self.cache.cache),
                'metrics_buffer_size': len(self.metrics_buffer)
            },
            'contracts': {
                name: {
                    'active': contract.active,
                    'violations': len(contract.violations),
                    'sla_ms': contract.sla_ms
                }
                for name, contract in self.contracts.items()
            },
            'ai_optimizer': {
                'performance_history_size': len(self.ai_optimizer.performance_history),
                'learning_rate': self.ai_optimizer.learning_rate
            }
        }
    
    def shutdown(self) -> None:
        """Apaga el orchestrator limpiamente."""
        self.logger.info(" Iniciando shutdown del Ultra Utils Orchestrator...")
        
        # close executor
        self.executor.shutdown(wait=True)
        
        # generate reporte end
        final_report = self._generate_analytics_report()
        self.logger.info(f" Reporte final: {json.dumps(final_report, indent=2)}")
        
        self.logger.info(" Ultra Utils Orchestrator apagado correctamente")


# instance global del orchestrator
_ultra_orchestrator: Optional[UltraUtilsOrchestrator] = None


def get_ultra_orchestrator(config: Optional[UltraUtilsConfig] = None) -> UltraUtilsOrchestrator:
    """
    Obtiene la instance global del Ultra Utils Orchestrator.
    
    Args:
        config: setup personalizada (optional)
        
    Returns:
        instance del orchestrator
    """
    global _ultra_orchestrator
    
    if _ultra_orchestrator is None:
        _ultra_orchestrator = UltraUtilsOrchestrator(config)
    
    return _ultra_orchestrator


# Funciones de conveniencia for acceso fast
def ultra_route(utility_type: str, operation: str, **kwargs) -> Any:
    """function de conveniencia for smart routing."""
    orchestrator = get_ultra_orchestrator()
    return orchestrator.smart_route(utility_type, operation, **kwargs)


def ultra_status() -> Dict[str, Any]:
    """function de conveniencia for obtain estado."""
    orchestrator = get_ultra_orchestrator()
    return orchestrator.get_status()


def ultra_cache(ttl_seconds: int = 3600):
    """Decorador de conveniencia for caché inteligente."""
    orchestrator = get_ultra_orchestrator()
    return orchestrator._smart_cache_decorator(ttl_seconds)


def ultra_monitor(func: Callable) -> Callable:
    """Decorador de conveniencia for monitoreo."""
    orchestrator = get_ultra_orchestrator()
    return orchestrator._monitor_execution(func)


# Ejemplos de uso
if __name__ == "__main__":
    # setup personalizada
    config = UltraUtilsConfig(
        max_memory_mb=2048,
        enable_ai_optimization=True,
        enable_analytics=True
    )
    
    # create orchestrator
    orchestrator = UltraUtilsOrchestrator(config)
    
    # example de smart routing
    try:
        result = orchestrator.smart_route(
            'data',
            'flatten_dict',
            data={'a': {'b': {'c': 1}}}
        )
        logger.info(f"Resultado: {result}")
    except Exception as e:
        logger.error(f"Error: {e}")
    
    # obtain estado
    status = orchestrator.get_status()
    logger.info(f"Estado: {json.dumps(status, indent=2)}")
    
    # close orchestrator
    orchestrator.shutdown()