"""
Monitoreo and Métricas
===================
Sistema de monitoreo en tiempo real.
"""

import os
import sys
import time
import logging
import threading
import psutil #type: ignore
from typing import Dict, List, Optional, Union, Callable

logger = logging.getLogger(__name__)

class MonitoringError(Exception):
    """error base for monitoreo."""
    pass

class RealTimeMonitor:
    """Monitor en tiempo real."""
    
    def __init__(
        self,
        interval: float = 1.0,
        metrics: Optional[List[str]] = None
    ):
        self.interval = interval
        self.metrics = metrics or ["cpu", "memory", "disk"]
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._data: Dict[str, List[float]] = {
            metric: [] for metric in self.metrics
        }
        
        logger.info(f" Monitor inicializado (interval={interval}s)")
    
    def start(self) -> None:
        """start monitoreo."""
        with self._lock:
            if self.running:
                return
                
            self.running = True
            self._thread = threading.Thread(target=self._monitor_loop)
            self._thread.daemon = True
            self._thread.start()
            
            logger.info(" Monitoreo iniciado")
    
    def stop(self) -> None:
        """stop monitoreo."""
        with self._lock:
            self.running = False
            if self._thread:
                self._thread.join()
                self._thread = None
                
            logger.info("️ Monitoreo detenido")
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        obtain métricas.
        
        Returns:
            Diccionario de métricas
        """
        with self._lock:
            return self._data.copy()
    
    def clear_metrics(self) -> None:
        """Clear metrics."""
        with self._lock:
            for metric in self._data:
                self._data[metric].clear()
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                metrics = {}
                
                if "cpu" in self.metrics:
                    metrics["cpu"] = psutil.cpu_percent()
                
                if "memory" in self.metrics:
                    memory = psutil.virtual_memory()
                    metrics["memory"] = memory.percent
                
                if "disk" in self.metrics:
                    disk = psutil.disk_usage("/")
                    metrics["disk"] = disk.percent
                
                with self._lock:
                    for metric, value in metrics.items():
                        self._data[metric].append(value)
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.warning(f"️ Error en monitoreo: {e}")

class ResourceMonitor:
    """Monitor de recursos del sistema."""
    
    def __init__(self):
        self.process = psutil.Process()
        logger.info(" Monitor de recursos inicializado")
    
    def get_memory_usage(self) -> float:
        """
        obtain uso de memory.
        
        Returns:
            percentage of memory usado
        """
        try:
            return self.process.memory_percent()
        except Exception as e:
            logger.warning(f"️ Error al obtener memoria: {e}")
            return 0.0
    
    def get_cpu_usage(self) -> float:
        """
        obtain uso de cpu.
        
        Returns:
            percentage of cpu usado
        """
        try:
            return self.process.cpu_percent()
        except Exception as e:
            logger.warning(f"️ Error al obtener CPU: {e}")
            return 0.0
    
    def get_disk_io(self) -> Dict[str, float]:
        """
        obtain estadísticas de I/or.
        
        Returns:
            Diccionario with bytes leídos/escritos
        """
        try:
            io = self.process.io_counters()
            return {
                "read_bytes": io.read_bytes,
                "write_bytes": io.write_bytes
            }
        except Exception as e:
            logger.warning(f"️ Error al obtener I/O: {e}")
            return {"read_bytes": 0, "write_bytes": 0}
    
    def get_threads(self) -> int:
        """
        obtain number of threads.
        
        Returns:
            number of threads
        """
        try:
            return self.process.num_threads()
        except Exception as e:
            logger.warning(f"️ Error al obtener threads: {e}")
            return 0

class MemoryMonitor:
    """Monitor de uso de memory."""
    
    def __init__(self):
        """Inicializa el monitor."""
        self.memory_usage = {}
    
    def track_memory(self, name: str, size: int):
        """Registra uso de memory."""
        self.memory_usage[name] = size
    
    def get_memory_usage(self, name: str) -> int:
        """Obtiene uso de memory."""
        return self.memory_usage.get(name, 0)

def setup_monitoring(
    interval: float = 1.0,
    metrics: Optional[List[str]] = None
) -> RealTimeMonitor:
    """
    configure monitoreo.
    
    Args:
        interval: Intervalo de muestreo
        metrics: list de métricas
        
    Returns:
        Monitor configurado
    """
    monitor = RealTimeMonitor(interval, metrics)
    monitor.start()
    return monitor 
