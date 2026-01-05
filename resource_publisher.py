#!/usr/bin/env python3
"""
Resource Monitor and Publisher for Capibara6
Envía información de uso de recursos a la VM services cada 2 segundos

Este servicio monitorea el uso de recursos en esta VM models-europe
y los envía a la VM services para que pueda tomar decisiones de
fallback a colas de trabajo cuando los recursos superen el 90%
"""

import time
import psutil
import requests
import json
from datetime import datetime
from threading import Thread
from typing import Dict, Any


class ResourceMonitor:
    """Monitorea el uso de recursos en esta VM y los envía a la VM services"""
    
    def __init__(self, services_vm_host: str = "34.175.255.139", services_vm_port: int = 5000):
        self.services_vm_host = services_vm_host
        self.services_vm_port = services_vm_port
        self.services_base_url = f"http://{services_vm_host}:{services_vm_port}"
        self.running = False
        self.publish_thread = None
        print(f"📊 ResourceMonitor inicializado para enviar datos a {self.services_base_url}")
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Obtiene el uso actual de recursos de esta VM"""
        try:
            # Uso de CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Uso de memoria
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            memory_total_mb = memory.total / (1024 * 1024)
            
            # Uso de disco
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_available_gb = disk.free / (1024**3)
            
            # Uso de red (para monitorear tráfico)
            net_io = psutil.net_io_counters()
            
            # Uso de procesos
            num_processes = len(psutil.pids())
            
            # Información de la VM models-europe específica
            vm_info = {
                'vm_id': 'models-europe',
                'vm_ip': '34.175.48.2',
                'location': 'europe-southwest1-b',
                'vm_type': 'C4A-standard-32 (32 vCPUs, 128 GB RAM)',
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            resource_data = {
                **vm_info,
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'high_usage': cpu_percent > 90
                },
                'memory': {
                    'percent': memory_percent,
                    'available_mb': round(memory_available_mb, 2),
                    'total_mb': round(memory_total_mb, 2),
                    'high_usage': memory_percent > 90
                },
                'disk': {
                    'percent': disk_percent,
                    'available_gb': round(disk_available_gb, 2),
                    'high_usage': disk_percent > 90
                },
                'system': {
                    'net_bytes_sent': net_io.bytes_sent,
                    'net_bytes_recv': net_io.bytes_recv,
                    'num_processes': num_processes
                },
                'models_status': self._get_model_server_status()
            }
            
            # Determinar estado general de recursos
            high_resource_usage = any([
                cpu_percent > 90,
                memory_percent > 90,
                disk_percent > 90
            ])
            
            resource_data['overall_status'] = {
                'high_resource_usage': high_resource_usage,
                'resource_threshold_exceeded': high_resource_usage,
                'recommendation': 'normal' if not high_resource_usage else 'queue_fallback'
            }
            
            return resource_data
            
        except Exception as e:
            print(f"❌ Error obteniendo uso de recursos: {e}")
            return {}
    
    def _get_model_server_status(self) -> Dict[str, Any]:
        """Obtiene el estado del servidor de modelos"""
        try:
            # Verificar si el servidor de modelos está corriendo
            response = requests.get(f"http://localhost:8082/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                return {
                    'status': 'healthy',
                    'port': 8082,
                    'models_loaded': health_data.get('models_loaded', 0),
                    'models_available': health_data.get('models_available', 0)
                }
            else:
                return {
                    'status': 'unhealthy',
                    'port': 8082,
                    'error': f'HTTP {response.status_code}'
                }
        except Exception:
            return {
                'status': 'unreachable',
                'port': 8082,
                'error': 'No se puede conectar al servidor de modelos'
            }
    
    def publish_resource_data(self):
        """Publica la información de recursos a la VM services"""
        try:
            resource_data = self.get_resource_usage()
            
            if not resource_data:
                print("⚠️  No se pudo obtener datos de recursos")
                return
            
            # Enviar datos a la VM services
            endpoint = f"{self.services_base_url}/api/resources/update"
            response = requests.post(
                endpoint,
                json=resource_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code in [200, 201, 204]:
                overall_status = resource_data.get('overall_status', {}).get('recommendation', 'normal')
                cpu_usage = resource_data.get('cpu', {}).get('percent', 0)
                mem_usage = resource_data.get('memory', {}).get('percent', 0)
                
                print(f"📈 [{datetime.now().strftime('%H:%M:%S')}] Recursos enviados - CPU: {cpu_usage}%, Mem: {mem_usage}% - Estado: {overall_status}")
            else:
                print(f"⚠️  Error al enviar recursos a services: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Error al publicar datos de recursos: {e}")
    
    def start_monitoring(self, interval_seconds: float = 2.0):
        """Inicia el monitoreo continuo de recursos"""
        if self.running:
            print("⚠️  El monitoreo ya está en marcha")
            return
            
        self.running = True
        print(f"🔄 Iniciando monitoreo de recursos cada {interval_seconds} segundos...")
        print(f"📡 Enviando datos a: {self.services_base_url}")
        
        def monitor_loop():
            while self.running:
                try:
                    self.publish_resource_data()
                    time.sleep(interval_seconds)
                except Exception as e:
                    print(f"❌ Error en el bucle de monitoreo: {e}")
                    time.sleep(interval_seconds)
        
        self.publish_thread = Thread(target=monitor_loop, daemon=True)
        self.publish_thread.start()
        
        print("✅ Monitoreo de recursos iniciado")
    
    def stop_monitoring(self):
        """Detiene el monitoreo de recursos"""
        self.running = False
        if self.publish_thread and self.publish_thread.is_alive():
            self.publish_thread.join(timeout=2)
        print("🛑 Monitoreo de recursos detenido")


def main():
    print("🦫 Capibara6 Resource Monitor Publisher")
    print("=" * 60)
    print("Este servicio envía información de uso de recursos")
    print("desde esta VM models-europe a la VM services cada 2 segundos")
    print("para que pueda tomar decisiones de fallback cuando")
    print("los recursos superen el 90% de uso")
    print("=" * 60)
    
    # Crear el monitor de recursos
    monitor = ResourceMonitor(
        services_vm_host="34.175.255.139",  # VM services
        services_vm_port=5000
    )
    
    try:
        # Iniciar monitoreo cada 2 segundos
        monitor.start_monitoring(interval_seconds=2.0)
        
        print(f"\n📊 MONITOREO ACTIVO")
        print("Presiona Ctrl+C para detener...")
        
        # Mostrar información inicial
        initial_data = monitor.get_resource_usage()
        if initial_data:
            cpu_pct = initial_data.get('cpu', {}).get('percent', 0)
            mem_pct = initial_data.get('memory', {}).get('percent', 0)
            print(f"Estado inicial - CPU: {cpu_pct}%, Memoria: {mem_pct}%")
        
        # Mantener el programa corriendo
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Deteniendo monitoreo de recursos...")
        monitor.stop_monitoring()
        print("👋 Servicio de monitoreo detenido")


if __name__ == "__main__":
    main()