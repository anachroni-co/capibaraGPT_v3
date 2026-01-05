#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capibara6 Services Manager - Gestor de Servicios
Controla todos los servicios de capibara6: backend, frontend, y componentes E2B
"""

import subprocess
import sys
import os
import time
import psutil
import requests
from typing import Dict, List, Tuple
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Capibara6ServiceManager:
    """Gestor de servicios de capibara6"""
    
    def __init__(self):
        self.root_dir = "/home/elect/capibara6"
        self.services = {
            'backend': {
                'name': 'Backend API',
                'port': 8000,
                'process_cmd': 'uvicorn.*main:app',
                'startup_cmd': ['uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000', '--reload'],
                'working_dir': 'backend',
                'health_url': 'http://localhost:8000/health',
                'is_running': False,
                'pid': None
            },
            'integrated_server': {
                'name': 'Integrated Server',
                'port': 5001,
                'process_cmd': 'capibara6_integrated_server',
                'startup_cmd': ['python', 'capibara6_integrated_server_ollama.py'],
                'working_dir': 'backend',
                'health_url': 'http://localhost:5001/health',
                'is_running': False,
                'pid': None
            },
            'frontend': {
                'name': 'Frontend Server',
                'port': 8080,
                'process_cmd': 'http.server',
                'startup_cmd': ['python', '-m', 'http.server', '8080'],
                'working_dir': 'web',
                'health_url': 'http://localhost:8080/',
                'is_running': False,
                'pid': None
            }
        }
        self.processes = {}
    
    def check_port_status(self, port: int) -> bool:
        """Verifica si un puerto está en uso"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)  # Timeout breve para evitar espera larga
                result = sock.connect_ex(('localhost', port))
                return result == 0
        except Exception:
            return False
    
    def find_processes_by_port(self, port: int) -> List[psutil.Process]:
        """Encuentra procesos que están usando un puerto determinado"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                connections = proc.connections()
                for conn in connections:
                    if conn.laddr and conn.laddr.port == port:
                        processes.append(proc)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return processes
    
    def find_process_by_cmd(self, cmd_pattern: str) -> List[psutil.Process]:
        """Encuentra procesos que coinciden con un patrón de comando"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if cmd_pattern in cmdline:
                        processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return processes
    
    def start_service(self, service_name: str) -> bool:
        """Inicia un servicio específico"""
        if service_name not in self.services:
            logger.error(f"❌ Servicio {service_name} no encontrado")
            return False
        
        service = self.services[service_name]
        
        # Verificar si ya está corriendo
        if self.check_port_status(service['port']):
            logger.info(f"⚠️  Servicio {service_name} ya está corriendo en el puerto {service['port']}")
            processes = self.find_processes_by_port(service['port'])
            for proc in processes:
                logger.info(f"   PID {proc.pid}: {' '.join(proc.cmdline()[:3]) if proc.cmdline() else proc.name()}")
            return True
        
        try:
            # Cambiar al directorio correspondiente
            original_dir = os.getcwd()
            working_path = os.path.join(self.root_dir, service['working_dir'])
            os.chdir(working_path)
            
            # Iniciar el proceso
            logger.info(f"🚀 Iniciando {service['name']} en puerto {service['port']}...")
            process = subprocess.Popen(
                service['startup_cmd'],
                cwd=working_path,
                stdout=subprocess.DEVNULL,  # Suprimir salida para limpiar
                stderr=subprocess.DEVNULL,  # Suprimir errores para limpiar
                preexec_fn=os.setsid  # Crear nuevo grupo de procesos
            )
            
            # Guardar información del proceso
            self.processes[service_name] = process
            self.services[service_name]['pid'] = process.pid
            self.services[service_name]['is_running'] = True
            
            logger.info(f"✅ Servicio {service_name} iniciado con PID {process.pid}")
            
            # Regresar al directorio original
            os.chdir(original_dir)
            
            # Esperar a que inicie
            time.sleep(3)
            
            # Verificar si está saludable
            if self.is_service_healthy(service_name):
                logger.info(f"✅ {service_name} está activo y saludable")
                return True
            else:
                logger.warning(f"⚠️  {service_name} iniciado pero no responde al health check")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error iniciando servicio {service_name}: {e}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Detiene un servicio específico"""
        if service_name not in self.services:
            logger.error(f"❌ Servicio {service_name} no encontrado")
            return False
        
        service = self.services[service_name]
        
        logger.info(f"🛑 Deteniendo {service['name']}...")
        
        # Primero intentar detener por PID guardado
        if service['pid']:
            try:
                proc = psutil.Process(service['pid'])
                proc.terminate()
                proc.wait(timeout=5)  # Esperar hasta 5 segundos
                logger.info(f"✅ Servicio {service_name} detenido (PID {service['pid']})")
            except psutil.NoSuchProcess:
                logger.info(f"✅ Servicio {service_name} ya estaba detenido")
            except psutil.TimeoutExpired:
                logger.warning(f"⚠️  Proceso {service['pid']} no respondió, forzando terminación...")
                try:
                    proc.kill()
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"❌ Error deteniendo servicio {service_name}: {e}")
        
        # Si no teníamos PID o si queremos asegurar, buscar por patrón de comando
        processes = self.find_process_by_cmd(service['process_cmd'])
        for proc in processes:
            if proc.pid != service['pid']:  # No duplicar si ya lo terminamos por PID
                try:
                    logger.info(f"Terminando proceso adicional (PID {proc.pid})")
                    proc.terminate()
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()
                except Exception:
                    pass
        
        # Marcar como detenido
        self.services[service_name]['is_running'] = False
        self.services[service_name]['pid'] = None
        
        if service_name in self.processes:
            del self.processes[service_name]
        
        return True
    
    def is_service_healthy(self, service_name: str) -> bool:
        """Verifica si un servicio está respondiendo"""
        service = self.services[service_name]
        try:
            response = requests.get(service['health_url'], timeout=5)
            return response.status_code in [200, 405]  # 405 también indica que el servidor está corriendo
        except:
            return False
    
    def start_all(self) -> Dict[str, bool]:
        """Inicia todos los servicios"""
        logger.info("🚀 Iniciando todos los servicios de capibara6...")
        
        results = {}
        # Iniciar en orden adecuado
        for service_name in ['frontend', 'backend', 'integrated_server']: 
            results[service_name] = self.start_service(service_name)
            time.sleep(2)  # Breve pausa entre servicios
        
        return results
    
    def stop_all(self) -> Dict[str, bool]:
        """Detiene todos los servicios"""
        logger.info("🛑 Deteniendo todos los servicios de capibara6...")
        
        results = {}
        # Detener en orden inverso (primero integrated_server, luego backend, finalmente frontend)
        for service_name in ['integrated_server', 'backend', 'frontend']:
            results[service_name] = self.stop_service(service_name)
        
        return results
    
    def check_all_status(self) -> Dict[str, Dict[str, any]]:
        """Verifica el estado de todos los servicios"""
        status = {}
        
        for service_name, service in self.services.items():
            port_open = self.check_port_status(service['port'])
            healthy = self.is_service_healthy(service_name)
            processes = self.find_processes_by_port(service['port'])
            
            status[service_name] = {
                'name': service['name'],
                'port': service['port'],
                'port_open': port_open,
                'healthy': healthy,
                'process_count': len(processes),
                'processes': [
                    {'pid': p.pid, 'name': p.name(), 'cmdline': ' '.join(p.cmdline()[:3]) if p.cmdline() else 'N/A'} 
                    for p in processes
                ],
                'is_running': port_open and healthy
            }
        
        return status
    
    def print_status_report(self):
        """Imprime un reporte de estado de servicios"""
        status = self.check_all_status()
        
        print("\n🦫 CAPIBARA6 - ESTADO DE SERVICIOS")
        print("=" * 70)
        
        for service_name, s in status.items():
            service_config = self.services[service_name]
            status_emoji = "✅" if s['is_running'] else ("⚠️" if s['port_open'] else "❌")
            
            print(f"\n{status_emoji} {s['name']} (puerto {s['port']})")
            print(f"   Estado: {'CORRIENDO' if s['is_running'] else 'DETENIDO'}")
            print(f"   Puerto: {'ABIERTO' if s['port_open'] else 'CERRADO'}")
            print(f"   Salud: {'✅ OK' if s['healthy'] else '❌ NO RESPONDE'}")
            print(f"   Procesos: {s['process_count']}")
            
            if s['processes']:
                for proc in s['processes']:
                    print(f"     - PID {proc['pid']}: {proc['cmdline']}")
        
        running_count = sum(1 for s in status.values() if s['is_running'])
        total_count = len(status)
        
        print(f"\n📊 RESUMEN: {running_count}/{total_count} servicios activos")
        
        if running_count == total_count:
            print("🎉 ¡Todos los servicios están activos y funcionales!")
        elif running_count == 0:
            print("😴 Todos los servicios están detenidos")
        else:
            print(f"⚡ {running_count} servicios activos, {total_count - running_count} detenidos")
        
        print("=" * 70)
    
    def restart_service(self, service_name: str) -> bool:
        """Reinicia un servicio específico"""
        logger.info(f"🔄 Reiniciando servicio {service_name}...")
        success = self.stop_service(service_name)
        time.sleep(2)
        if success:
            return self.start_service(service_name)
        else:
            logger.error(f"❌ Error deteniendo servicio {service_name} para reiniciar")
            return False


def main():
    """Función principal del gestor de servicios"""
    manager = Capibara6ServiceManager()
    
    if len(sys.argv) < 2:
        print("uso: python services_manager.py [start|stop|status|restart] [service_name|all]")
        print("\nComandos disponibles:")
        print("  status                 - Verificar estado de todos los servicios")
        print("  start all              - Iniciar todos los servicios")
        print("  stop all               - Detener todos los servicios")
        print("  start <service>        - Iniciar servicio específico")
        print("  stop <service>         - Detener servicio específico")
        print("  restart <service>      - Reiniciar servicio específico")
        print("\nServicios disponibles:")
        for name, config in manager.services.items():
            print(f"  {name:<20} - {config['name']} (puerto {config['port']})")
        return
    
    action = sys.argv[1]
    
    if action == 'status':
        manager.print_status_report()
        
    elif action == 'start':
        if len(sys.argv) > 2:
            service_name = sys.argv[2]
            if service_name == 'all':
                results = manager.start_all()
                print("\nResultados de inicio:")
                for service, success in results.items():
                    service_name_display = manager.services[service]['name']
                    status = "✅" if success else "❌"
                    print(f"  {status} {service_name_display}: {'ÉXITO' if success else 'FALLO'}")
            elif service_name in manager.services:
                success = manager.start_service(service_name)
                if success:
                    print(f"✅ Servicio {service_name} iniciado correctamente")
                else:
                    print(f"❌ Error iniciando servicio {service_name}")
            else:
                print(f"❌ Servicio '{service_name}' no encontrado")
                print(f"   Servicios disponibles: {list(manager.services.keys())}")
        else:
            print("❌ Debe especificar un servicio. Use 'all' para iniciar todos o un nombre específico.")
    
    elif action == 'stop':
        if len(sys.argv) > 2:
            service_name = sys.argv[2]
            if service_name == 'all':
                results = manager.stop_all()
                print("\nResultados de detención:")
                for service, success in results.items():
                    service_name_display = manager.services[service]['name']
                    status = "✅" if success else "❌"
                    print(f"  {status} {service_name_display}: {'DETENIDO' if success else 'ERROR'}")
            elif service_name in manager.services:
                success = manager.stop_service(service_name)
                if success:
                    print(f"✅ Servicio {service_name} detenido correctamente")
                else:
                    print(f"❌ Error deteniendo servicio {service_name}")
            else:
                print(f"❌ Servicio '{service_name}' no encontrado")
                print(f"   Servicios disponibles: {list(manager.services.keys())}")
        else:
            print("❌ Debe especificar un servicio. Use 'all' para detener todos o un nombre específico.")
    
    elif action == 'restart':
        if len(sys.argv) > 2:
            service_name = sys.argv[2]
            if service_name in manager.services:
                success = manager.restart_service(service_name)
                if success:
                    print(f"✅ Servicio {service_name} reiniciado correctamente")
                else:
                    print(f"❌ Error reiniciando servicio {service_name}")
            else:
                print(f"❌ Servicio '{service_name}' no encontrado")
                print(f"   Servicios disponibles: {list(manager.services.keys())}")
        else:
            print("❌ Debe especificar un servicio para reiniciar.")
    
    else:
        print(f"❌ Comando '{action}' no reconocido")
        print("   Comandos válidos: status, start, stop, restart")


if __name__ == "__main__":
    main()