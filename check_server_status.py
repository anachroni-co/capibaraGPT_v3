#!/usr/bin/env python3
"""
Script para verificar si el servidor multimodelo ARM-Axion está corriendo
"""

import requests
import subprocess
import sys
from typing import Dict, Optional

def check_server_status():
    """Verifica si el servidor está corriendo"""
    try:
        # Intentar obtener información del servidor (ruta correcta)
        response = requests.get("http://localhost:8080/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            experts = data.get("experts", [])
            print(f"✅ Servidor multimodelo ARM-Axion está corriendo")
            print(f"   Expertos disponibles: {len(experts)}")
            for expert in experts:
                print(f"   - {expert.get('expert_id', 'unknown')} ({expert.get('domain', 'unknown')})")
            return True
        else:
            # Intentar con otra ruta
            response = requests.get("http://localhost:8080/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Servidor multimodelo ARM-Axion está corriendo")
                print(f"   Estado: {health_data.get('status', 'unknown')}")
                return True
            else:
                print(f"❌ Servidor respondió con código {response.status_code}")
                return False
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar al servidor multimodelo ARM-Axion")
        print("   El servidor puede no estar corriendo o estar en un puerto diferente")
        return False
    except requests.exceptions.Timeout:
        print("❌ Timeout al intentar conectar con el servidor")
        return False
    except Exception as e:
        print(f"❌ Error al verificar el servidor: {e}")
        return False

def check_running_processes():
    """Verifica si hay procesos del servidor corriendo"""
    try:
        # Buscar procesos que contengan "inference_server" o "multi_model_server"
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        running_servers = []
        for line in result.stdout.split('\n'):
            if ('inference_server' in line or 
                'multi_model_server' in line or 
                'start_vllm_arm_axion' in line or
                'python' in line and 'vllm' in line):
                if 'grep' not in line and line.strip():
                    running_servers.append(line.strip())
        
        if running_servers:
            print("\n🔍 Procesos del servidor encontrados:")
            for server in running_servers:
                print(f"   {server}")
        else:
            print("\n❌ No se encontraron procesos de servidor corriendo")
        
        return len(running_servers) > 0
    except Exception as e:
        print(f"❌ Error al verificar procesos: {e}")
        return False

def check_ports():
    """Verifica qué puertos están ocupados por el servidor"""
    try:
        import socket
        ports_to_check = [8080, 8082, 8081, 5000, 5003]
        
        print("\n🔍 Verificando puertos comunes:")
        for port in ports_to_check:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                if result == 0:
                    print(f"   Puerto {port}: ABIERTO")
                else:
                    print(f"   Puerto {port}: CERRADO")
    except Exception as e:
        print(f"❌ Error al verificar puertos: {e}")

def main():
    print("🔍 Verificando estado del servidor multimodelo ARM-Axion...")
    print("="*60)
    
    # Verificar si el servidor está respondiendo
    server_responding = check_server_status()
    
    # Verificar procesos corriendo
    processes_running = check_running_processes()
    
    # Verificar puertos
    check_ports()
    
    print("\n" + "="*60)
    if server_responding:
        print("✅ El servidor multimodelo ARM-Axion está corriendo y respondiendo")
        return True
    elif processes_running:
        print("⚠️  Hay procesos del servidor corriendo, pero no responde en la API")
        print("   Puede estar inicializando o haber un problema de configuración")
        return False
    else:
        print("❌ El servidor multimodelo ARM-Axion no parece estar corriendo")
        print("\nPara iniciar el servidor, puedes usar:")
        print("   cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration/")
        print("   python3 inference_server.py --host 0.0.0.0 --port 8080")
        print("\nO usando el script de inicio:")
        print("   cd /home/elect/capibara6")
        print("   ./start_vllm_arm_axion.sh")
        return False

if __name__ == "__main__":
    main()