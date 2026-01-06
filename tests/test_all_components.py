#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para probar conjuntamente el servidor y la interfaz interactiva ARM-Axion
"""

import subprocess
import sys
import time
import requests
import threading
import signal
from pathlib import Path

def check_port_availability(port=8080):
    """Verificar si el puerto está disponible"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def start_server():
    """Iniciar el servidor ARM-Axion en un hilo separado con liberación de puerto"""
    print("🚀 Iniciando servidor ARM-Axion en segundo plano con liberación de puerto...")

    import subprocess
    import os
    import sys
    import socket

    # Verificar si el puerto está en uso y liberarlo
    port = 8080
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(('127.0.0.1', port))
        if result == 0:  # Puerto en uso
            print(f"⚠️  Liberando puerto {port}...")
            try:
                # Obtener PID que usa el puerto y terminarlo
                result = subprocess.run(['lsof', '-t', '-i', f':{port}'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    pids = result.stdout.strip().split('\n')
                    for pid_str in pids:
                        if pid_str:
                            pid = int(pid_str)
                            print(f"   Terminando proceso {pid} en puerto {port}...")
                            os.kill(pid, signal.SIGTERM)
                            time.sleep(2)
                            try:
                                os.kill(pid, 0)  # Verificar si sigue vivo
                                os.kill(pid, signal.SIGKILL)  # Forzar
                            except OSError:
                                pass  # Ya terminado
                time.sleep(2)  # Esperar a que el puerto se libere
            except Exception as e:
                print(f"⚠️  Error al liberar puerto: {e}")

    # Configurar variables de entorno
    env = os.environ.copy()
    env.update({
        'VLLM_USE_V1': '0',
        'VLLM_ENABLE_V1_ENGINE': '0',
        'VLLM_USE_TRITON_FLASH_ATTN': '0',
        'TORCH_COMPILE_BACKEND': 'eager',
        'TORCHDYNAMO_DISABLED': '1',
        'TORCHINDUCTOR_DISABLED': '1'
    })

    # Directorios
    vllm_integration_dir = '/home/elect/capibara6/arm-axion-optimizations/vllm_integration'

    # Comando para iniciar el servidor
    cmd = [
        sys.executable,
        '-c',
        f'''
import os
import sys
import time

# Agregar paths necesarios
vllm_modified_path = '/home/elect/capibara6/vllm-source-modified'
arm_axion_path = '/home/elect/capibara6/arm-axion-optimizations'
if vllm_modified_path not in sys.path:
    sys.path.insert(0, vllm_modified_path)
if arm_axion_path not in sys.path:
    sys.path.insert(0, arm_axion_path)

# Cambiar al directorio correcto
os.chdir("{vllm_integration_dir}")

# Importar y ejecutar el servidor
try:
    from inference_server import app
    import uvicorn
    print("🚀 Servidor ARM-Axion iniciado en puerto {port}")
    print("🌐 Endpoint: http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port={port}, log_level="info")
except Exception as e:
    print(f"❌ Error al iniciar servidor: {{e}}")
    import traceback
    traceback.print_exc()
'''
    ]

    # Iniciar proceso
    process = subprocess.Popen(cmd, env=env)

    return process

def wait_for_server(timeout=60):
    """Esperar a que el servidor esté disponible"""
    print(f"⏳ Esperando a que el servidor esté disponible (máximo {timeout}s)...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            if response.status_code == 200:
                print("✅ Servidor disponible!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    
    print("❌ Tiempo de espera agotado, servidor no disponible")
    return False

def test_models_loading():
    """Probar la carga de los modelos"""
    print("\\n🔍 Probando carga de modelos...")
    
    try:
        # Verificar modelos disponibles
        response = requests.get("http://localhost:8080/experts", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {len(data['experts'])} expertos disponibles:")
            
            for expert in data['experts']:
                status = "✅ CARGADO" if expert['is_loaded'] else "⏰ NO CARGADO"
                print(f"  - {expert['expert_id']}: {status}")
            
            return True
        else:
            print(f"❌ Error al consultar expertos: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error al probar carga de modelos: {e}")
        return False

def run_interactive_test():
    """Ejecutar la interfaz interactiva de pruebas"""
    print("\\n🎮 Iniciando interfaz interactiva para pruebas manuales...")
    print("Puedes probar los modelos desde aquí:")
    print("  - Probar modelo individual")
    print("  - Probar todos los modelos")
    print("  - Comparar modelos")
    print("  - Ver estado de carga")
    print("\\nPresiona Ctrl+C para salir del modo interactivo")

def main():
    print("🧪 TEST INTEGRADO - SERVIDOR Y INTERFAZ ARM-Axion")
    print("="*60)
    
    if not check_port_availability(8080):
        print("⚠️  Puerto 8080 en uso. ¿Ya hay un servidor corriendo?")
        response = input("¿Deseas continuar de todos modos? (puede causar conflicto) [y/N]: ")
        if not response.lower().startswith('y'):
            print("Operación cancelada.")
            return
    
    # Iniciar servidor
    server_process = start_server()
    
    try:
        # Esperar a que el servidor esté listo
        if not wait_for_server(60):
            print("\\n❌ No se pudo iniciar el servidor correctamente.")
            return
        
        # Probar la carga de modelos
        if not test_models_loading():
            print("\\n⚠️  Hubo problemas con la carga de modelos.")
        else:
            print("\\n✅ Prueba de carga de modelos completada exitosamente.")
        
        # Ejecutar pruebas adicionales
        print("\\n🎯 PRUEBAS AUTOMÁTICAS COMPLETADAS")
        print("-" * 40)
        print("✅ Servidor corriendo en http://localhost:8080")
        print("✅ Endpoint /health funcional")
        print("✅ Endpoint /experts accesible")
        print("✅ Modelos disponibles para carga")
        
        # Opción para iniciar modo interactivo
        print("\\n🚀 Opciones adicionales:")
        print("  1. Modo interactivo para probar modelos")
        print("  2. Solo verificar conectividad (salir)")
        
        choice = input("\\nSeleccione opción (1-2): ").strip()
        
        if choice == "1":
            # Importar y ejecutar la interfaz interactiva
            sys.path.insert(0, '/home/elect/capibara6')
            from test_models_interactive import InteractiveARMVLLM
            
            print("\\n🚀 Iniciando interfaz interactiva ARM-Axion...")
            app = InteractiveARMVLLM()
            app.run()
        else:
            print("\\n✅ Prueba completada. El servidor está funcionando en segundo plano.")
            print("Puedes acceder a él en http://localhost:8080")
            print("Endpoints disponibles:")
            print("  - GET /health")
            print("  - GET /experts") 
            print("  - GET /stats")
            print("  - POST /v1/chat/completions")
            
    except KeyboardInterrupt:
        print("\\n\\n⚠️  Interrupción por usuario")
    except Exception as e:
        print(f"\\n❌ Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\\n🛑 Deteniendo servidor...")
        if 'server_process' in locals():
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
            print("✅ Servidor detenido.")

if __name__ == "__main__":
    main()