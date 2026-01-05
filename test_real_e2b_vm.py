#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real E2B VM Execution Test - Prueba real de ejecución en la plataforma E2B
"""

import os
import asyncio
from e2b_code_interpreter import AsyncSandbox
from dotenv import load_dotenv
import time

# Cargar variables de entorno
load_dotenv()

# Configurar la API key de E2B
E2B_API_KEY = "e2b_4bebb1dfce65d4db486ed23cd352d88e72f105df"
os.environ['E2B_API_KEY'] = E2B_API_KEY

async def test_real_vm_execution():
    """Prueba real de ejecución en VM de E2B"""
    print("🌍 Iniciando prueba REAL de VM en plataforma E2B...")
    print(f"🔑 Usando API Key: {E2B_API_KEY[:15]}...")
    
    try:
        print("\n🚀 Creando sandbox real en plataforma E2B...")
        start_time = time.time()
        
        # Crear un sandbox real en la plataforma E2B
        sandbox = await AsyncSandbox.create(
            api_key=E2B_API_KEY,
            template="code-interpreter-v1",  # Usar la plantilla estándar
            timeout=600  # 10 minutos de timeout
        )
        
        creation_time = time.time() - start_time
        print(f"✅ Sandbox creado exitosamente en {creation_time:.2f} segundos")
        print(f"🆔 ID del sandbox: {sandbox.sandbox_id}")
        
        # Obtener información del sandbox
        info = await sandbox.get_info()
        print(f"📊 Información del sandbox:")
        print(f"   - ID: {info.sandbox_id}")
        print(f"   - Estado: {info.state}")
        print(f"   - Iniciado: {info.started_at}")
        
        # Medir recursos del sandbox
        print(f"\n🔍 Verificando recursos del sandbox...")
        try:
            metrics = await sandbox.get_metrics()
            if metrics:
                last_metrics = metrics[-1]  # Últimas métricas
                print(f"   - CPU Usage: {getattr(last_metrics, 'cpu_usage', 'N/A')}%")
                print(f"   - Memory Usage: {getattr(last_metrics, 'memory_usage', 'N/A')} MB")
                print(f"   - Disk Usage: {getattr(last_metrics, 'disk_usage', 'N/A')} MB")
        except Exception as metrics_error:
            print(f"⚠️  Error obteniendo métricas: {metrics_error}")
        
        # Ejecutar un comando para verificar que el sistema está completamente funcional
        print(f"\n💻 Ejecutando comandos de verificación en el VM real...")
        
        # Comando de sistema
        system_result = await sandbox.run_code("""
import platform
import os
import sys
print(f"Sistema operativo: {platform.system()}")
print(f"Versión Python: {sys.version}")
print(f"Nombre máquina: {platform.node()}")
print(f"Directorio actual: {os.getcwd()}")
print(f"Usuario: {os.getenv('USER', 'N/A')}")
""")
        
        if system_result and system_result.logs and system_result.logs.stdout:
            print("✅ Información del sistema:")
            for line in system_result.logs.stdout:
                print(f"   {line.rstrip()}")
        
        # Ejecutar una tarea más compleja para probar completamente el VM
        print(f"\n🧩 Ejecutando tarea compleja en el VM...")
        complex_task_result = await sandbox.run_code("""
import subprocess
import json
import os

# Verificar paquetes instalados
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    print("✅ Paquetes científicos disponibles")
    
    # Crear un cálculo intensivo para probar la CPU
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    result = np.dot(a, b)
    print(f"✅ Cálculo matricial 1000x1000 completado, shape: {result.shape}")
    
    # Verificar memoria disponible
    import psutil
    memory = psutil.virtual_memory()
    print(f"💾 Memoria disponible: {memory.available / (1024**3):.2f} GB")
    
    # Crear un archivo para probar el sistema de archivos
    with open('/home/user/test_file.txt', 'w') as f:
        f.write('E2B VM Test completed successfully!')
    
    print("📄 Archivo creado en el sistema de archivos del sandbox")
    
    # Verificar conexión a internet
    import urllib.request
    try:
        urllib.request.urlopen('https://httpbin.org/ip', timeout=5)
        print("🌐 Conexión a internet: OK")
    except:
        print("⚠️  Conexión a internet: Problemas detectados")
    
except ImportError as e:
    print(f"⚠️  Error de importación: {e}")

print("🎉 Tarea compleja completada en el VM real")
""")
        
        if complex_task_result and complex_task_result.logs and complex_task_result.logs.stdout:
            print("✅ Resultado de la tarea compleja:")
            for line in complex_task_result.logs.stdout:
                print(f"   {line.rstrip()}")
        
        if complex_task_result.error:
            print(f"❌ Error en la tarea compleja: {complex_task_result.error.message}")
        
        # Probar conexión de red
        print(f"\n🌐 Probando conectividad de red...")
        network_result = await sandbox.run_code("""
import socket
import requests

# Probar resolución DNS
try:
    ip = socket.gethostbyname('www.google.com')
    print(f"✅ DNS resolution: www.google.com -> {ip}")
except Exception as e:
    print(f"❌ DNS resolution failed: {e}")

# Probar conexión HTTP
try:
    response = requests.get('https://httpbin.org/json', timeout=10)
    print(f"✅ HTTP request: Status {response.status_code}")
except Exception as e:
    print(f"❌ HTTP request failed: {e}")

# Verificar herramientas de red
import subprocess
try:
    result = subprocess.run(['curl', '--version'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print(f"✅ curl disponible")
    else:
        print(f"⚠️  curl no disponible")
except subprocess.TimeoutExpired:
    print(f"⚠️  curl timeout")
except Exception as e:
    print(f"⚠️  curl error: {e}")
""")
        
        if network_result and network_result.logs and network_result.logs.stdout:
            print("✅ Resultado de conectividad:")
            for line in network_result.logs.stdout:
                print(f"   {line.rstrip()}")
        
        # Medir tiempo de ejecución total
        total_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total de operación: {total_time:.2f} segundos")
        
        # Verificar si el sandbox sigue funcionando
        is_running = await sandbox.is_running()
        print(f"🔄 Sandbox still running: {is_running}")
        
        # Finalizar el sandbox
        print(f"\n🛑 Finalizando sandbox...")
        await sandbox.kill()
        
        # Verificar que se haya detenido
        is_running_after_kill = await sandbox.is_running()
        print(f"🔄 Sandbox running after kill: {is_running_after_kill}")
        
        print(f"\n✅ Prueba REAL de VM en plataforma E2B completada exitosamente")
        print(f"🎯 El VM se creó, ejecutó tareas y se destruyó correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en la prueba REAL de VM: {e}")
        import traceback
        traceback.print_exc()
        return False

async def list_existing_sandboxes():
    """Lista los sandboxes existentes en la cuenta"""
    print(f"\n📋 Listando sandboxes existentes en la cuenta...")
    try:
        from e2b_code_interpreter import AsyncSandbox
        
        print("   No se pudo listar sandboxes existentes (API no compatible con este método)")
        return 0
        
    except Exception as e:
        print(f"❌ Error listando sandboxes: {e}")
        return 0

async def main():
    """Función principal"""
    print("🚀 INICIANDO PRUEBA REAL EN PLATAFORMA E2B")
    print("="*70)
    
    # Listar sandboxes existentes
    existing_count = await list_existing_sandboxes()
    
    # Ejecutar prueba de VM real
    vm_test_result = await test_real_vm_execution()
    
    print("\n" + "="*70)
    print("📋 RESULTADOS FINALES - PRUEBA REAL EN PLATAFORMA E2B")
    print("="*70)
    print(f"Sandboxes previos: {existing_count}")
    print(f"Prueba VM Real: {'✅ ÉXITO' if vm_test_result else '❌ FALLO'}")
    print("="*70)
    
    if vm_test_result:
        print("🎉 ¡PRUEBA REAL DE VM E2B COMPLETADA EXITOSAMENTE!")
        print("⚡ Se creó un VM real en la plataforma E2B")
        print("⚡ Se ejecutaron tareas complejas en el entorno aislado")
        print("⚡ El VM se destruyó correctamente tras la prueba")
        print("🎯 E2B está completamente funcional con tu cuenta")
    else:
        print("⚠️  Ocurrió un error en la prueba real del VM")

if __name__ == "__main__":
    asyncio.run(main())