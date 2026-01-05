#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para iniciar el servidor vLLM ARM-Axion con liberación previa del puerto
"""

import sys
import os
import signal
import subprocess
import time
import socket
import argparse
from pathlib import Path


def check_port(port):
    """Verificar si un puerto está en uso"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(('127.0.0.1', port))
        return result == 0  # Devuelve True si está en uso


def kill_process_on_port(port):
    """Matar proceso que está usando un puerto específico"""
    try:
        # Obtener PID que usa el puerto
        result = subprocess.run(['lsof', '-t', '-i', f':{port}'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid_str in pids:
                if pid_str:
                    pid = int(pid_str)
                    print(f"🛑 Terminando proceso {pid} en el puerto {port}...")
                    os.kill(pid, signal.SIGTERM)
                    # Esperar un momento antes de forzar la terminación si no responde
                    time.sleep(2)
                    # Verificar si el proceso sigue corriendo y forzar si es necesario
                    try:
                        os.kill(pid, 0)  # Verificar si el proceso sigue vivo
                        os.kill(pid, signal.SIGKILL)  # Forzar terminación
                        print(f"💀 Proceso {pid} forzadamente terminado")
                    except OSError:
                        print(f"✅ Proceso {pid} terminado exitosamente")
            return True
        return False
    except subprocess.CalledProcessError:
        # No hay proceso usando el puerto
        return False
    except Exception as e:
        print(f"⚠️  Error al intentar liberar el puerto {port}: {e}")
        return False


def setup_environment():
    """Configurar variables de entorno para compatibilidad ARM-Axion"""
    env = os.environ.copy()
    
    # Variables de entorno para compatibilidad ARM-Axion
    env_updates = {
        'VLLM_USE_V1': '0',
        'VLLM_ENABLE_V1_ENGINE': '0',
        'VLLM_USE_TRITON_FLASH_ATTN': '0',
        'TORCH_COMPILE_BACKEND': 'eager',
        'TORCHDYNAMO_DISABLED': '1',
        'TORCHINDUCTOR_DISABLED': '1',
        'CUDA_VISIBLE_DEVICES': ''  # Asegurar que no haya GPU detectada para forzar CPU
    }
    
    env.update(env_updates)
    return env


def main():
    parser = argparse.ArgumentParser(description='Iniciar servidor ARM-Axion con liberación de puerto')
    parser.add_argument('--port', type=int, default=8080, help='Puerto para el servidor (por defecto: 8080)')
    parser.add_argument('--host', default='0.0.0.0', help='Host para el servidor (por defecto: 0.0.0.0)')
    parser.add_argument('--config', default='config.five_models.optimized.json', 
                       help='Archivo de configuración (por defecto: config.five_models.optimized.json)')
    
    args = parser.parse_args()
    
    # Configurar directorios y paths
    capibara6_root = "/home/elect/capibara6"
    vllm_integration_dir = f"{capibara6_root}/arm-axion-optimizations/vllm_integration"
    vllm_source_dir = f"{capibara6_root}/vllm-source-modified"
    arm_axion_dir = f"{capibara6_root}/arm-axion-optimizations"
    
    # Mostrar información
    print("🚀 Iniciando servidor vLLM ARM-Axion con liberación de puerto...")
    print(f" Puerto: {args.port}")
    print(f" Host: {args.host}")
    print(f" Configuración: {args.config}")
    print()
    
    # Verificar existencia de directorios
    if not os.path.isdir(vllm_integration_dir):
        print(f"❌ Directorio vllm_integration no encontrado: {vllm_integration_dir}")
        return 1
    
    if not os.path.isdir(vllm_source_dir):
        print(f"❌ Directorio vllm-source-modified no encontrado: {vllm_source_dir}")
        return 1
    
    # Añadir paths al PYTHONPATH
    python_paths = [vllm_source_dir, arm_axion_dir]
    python_path_str = ':'.join(python_paths)
    
    # Verificar si el puerto está en uso y liberarlo si es necesario
    if check_port(args.port):
        print(f"⚠️  El puerto {args.port} está en uso. Liberando puerto...")
        if kill_process_on_port(args.port):
            print(f"✅ Puerto {args.port} liberado")
            time.sleep(2)  # Esperar un momento para que el puerto se libere
        else:
            print(f"⚠️  No se pudo liberar completamente el puerto {args.port}, pero continuando...")
    else:
        print(f"✅ Puerto {args.port} está libre")
    
    # Cambiar al directorio correcto
    os.chdir(vllm_integration_dir)
    
    # Configurar ambiente
    env = setup_environment()
    
    # Iniciar servidor
    print(f"🚀 Iniciando servidor ARM-Axion en puerto {args.port}...")
    print(f"   Endpoint: http://{args.host}:{args.port}")
    print(f"   Configuración: {args.config}")
    print("   Presiona Ctrl+C para detener")
    print()
    
    try:
        # Comando para iniciar el servidor
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "inference_server:app", 
            "--host", args.host,
            "--port", str(args.port),
            "--log-level", "info"
        ]
        
        # Iniciar proceso
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=vllm_integration_dir
        )
        
        print(f"✅ Servidor iniciado con PID: {process.pid}")
        print(f"🌐 Accede a: http://{args.host}:{args.port}")
        print()
        
        # Esperar a que el proceso termine
        process.wait()
        
    except KeyboardInterrupt:
        print(f"\\n⚠️  Interrupción por usuario")
        if process.poll() is None:  # Si el proceso aún está corriendo
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                print("💀 Servidor forzadamente detenido")
            print("✅ Servidor detenido")
    except Exception as e:
        print(f"❌ Error al iniciar servidor: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())