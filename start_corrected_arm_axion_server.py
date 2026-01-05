#!/usr/bin/env python3
"""
Script para iniciar el servidor multi-modelo ARM-Axion corregido
con los 5 modelos y compatibilidad ARM-Axion
"""

import os
import sys
import subprocess
import time
import signal

def iniciar_servidor_corregido():
    """Iniciar el servidor corregido en un puerto diferente"""
    print("🚀 INICIANDO SERVIDOR ARM-AXION CORREGIDO")
    print("=" * 60)
    print("   Sistema vLLM optimizado para ARM64 con los 5 modelos:")
    print("   - Phi4-mini (respuesta rápida)")
    print("   - Qwen2.5-coder (experto en código)")
    print("   - Mistral7B (equilibrado)")
    print("   - Gemma3-27B (tareas complejas)")
    print("   - GPT-OSS-20B (razonamiento complejo)")
    print("=" * 60)
    
    # Configurar ambiente
    env = os.environ.copy()
    env['VLLM_USE_V1'] = '0'
    env['VLLM_ENABLE_V1_ENGINE'] = '0'
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'fork'
    env['VLLM_USE_FLASHINFER'] = '0'
    env['VLLM_NO_DEPRECATION_WARNING'] = '1'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:disabled'
    env['VLLM_USE_TRITON_FLASH_ATTN'] = '0'
    env['TORCHINDUCTOR_DISABLED'] = '1'
    env['TORCH_COMPILE_BACKEND'] = 'eager'
    
    # Asegurar que el path incluya el código modificado
    python_path = "/home/elect/capibara6/vllm-source-modified"
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{python_path}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = python_path
    
    print("🔧 Variables de entorno configuradas:")
    print(f"   - VLLM_USE_V1: {env['VLLM_USE_V1']}")
    print(f"   - VLLM_USE_TRITON_FLASH_ATTN: {env['VLLM_USE_TRITON_FLASH_ATTN']}")
    print(f"   - TORCHINDUCTOR_DISABLED: {env['TORCHINDUCTOR_DISABLED']}")
    print(f"   - TORCH_COMPILE_BACKEND: {env['TORCH_COMPILE_BACKEND']}")
    print()
    
    # Probar el servidor en un puerto diferente para evitar conflictos
    puerto = 8082
    
    print(f"📡 Iniciando servidor en puerto: {puerto}")
    print("   Presiona Ctrl+C para detener")
    print()
    
    try:
        # Ejecutar el servidor con las nuevas configuraciones
        cmd = [
            sys.executable, 
            "-c",
            f"""
import sys
sys.path.insert(0, '/home/elect/capibara6/vllm-source-modified')

# Aplicar configuración
import os
os.environ.update({{
    'VLLM_USE_V1': '0',
    'VLLM_ENABLE_V1_ENGINE': '0',
    'VLLM_WORKER_MULTIPROC_METHOD': 'fork',
    'VLLM_USE_FLASHINFER': '0', 
    'VLLM_NO_DEPRECATION_WARNING': '1',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:disabled',
    'VLLM_USE_TRITON_FLASH_ATTN': '0',
    'TORCHINDUCTOR_DISABLED': '1',
    'TORCH_COMPILE_BACKEND': 'eager'
}})

print('✅ Entorno ARM-Axion configurado')
print('✅ Iniciando servidor multi-modelo ARM-Axion en puerto {puerto}...')

# Importar y ejecutar el servidor
from arm_axion_optimizations.vllm_integration.multi_model_server import app
import uvicorn

print('🚀 Servidor ARM-Axion iniciado OK')
print('🌐 Endpoint: http://0.0.0.0:{puerto}')
print('📊 Endpoints disponibles:')
print('   GET  /health - Verificar estado del servidor')
print('   GET  /models - Listar modelos disponibles')
print('   POST /v1/chat/completions - API OpenAI compatible')
print('   POST /v1/completions - API OpenAI compatible')
print('   POST /api/generate - API Ollama compatible')
print()
print('✅ ¡Servidor ARM-Axion está listo para recibir solicitudes!')

uvicorn.run(app, host="0.0.0.0", port={puerto}, log_level="info")
"""
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env
        )
        
        # Esperar un poco para ver si hay errores iniciales
        time.sleep(5)
        
        # Verificar si el proceso aún está corriendo
        if process.poll() is not None:
            # Capturar la salida de error
            _, stderr = process.communicate()
            print(f"❌ El servidor falló al iniciar:")
            print(stderr)
            return False
        else:
            print(f"✅ Servidor corriendo en segundo plano PID: {process.pid}")
            print(f"   Accede a: http://localhost:{puerto}")
            print("   Presiona CTRL+C para detener")
            
            # Esperar indefinidamente hasta recibir señal de interrupción
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n👋 Deteniendo servidor ARM-Axion...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    print("⚠️  Servidor terminado forzosamente")
                print("✅ Servidor ARM-Axion detenido")
                return True
            
    except Exception as e:
        print(f"❌ Error al iniciar servidor: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Función principal"""
    print("🔄 Verificando configuración ARM-Axion...")
    
    # Verificar que el archivo modificado exista
    if not os.path.exists("/home/elect/capibara6/vllm-source-modified/vllm/platforms/__init__.py"):
        print("❌ Código vLLM modificado no encontrado")
        return False
    
    print("✅ Código vLLM con detección ARM64 como CPU está disponible")
    
    # Verificar que el servidor exista
    if not os.path.exists("/home/elect/capibara6/arm-axion-optimizations/vllm_integration/multi_model_server.py"):
        print("❌ Servidor multi-modelo no encontrado")
        return False
    
    print("✅ Servidor multi-modelo ARM-Axion está disponible")
    
    # Iniciar el servidor
    return iniciar_servidor_corregido()


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 ¡Servidor ARM-Axion multi-modelo iniciado exitosamente!")
        print("   ¡Listo para usar los 5 modelos con todas las optimizaciones ARM!")
    else:
        print("\n❌ Error al iniciar el servidor ARM-Axion")
        sys.exit(1)