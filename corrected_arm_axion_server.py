#!/usr/bin/env python3
"""
Corrección del problema de compilador Triton en ARM-Axion
y levantamiento del servidor vLLM con los 5 modelos
"""

import os
import sys

def setup_environment():
    """Configurar el entorno para ARM-Axion con vLLM"""
    
    print("🔧 CONFIGURANDO ENTORNO ARM-AXION")
    print("="*50)
    
    # Variables críticas para ARM-Axion
    os.environ['VLLM_USE_V1'] = '0'  # Deshabilitar V1 engine para evitar problemas
    os.environ['VLLM_ENABLE_V1_ENGINE'] = '0'  # Deshabilitar V1 engine
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['TORCHINDUCTOR_COMPILE_CONFIG'] = '{"enable_auto_functionalized_v2": false}'
    os.environ['TORCH_COMPILE_BACKEND'] = 'aot_eager'  # Usar backend más compatible
    
    # Evitar el uso de Triton que causa problemas
    os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '0'
    os.environ['TORCHINDUCTOR_DISABLED'] = '1'  # Deshabilitar inductor si causa problemas
    
    print(f"✓ VLLM_USE_V1: {os.environ.get('VLLM_USE_V1')}")
    print(f"✓ VLLM_ENABLE_V1_ENGINE: {os.environ.get('VLLM_ENABLE_V1_ENGINE')}")
    print(f"✓ TORCH_COMPILE_BACKEND: {os.environ.get('TORCH_COMPILE_BACKEND')}")
    print(f"✓ TORCHINDUCTOR_DISABLED: {os.environ.get('TORCHINDUCTOR_DISABLED')}")
    
    # Asegurar el path al código modificado
    vllm_path = '/home/elect/capibara6/vllm-source-modified'
    if vllm_path not in sys.path:
        sys.path.insert(0, vllm_path)
    
    print(f"✓ PYTHONPATH incluye: {vllm_path}")


def test_plataforma():
    """Verificar que la plataforma ARM-Axion esté detectada correctamente"""
    print("\n🔍 VERIFICANDO DETECCIÓN DE PLATAFORMA")
    print("-" * 40)
    
    from vllm.platforms import current_platform
    
    print(f"Plataforma: {current_platform}")
    print(f"Tipo de dispositivo: {current_platform.device_type}")
    print(f"¿Es CPU?: {current_platform.is_cpu()}")
    
    if current_platform.is_cpu() and current_platform.device_type == "cpu":
        print("✅ Plataforma ARM-Axion detectada correctamente")
        return True
    else:
        print("❌ Problema con detección de plataforma")
        return False


def start_server(port=8082):
    """Levantar el servidor con configuración ARM compatible"""
    print(f"\n🚀 LEVANTANDO SERVIDOR EN PUERTO {port}")
    print("-" * 40)
    
    # Importar después de configurar entorno
    from multi_model_server import app
    import uvicorn
    
    print("Servidor ARM-Axion listo para iniciar...")
    print(f"  - Puerto: {port}")
    print(f"  - Host: 0.0.0.0") 
    print(f"  - Backend: CPU")
    print(f"  - Modelos: 5 ARM-Axion optimizados")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except OSError as e:
        if "address already in use" in str(e):
            print(f"⚠️  Puerto {port} ya está en uso")
            new_port = port + 1
            print(f"   Intentando con puerto alternativo: {new_port}")
            uvicorn.run(app, host="0.0.0.0", port=new_port, log_level="info")
        else:
            raise e


def show_interactive_info():
    """Mostrar información sobre el script interactivo"""
    print("\n🎮 SCRIPTS INTERACTIVOS DISPONIBLES")
    print("-" * 40)
    
    print("Para probar los 5 modelos interactivamente:")
    print("  cd /home/elect/capibara6")
    print("  python3 interactive_test_interface.py")
    print("")
    print("Opciones del script interactivo:")
    print("  1. Probar modelo individual")
    print("  2. Probar sistema de router semántico") 
    print("  3. Probar sistema de consenso")
    print("  4. Probar todos los modelos con análisis comparativo")
    print("  5. Información del sistema")
    print("  6. Salir")


def main():
    """Función principal para corregir y levantar el servidor"""
    print("🚀 INICIANDO SERVIDOR vLLM ARM-AXION - SOLUCIÓN DE COMPILACIÓN")
    print("   Corrección de problemas con Triton/Inductor en ARM64")
    
    # Configurar entorno
    setup_environment()
    
    # Verificar plataforma
    plataforma_ok = test_plataforma()
    
    if not plataforma_ok:
        print("\n❌ No se puede continuar sin detección correcta de plataforma")
        return False
    
    # Mostrar información sobre el script interactivo
    show_interactive_info()
    
    # Ofrecer iniciar el servidor
    print(f"\n💡 Para levantar el servidor de todos los modelos:")
    print("   python3 corrected_arm_axion_server.py [puerto]")
    print("   (por defecto puerto 8082 para evitar conflictos)")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082, help="Puerto para el servidor")
    args = parser.parse_args()
    
    print(f"\n🔥 Iniciando servidor ARM-Axion en puerto {args.port}...")
    start_server(args.port)


if __name__ == "__main__":
    main()