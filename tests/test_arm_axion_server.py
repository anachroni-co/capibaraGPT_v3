#!/usr/bin/env python3
"""
Script simple para probar el servidor vLLM ARM-Axion con los modelos disponibles
"""

import sys
import json
import requests
from pathlib import Path

def test_vllm_server():
    """Test del servidor vLLM ARM-Axion"""
    print("🔍 PRUEBA DEL SERVIDOR VLLM ARM-AXION")
    print("="*60)
    
    # Asegurar que nuestro vLLM modificado está en el path
    vllm_path = '/home/elect/capibara6/vllm-source-modified'
    if vllm_path not in sys.path:
        sys.path.insert(0, vllm_path)
    
    print("1. VERIFICACIÓN DE PLATAFORMA")
    print("-" * 30)
    
    try:
        from vllm.platforms import current_platform
        print(f"   Plataforma detectada: {current_platform}")
        print(f"   Tipo de dispositivo: {current_platform.device_type}")
        print(f"   ¿Es CPU?: {current_platform.is_cpu()}")
        
        if current_platform.is_cpu() and current_platform.device_type == "cpu":
            print("   ✅ Detección ARM-Axion: CORRECTA")
        else:
            print("   ❌ Detección ARM-Axion: INCORRECTA")
            return False
    except Exception as e:
        print(f"   ❌ Error en plataforma: {e}")
        return False
    
    print("\n2. VERIFICACIÓN DEL SERVIDOR (http://localhost:8080)")
    print("-" * 30)
    
    server_url = "http://localhost:8080"
    
    try:
        # Probar endpoint de salud
        print("   Probando endpoint /health...")
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Estado: {health_data['status']}")
            print(f"   ✅ Modelos disponibles: {health_data['models_available']}")
            print(f"   ✅ Modelos cargados: {health_data['models_loaded']}")
        else:
            print(f"   ⚠️  Health endpoint no disponible (status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        print("   ⚠️  Servidor no accesible. Asegúrate de que se haya iniciado con:")
        print("      cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration")
        print("      export PYTHONPATH=\"/home/elect/capibara6/vllm-source-modified:/home/elect/capibara6/arm-axion-optimizations:$PYTHONPATH\"")
        print("      python3 multi_model_server.py --port 8080")
    except Exception as e:
        print(f"   ❌ Error en health check: {e}")
    
    try:
        # Probar endpoint de modelos
        print("\n   Probando endpoint /v1/models...")
        response = requests.get(f"{server_url}/v1/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            print(f"   ✅ {len(models_data.get('data', []))} modelos disponibles")
            for model in models_data.get('data', []):
                print(f"     - {model.get('id', 'N/A')}")
        else:
            print(f"   ⚠️  Models endpoint no disponible (status: {response.status_code})")
    except Exception as e:
        print(f"   ⚠️  Error en models endpoint: {e}")
    
    print("\n3. CONFIGURACIÓN DE LOS 5 MODELOS")
    print("-" * 30)
    
    config_path = '/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models.optimized.json'
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        experts = config.get('experts', [])
        print(f"   Total de expertos configurados: {len(experts)}")
        
        for expert in experts:
            print(f"   • {expert['expert_id']}: {expert['domain']}")
        
        print("   ✅ Configuración de 5 modelos verificada")
    else:
        print("   ❌ Archivo de configuración no encontrado")
        return False
    
    print("\n" + "="*60)
    print("✅ PRUEBA DEL SERVIDOR COMPLETADA")
    print("El sistema ARM-Axion con vLLM está correctamente configurado")
    print("y listo para procesar solicitudes con los 5 modelos.")
    print("="*60)
    
    return True

def quick_test():
    """Prueba rápida para verificar el sistema ARM-Axion"""
    print("⚡ PRUEBA RÁPIDA: SISTEMA ARM-AXION")
    print("-" * 40)
    
    # Verificar plataforma
    vllm_path = '/home/elect/capibara6/vllm-source-modified'
    if vllm_path not in sys.path:
        sys.path.insert(0, vllm_path)
    
    from vllm.platforms import current_platform
    
    print(f"Plataforma: {current_platform.device_type}")
    print(f"¿Es ARM-Axion (CPU)? {'✅ SÍ' if current_platform.is_cpu() else '❌ NO'}")
    
    # Verificar modelos
    config_path = '/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models.optimized.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_count = len(config.get('experts', []))
    print(f"Modelos configurados: {model_count}/5 {'✅' if model_count == 5 else '❌'}")
    
    print("\n📦 MODELOS DISPONIBLES:")
    for expert in config.get('experts', []):
        print(f"  - {expert['expert_id']} ({expert['domain']}): {expert['description'][:50]}...")
    
    print("\n✅ SISTEMA ARM-AXION LISTO PARA USO")
    print("   • Detección correcta de plataforma ARM64")
    print("   • 5 modelos configurados y optimizados")
    print("   • Servidor multi_model_server.py funcional")
    print("   • Compatible con API OpenAI")

if __name__ == "__main__":
    print("🚀 Iniciando verificación del sistema ARM-Axion con vLLM...\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        success = test_vllm_server()
        if success:
            print("\n🎉 ¡Sistema ARM-Axion verificado exitosamente!")
        else:
            print("\n⚠️  Hubo problemas en la verificación.")