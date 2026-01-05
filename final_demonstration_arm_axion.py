#!/usr/bin/env python3
"""
DEMOSTRACIÓN FINAL - SISTEMA ARM-Axion CON vLLM Y 5 MODELOS
"""

import sys
import os
import json
import requests
import time

def demonstrate_arm_axion_functionality():
    print("🎯 DEMOSTRACIÓN FINAL DEL SISTEMA ARM-Axion")
    print("="*70)
    print()
    
    # Verificar detección de plataforma ARM
    print("1. ✅ DETECCIÓN DE PLATAFORMA ARM-Axion:")
    print("   - Verificando que ARM64 se detecte como CPU...")
    
    sys.path.insert(0, '/home/elect/capibara6/vllm-source-modified')
    try:
        from vllm.platforms import current_platform
        print(f"     Plataforma: {current_platform}")
        print(f"     Tipo de dispositivo: {current_platform.device_type}")
        print(f"     ¿Es CPU?: {current_platform.is_cpu()}")
        
        if current_platform.is_cpu() and current_platform.device_type == "cpu":
            print("     ✅ Detección ARM-Axion: CORRECTA")
            platform_ok = True
        else:
            print("     ❌ Detección ARM-Axion: INCORRECTA")
            platform_ok = False
            
    except Exception as e:
        print(f"     ❌ Error verificando plataforma: {e}")
        platform_ok = False
    
    print()
    
    # Verificar estado del servidor
    print("2. ✅ ESTADO DEL SERVIDOR ARM-Axion:")
    
    server_url = "http://localhost:8081"
    try:
        response = requests.get(f"{server_url}/", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print(f"     Servidor: {info.get('name', 'Desconocido')}")
            print(f"     Backend: {info.get('backend', 'Desconocido')}")
            print(f"     Plataforma: {info.get('platform', 'Desconocido')}")
            print(f"     Modelos disponibles: {info.get('models_available', 0)}")
            print(f"     Modelos cargados: {info.get('models_loaded', 0)}")
            server_info_ok = True
        else:
            print(f"     ❌ Servidor no responde en {server_url}")
            server_info_ok = False
    except Exception as e:
        print(f"     ❌ Error conectando al servidor: {e}")
        server_info_ok = False
    
    print()
    
    # Verificar health
    print("3. ✅ VERIFICACIÓN DE SALUD:")
    try:
        health_response = requests.get(f"{server_url}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"     Estado: {health_data['status']}")
            print(f"     Modelos cargados: {health_data['models_loaded']}")
            print(f"     ✅ Servidor saludable")
            health_ok = True
        else:
            print(f"     ❌ Servidor no saludable: {health_response.status_code}")
            health_ok = False
    except Exception as e:
        print(f"     ❌ Error verificando salud: {e}")
        health_ok = False
    
    print()
    
    # Verificar modelos disponibles
    print("4. ✅ VERIFICACIÓN DE MODELOS ARM-Axion:")
    try:
        models_response = requests.get(f"{server_url}/models", timeout=10)
        if models_response.status_code == 200:
            models_data = models_response.json()
            models = models_data.get("models", [])
            
            print(f"     Modelos disponibles: {len(models)}")
            for model in models:
                print(f"     - {model['id']}: {model['status']}")
            
            expected_models = {
                "phi4-fast", 
                "qwen25-coder", 
                "mistral7b-balanced", 
                "gemma3-27b", 
                "gptoss-20b"
            }
            
            actual_models = {model["id"] for model in models}
            
            if expected_models.issubset(actual_models):
                print(f"     ✅ Todos los modelos ARM-Axion encontrados: {len(expected_models)}")
                models_ok = True
            else:
                missing = expected_models - actual_models
                print(f"     ⚠️  Modelos faltantes: {missing}")
                models_ok = False
        else:
            print(f"     ❌ No se pudieron obtener modelos: {models_response.status_code}")
            models_ok = False
    except Exception as e:
        print(f"     ❌ Error obteniendo modelos: {e}")
        models_ok = False

    print()

    print("="*70)
    print("📋 RESUMEN DE VERIFICACIÓN ARM-Axion")
    print("="*70)

    print(f"   Detección de plataforma ARM: {'✅' if platform_ok else '❌'}")
    print(f"   Información del servidor: {'✅' if server_info_ok else '❌'}")
    print(f"   Salud del servidor: {'✅' if health_ok else '❌'}")
    print(f"   Modelos ARM-Axion disponibles: {'✅' if models_ok else '❌'}")

    all_checks_passed = all([platform_ok, server_info_ok, health_ok, models_ok])

    print()
    if all_checks_passed:
        print("🎉 ¡TODOS LOS ELEMENTOS CLAVE FUNCIONAN CORRECTAMENTE!")
        print()
        print("✅ SISTEMA ARM-Axion CON vLLM Y 5 MODELOS:")
        print("   • Detección correcta de ARM64 como plataforma CPU")
        print("   • Servidor inicializado y respondiendo en puerto 8081")
        print("   • 5 modelos ARM-Axion disponibles:")
        print("     - phi4-fast (respuesta rápida)")
        print("     - qwen25-coder (experto en código)")
        print("     - mistral7b-balanced (equilibrado)")
        print("     - gemma3-27b (contexto largo)")
        print("     - gptoss-20b (razonamiento complejo)")
        print("   • Backend clásico de vLLM con parches ARM")
        print("   • API REST disponible con compatibilidad OpenAI")
        print()
        print("🚀 ¡EL SISTEMA ARM-Axion ESTÁ LISTO PARA USO!")
    else:
        print("❌ Algunos componentes no están funcionando correctamente")
        if not platform_ok:
            print("   - Problema con detección de plataforma ARM64")
        if not server_info_ok:
            print("   - Problema con el servidor")
        if not health_ok:
            print("   - Problema con salud del servidor")
        if not models_ok:
            print("   - Problema con disponibilidad de modelos")

    print("="*70)

    return all_checks_passed


def main():
    print("🔍 INICIANDO DEMOSTRACIÓN FINAL DEL SISTEMA ARM-Axion")
    print("   Con vLLM multi-modelo y 5 modelos ARM-Axion optimizados")
    print("   Google Cloud C4A-standard-32 ARM Axion")
    print()
    
    success = demonstrate_arm_axion_functionality()
    
    if success:
        print()
        print("🎊 ¡VERIFICACIÓN COMPLETA EXITOSA!")
        print("💡 El sistema ARM-Axion con vLLM y los 5 modelos está completamente funcional")
    else:
        print()
        print("⚠️  La verificación encontró problemas que necesitan revisión")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)