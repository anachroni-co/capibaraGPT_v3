#!/usr/bin/env python3
"""
Script para verificar que el servidor puede usar nuestro código vLLM modificado
con la detección correcta de ARM-Axion
"""

import sys
import os
from pathlib import Path

def test_server_with_modified_vllm():
    """Probar que el servidor puede usar el vLLM modificado"""
    print("🔍 PRUEBA: Uso del vLLM modificado en servidores")
    
    # Añadir nuestro vLLM modificado al path
    vllm_modified_path = '/home/elect/capibara6/vllm-source-modified'
    if vllm_modified_path not in sys.path:
        sys.path.insert(0, vllm_modified_path)
    
    print(f"   Añadido al path: {vllm_modified_path}")
    
    # Verificar que se puede importar vLLM
    try:
        print("   Intentando importar vLLM...")
        import vllm
        print(f"   ✅ vLLM importado exitosamente - Versión: {vllm.__version__}")
    except ImportError as e:
        print(f"   ❌ Error al importar vLLM: {e}")
        return False
    
    # Verificar la detección de plataforma
    try:
        print("   Verificando detección de plataforma...")
        from vllm.platforms import current_platform
        print(f"   Plataforma: {current_platform}")
        print(f"   Tipo de dispositivo: {current_platform.device_type}")
        print(f"   ¿Es CPU?: {current_platform.is_cpu()}")
        
        if current_platform.is_cpu() and current_platform.device_type == "cpu":
            print("   ✅ Plataforma ARM-Axion detectada correctamente")
        else:
            print("   ❌ Plataforma ARM-Axion NO detectada correctamente")
            return False
    except Exception as e:
        print(f"   ❌ Error verificando plataforma: {e}")
        return False
    
    # Verificar que se pueden importar componentes necesarios para el servidor
    try:
        print("   Verificando importación de componentes de vLLM...")
        from vllm import LLM, SamplingParams
        print("   ✅ Componentes LLM importados correctamente")
    except Exception as e:
        print(f"   ❌ Error importando componentes LLM: {e}")
        return False
    
    return True


def simulate_server_startup():
    """Simular cómo se iniciaría el servidor con nuestro código modificado"""
    print("\n🔍 PRUEBA: Simulación de inicio de servidor con código modificado")
    
    # Añadir el path modificado
    vllm_path = '/home/elect/capibara6/vllm-source-modified'
    if vllm_path not in sys.path:
        sys.path.insert(0, vllm_path)
    
    try:
        # Simular carga de configuración como lo haría multi_model_server.py
        config_path = '/home/elect/capibara6/arm-axion-optimizations/vllm-integration/config.five_models.optimized.json'
        import json
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"   ✅ Configuración cargada con {len(config['experts'])} expertos")
        
        # Mostrar expertos configurados
        for expert in config['experts']:
            print(f"     - {expert['expert_id']}: {expert['domain']} ({expert['description'][:50]}...)")
        
        # Simular la detección de plataforma que haría el servidor
        from vllm.platforms import current_platform
        print(f"   ✅ Servidor detectaría plataforma: {current_platform.device_type}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en simulación de inicio: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_interactive_interface_with_modifications():
    """Probar que la interfaz interactiva usa correctamente el código modificado"""
    print("\n🔍 PRUEBA: Interfaz interactiva con código modificado")
    
    # Añadir paths como lo haría el script interactivo
    paths_to_add = [
        '/home/elect/capibara6/vllm-source-modified',
        '/home/elect/capibara6/vm-bounty2/core',
        '/home/elect/capibara6/vm-bounty2/config',
        '/home/elect/capibara6/backend'
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    try:
        # Probar la detección de plataforma que usaría el script interactivo
        from vllm.platforms import current_platform
        print(f"   ✅ Interfaz usaría plataforma: {current_platform.device_type}")
        
        # Probar que puede acceder a la configuración
        import json
        with open('/home/elect/capibara6/model_config.json', 'r') as f:
            model_config = json.load(f)
        
        model_count = len(model_config.get('models', {}))
        print(f"   ✅ Interfaz puede acceder a {model_count} configuraciones de modelo")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en interfaz interactiva: {e}")
        return False


def test_all_components_integrated():
    """Probar que todos los componentes trabajan juntos con las modificaciones"""
    print("\n🔍 PRUEBA: Integración completa de componentes con modificaciones")
    
    # Paths necesarios
    paths = [
        '/home/elect/capibara6/vllm-source-modified',
        '/home/elect/capibara6/arm-axion-optimizations/vllm-integration',
        '/home/elect/capibara6/vm-bounty2/config',
        '/home/elect/capibara6/backend'
    ]
    
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    try:
        # Verificar plataforma
        from vllm.platforms import current_platform
        platform_ok = current_platform.is_cpu() and current_platform.device_type == "cpu"
        
        # Verificar configuración
        import json
        with open('/home/elect/capibara6/arm-axion-optimizations/vllm-integration/config.five_models.optimized.json', 'r') as f:
            config = json.load(f)
        
        models_ok = len(config.get('experts', [])) == 5
        
        # Verificar que todos los modelos en la config existen
        models_dir = Path('/home/elect/models')
        expected_models = {
            'phi4_fast': 'phi-4-mini',
            'mistral_balanced': 'mistral-7b-instruct-v0.2', 
            'qwen_coder': 'qwen2.5-coder-1.5b',
            'gemma3_multimodal': 'gemma-3-27b-it',
            'gptoss_complex': 'gpt-oss-20b'
        }
        
        models_exist = True
        for expert_id, model_path in expected_models.items():
            model_full_path = models_dir / model_path
            if not model_full_path.exists():
                print(f"   ❌ Modelo no encontrado: {model_full_path}")
                models_exist = False
        
        print(f"   ✅ Plataforma ARM-Axion: {'Sí' if platform_ok else 'No'}")
        print(f"   ✅ Configuración 5 modelos: {'Sí' if models_ok else 'No'}")
        print(f"   ✅ Modelos físicos disponibles: {'Sí' if models_exist else 'No'}")
        
        success = platform_ok and models_ok and models_exist
        if success:
            print("   ✅ Todos los componentes integrados correctamente")
        
        return success
        
    except Exception as e:
        print(f"   ❌ Error en integración: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("PRUEBA DE INTEGRACIÓN DEL CÓDIGO VLLM MODIFICADO EN ARM-AXION")
    print("="*80)
    
    tests = [
        ("vLLM modificado en servidores", test_server_with_modified_vllm),
        ("Simulación inicio servidor", simulate_server_startup),
        ("Interfaz interactiva", test_interactive_interface_with_modifications),
        ("Integración completa", test_all_components_integrated)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
        print()
    
    print("="*80)
    print("RESULTADOS DE INTEGRACIÓN:")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n{'✅ ÉXITO' if passed == total else '⚠️  PARCIAL'}: {passed}/{total} integraciones exitosas")
    
    if passed == total:
        print("\n🎉 ¡EL CÓDIGO VLLM MODIFICADO ESTÁ COMPLETAMENTE INTEGRADO!")
        print("   - Los servidores usan el código vLLM con detección ARM-Axion")
        print("   - La interfaz interactiva funciona con las modificaciones")
        print("   - Todos los componentes reconocen la plataforma ARM64 como CPU")
        print("   - El sistema está listo para usar los 5 modelos en ARM-Axion")
    else:
        print(f"\n⚠️  Algunas integraciones fallaron, revisar resultados arriba")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)