#!/usr/bin/env python3
"""
Script para verificar conexión con el servidor BB
Prueba los 3 modelos: gpt-oss-20b, phi-mini, mixtral
"""

import requests
import json
import sys
from models_config import MODELS_CONFIG, get_active_models

def test_model_connection(model_id, config):
    """Prueba la conexión con un modelo específico"""
    print(f"\n{'='*60}")
    print(f"🧪 Probando: {config['name']}")
    print(f"{'='*60}")
    print(f"URL: {config['server_url']}")
    print(f"Hardware: {config['hardware']}")
    print(f"Status: {config['status']}")

    try:
        # Intentar health check primero
        health_url = config['server_url'].replace('/completion', '/health')
        print(f"\n📡 Health check: {health_url}")

        try:
            health_response = requests.get(health_url, timeout=5)
            if health_response.ok:
                print("✅ Health check OK")
                print(f"   {health_response.json()}")
            else:
                print(f"⚠️  Health check respondió con status {health_response.status_code}")
        except Exception as e:
            print(f"⚠️  Health check no disponible: {e}")

        # Test de completion simple
        print(f"\n📡 Test de completion...")
        test_payload = {
            'prompt': '¿Qué es Python?',
            'n_predict': 50,
            'temperature': 0.7,
            'top_p': 0.9,
            'stream': False
        }

        response = requests.post(
            config['server_url'],
            json=test_payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )

        if response.ok:
            data = response.json()
            content = data.get('content', '')
            tokens = data.get('tokens_predicted', 0)

            print(f"✅ Completion exitoso")
            print(f"   Tokens generados: {tokens}")
            print(f"   Respuesta: {content[:100]}..." if len(content) > 100 else f"   Respuesta: {content}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text[:200]}")
            return False

    except requests.exceptions.Timeout:
        print(f"❌ Timeout: El servidor no respondió en 30 segundos")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Error de conexión: {e}")
        print(f"   Verifica que el servidor esté corriendo en {config['server_url']}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def main():
    print("="*60)
    print("🔌 Test de Conexión con Backend BB")
    print("="*60)
    print(f"\nProbando conexión con modelos del servidor BB...")
    print(f"IP: 34.175.215.109")
    print(f"Puertos: 8080 (gpt-oss-20b), 8081 (phi), 8082 (mixtral)")

    # Obtener modelos activos
    active_models = get_active_models()

    if not active_models:
        print("\n❌ No hay modelos activos configurados")
        sys.exit(1)

    print(f"\n📋 Modelos activos: {len(active_models)}")
    for model_id in active_models:
        config = MODELS_CONFIG[model_id]
        print(f"   • {config['name']} ({model_id})")

    # Probar cada modelo
    results = {}
    for model_id in active_models:
        config = MODELS_CONFIG[model_id]
        success = test_model_connection(model_id, config)
        results[model_id] = success

    # Resumen
    print(f"\n{'='*60}")
    print("📊 Resumen de Resultados")
    print("="*60)

    total = len(results)
    successful = sum(1 for success in results.values() if success)
    failed = total - successful

    print(f"\nTotal de modelos probados: {total}")
    print(f"✅ Exitosos: {successful}")
    print(f"❌ Fallidos: {failed}")

    print(f"\n📝 Detalle:")
    for model_id, success in results.items():
        status = "✅ OK" if success else "❌ FAIL"
        config = MODELS_CONFIG[model_id]
        print(f"   {status} - {config['name']} ({model_id})")

    # Recomendaciones
    if failed > 0:
        print(f"\n⚠️  Recomendaciones:")
        print(f"   1. Verifica que el servidor BB esté corriendo")
        print(f"   2. Verifica que los modelos estén cargados en los puertos correctos:")
        for model_id, success in results.items():
            if not success:
                config = MODELS_CONFIG[model_id]
                port = config['server_url'].split(':')[-1].split('/')[0]
                print(f"      • {config['name']}: puerto {port}")
        print(f"   3. Verifica conectividad de red: ping 34.175.215.109")
        print(f"   4. Verifica logs del servidor BB")
    else:
        print(f"\n🎉 ¡Todos los modelos respondieron correctamente!")
        print(f"   El sistema está listo para usar")

    print("\n" + "="*60)

    # Exit code
    sys.exit(0 if failed == 0 else 1)

if __name__ == '__main__':
    main()
