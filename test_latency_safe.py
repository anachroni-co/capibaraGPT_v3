#!/usr/bin/env python3
"""
Script de pruebas de latencia seguro para servidor multi-modelo ARM Axion
Prueba cada modelo individualmente con pausas entre pruebas para evitar saturación
"""

import requests
import time
import json
from typing import Dict, List, Optional
import statistics

# Configuración
SERVER_URL = "http://localhost:8082"
TEST_PROMPTS = {
    "simple": "Hola, ¿qué tal?",
    "technical": "Explica qué es el algoritmo de ordenamiento quicksort",
    "coding": "Escribe una función Python para calcular números primos",
    "multilingual": "Translate 'Hello, how are you?' to Spanish",
    "complex": "Analiza las ventajas y desventajas de usar arquitecturas de microservicios vs monolitos"
}

def get_available_experts() -> List[str]:
    """Obtiene la lista de expertos disponibles"""
    try:
        response = requests.get(f"{SERVER_URL}/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        else:
            print(f"❌ Error al obtener modelos: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return []

def get_loaded_experts() -> List[str]:
    """Obtiene la lista de expertos actualmente cargados"""
    try:
        response = requests.get(f"{SERVER_URL}/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("models_loaded", [])
        else:
            print(f"❌ Error al obtener estado de modelos: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error de conexión obteniendo estado: {e}")
        return []

def test_model_latency(model_id: str, prompt: str, prompt_type: str, max_tokens: int = 30) -> Dict:
    """Prueba la latencia de un modelo específico"""
    print(f"   Prueba {prompt_type} para {model_id}... ", end="", flush=True)

    try:
        start_time = time.time()

        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1
            },
            timeout=60  # 1 minuto timeout para evitar bloqueos largos
        )

        total_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                
                # Extraer métricas de rendimiento si están disponibles
                performance = result.get("performance", {})
                tokens_per_second = performance.get("tokens_per_second", 0)
                
                print(f"✅ ({total_time:.2f}s, {tokens_per_second:.2f} t/s)")
                return {
                    "model": model_id,
                    "prompt_type": prompt_type,
                    "success": True,
                    "latency": total_time,
                    "tokens_per_second": tokens_per_second,
                    "response_preview": content[:60] + "..." if len(content) > 60 else content
                }
            else:
                print(f"❌ (Respuesta sin choices)")
                return {
                    "model": model_id,
                    "prompt_type": prompt_type,
                    "success": False,
                    "latency": total_time,
                    "error": "No choices in response",
                    "response_preview": ""
                }
        else:
            print(f"❌ (Status {response.status_code})")
            return {
                "model": model_id,
                "prompt_type": prompt_type,
                "success": False,
                "latency": total_time,
                "error": f"HTTP {response.status_code}",
                "response_preview": response.text[:100]
            }

    except requests.exceptions.Timeout:
        print(f"⏰ (Timeout)")
        return {
            "model": model_id,
            "prompt_type": prompt_type,
            "success": False,
            "latency": 60.0,  # Valor máximo de timeout
            "error": "Timeout",
            "response_preview": ""
        }
    except Exception as e:
        print(f"💥 (Error: {str(e)[:30]})")
        return {
            "model": model_id,
            "prompt_type": prompt_type,
            "success": False,
            "latency": 0,
            "error": str(e),
            "response_preview": ""
        }

def run_safe_latency_test():
    """Ejecuta una prueba de latencia segura sin saturar el servidor"""
    print("🔍 Obteniendo modelos disponibles...")
    available_models = get_available_experts()
    
    print("🔍 Obteniendo modelos ya cargados...")
    loaded_models = get_loaded_experts()
    
    print(f"✅ Modelos disponibles: {available_models}")
    print(f"✅ Modelos ya cargados: {loaded_models}")

    # Definir un modelo para probar (el que ya está cargado y funciona)
    if "aya_expanse_multilingual" in available_models:
        test_model = "aya_expanse_multilingual"
    elif loaded_models:
        test_model = loaded_models[0]
    elif available_models:
        test_model = available_models[0]
    else:
        print("❌ No se encontraron modelos disponibles")
        return

    print(f"\n🧪 Iniciando prueba de latencia para: {test_model}")
    print(f"   (Esta prueba es segura y no sobrecargará el servidor)")

    results = []
    successful_latencies = []

    for prompt_type, prompt in TEST_PROMPTS.items():
        # Realizar la prueba
        result = test_model_latency(test_model, prompt, prompt_type)
        results.append(result)
        
        if result["success"]:
            successful_latencies.append(result["latency"])
        
        # Pausa entre pruebas para evitar saturar el servidor
        if prompt_type != list(TEST_PROMPTS.keys())[-1]:  # No hacer pausa después de la última prueba
            print("   ⏳ Esperando 3 segundos entre pruebas...")
            time.sleep(3)

    # Resultados
    print(f"\n📊 RESULTADOS PARA {test_model}")
    print("-" * 50)
    
    for result in results:
        status = "✅" if result["success"] else "❌"
        latency = f"{result['latency']:.2f}s" if result["success"] else "N/A"
        tps = f"{result['tokens_per_second']:.2f}" if result["success"] and result.get("tokens_per_second", 0) > 0 else "N/A"
        
        print(f"   {status} {result['prompt_type']}: {latency} ({tps} t/s)")
        if result["success"]:
            print(f"     → '{result['response_preview']}'")

    if successful_latencies:
        avg_latency = statistics.mean(successful_latencies)
        min_latency = min(successful_latencies)
        max_latency = max(successful_latencies)
        print(f"\n📈 ESTADÍSTICAS GENERALES:")
        print(f"   Promedio latencia: {avg_latency:.2f}s")
        print(f"   Mínimo latencia: {min_latency:.2f}s")
        print(f"   Máximo latencia: {max_latency:.2f}s")
    else:
        print(f"\n❌ No se obtuvieron resultados exitosos para {test_model}")

    print(f"\n✅ Prueba de latencia completada sin saturar el servidor.")

def main():
    print("🚀 Iniciando prueba de latencia SEGURA para servidor multimodelo...")
    print(f"   Servidor: {SERVER_URL}")
    print()

    run_safe_latency_test()

if __name__ == "__main__":
    main()