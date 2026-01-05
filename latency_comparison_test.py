#!/usr/bin/env python3
"""
Script para comparar latencia antes y después de las optimizaciones
Este script solo prueba el modelo que ya está cargado para evitar problemas de lazy loading
"""

import requests
import time
import json
from typing import Dict, List
import statistics

# Configuración
SERVER_URL = "http://localhost:8082"
TEST_PROMPTS = [
    "Hola, ¿cómo estás?",
    "¿Qué es la inteligencia artificial?",
    "Explica brevemente el algoritmo de quicksort",
    "Escribe una función en Python para calcular números primos",
    "Traduce 'Hello world' al español y francés"
]

def test_model_performance(model_id: str = "aya_expanse_multilingual", iterations: int = 5):
    """
    Prueba la latencia del modelo actualmente cargado
    """
    print(f"🚀 Probando rendimiento de {model_id} con {iterations} iteraciones")
    print(f"   Servidor: {SERVER_URL}")

    latencies = []
    tokens_per_second_values = []

    for i in range(iterations):
        prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        print(f"   Iteración {i+1}/{iterations}: '{prompt[:30]}...'", end=" ")

        start_time = time.time()

        try:
            response = requests.post(
                f"{SERVER_URL}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.1
                },
                timeout=120
            )

            total_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                
                # Extraer métricas de rendimiento - verificamos la estructura correcta
                usage = result.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                
                # Calcular tokens por segundo
                tokens_ps = completion_tokens / total_time if total_time > 0 else 0

                latencies.append(total_time)
                tokens_per_second_values.append(tokens_ps)

                print(f"✅ {total_time:.2f}s ({tokens_ps:.2f} tok/s)")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:100]}")
                latencies.append(None)
                tokens_per_second_values.append(None)

        except requests.exceptions.Timeout:
            print("⏰ Timeout")
            latencies.append(None)
            tokens_per_second_values.append(None)
        except Exception as e:
            print(f"💥 Error: {e}")
            latencies.append(None)
            tokens_per_second_values.append(None)

        # Pausa entre pruebas para evitar saturación
        time.sleep(2)

    # Filtrar valores válidos
    valid_latencies = [l for l in latencies if l is not None and l > 0]
    valid_tps = [t for t in tokens_per_second_values if t is not None and t > 0]

    if valid_latencies:
        print(f"\n📊 RESULTADOS PARA {model_id}")
        print(f"   Iteraciones exitosas: {len(valid_latencies)}/{iterations}")
        print(f"   Promedio latencia: {statistics.mean(valid_latencies):.2f}s")
        if len(valid_latencies) > 1:
            print(f"   Desviación estándar: {statistics.stdev(valid_latencies):.2f}s")
        print(f"   Mínimo latencia: {min(valid_latencies):.2f}s")
        print(f"   Máximo latencia: {max(valid_latencies):.2f}s")

        if valid_tps:
            print(f"   Promedio tokens/seg: {statistics.mean(valid_tps):.2f}")
            if len(valid_tps) > 1:
                print(f"   Desviación estándar: {statistics.stdev(valid_tps):.2f}")
            print(f"   Rango tokens/seg: {min(valid_tps):.2f} - {max(valid_tps):.2f}")

        result = {
            "model": model_id,
            "latencies": valid_latencies,
            "tokens_per_second": valid_tps,
            "avg_latency": statistics.mean(valid_latencies) if valid_latencies else 0,
            "min_latency": min(valid_latencies) if valid_latencies else 0,
            "max_latency": max(valid_latencies) if valid_latencies else 0,
            "successful_requests": len(valid_latencies),
            "total_requests": iterations
        }
        
        return result
    else:
        print(f"\n❌ No se obtuvieron resultados válidos para {model_id}")
        return None

def compare_optimization_levels():
    """
    Compara diferentes niveles de optimización
    """
    print("🔍 Obteniendo modelos cargados...")
    try:
        response = requests.get(f"{SERVER_URL}/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            loaded_models = stats.get("models_loaded", [])
            print(f"   Modelos actualmente cargados: {loaded_models}")
            
            if loaded_models:
                # Probar cada modelo cargado
                results = []
                for model_id in loaded_models:
                    result = test_model_performance(model_id)
                    if result:
                        results.append(result)
                
                return results
            else:
                # Si no hay modelos cargados, probar con uno disponible
                print("   No hay modelos cargados, probando con aya_expanse_multilingual...")
                available_response = requests.get(f"{SERVER_URL}/v1/models", timeout=10)
                if available_response.status_code == 200:
                    available = available_response.json()
                    available_models = [m["id"] for m in available.get("data", [])]
                    
                    # Intentar con el primer modelo disponible
                    if available_models:
                        result = test_model_performance(available_models[0])
                        if result:
                            return [result]
        
        print("❌ No se pudo obtener información del servidor")
        return []
        
    except Exception as e:
        print(f"❌ Error obteniendo modelos: {e}")
        return []

def main():
    print("🚀 Comparando rendimiento con optimizaciones ARM-Axion aplicadas")
    print("=" * 70)
    print("   • FP8 KV Cache: Reducción de uso de memoria")
    print("   • Captured Graphs: Menor overhead de compilación")
    print("   • Scheduler tuning: Optimización para latencia")
    print("   • Dtype ajustado a float16: Mayor velocidad en ARM")
    print("=" * 70)
    
    results = compare_optimization_levels()
    
    if results:
        print(f"\n📈 RESUMEN FINAL:")
        for result in results:
            print(f"   {result['model']}:")
            print(f"      - Iteraciones: {result['successful_requests']}/{result['total_requests']}")
            print(f"      - Latencia promedio: {result['avg_latency']:.2f}s")
            print(f"      - Rango de latencia: {result['min_latency']:.2f}s - {result['max_latency']:.2f}s")
            
            if result['tokens_per_second']:
                avg_tps = sum(result['tokens_per_second']) / len(result['tokens_per_second'])
                print(f"      - Velocidad promedio: {avg_tps:.2f} tokens/seg")
        
        # Guardar resultados
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"latency_comparison_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Resultados guardados en: {filename}")
    else:
        print("❌ No se pudieron obtener resultados de rendimiento")
    
    print("\n✅ Prueba de comparación de latencia completada")

if __name__ == "__main__":
    main()