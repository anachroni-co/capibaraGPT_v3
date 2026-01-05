#!/usr/bin/env python3
"""
Prueba de latencia individual de cada modelo experto en el sistema ARM-Axion
"""

import requests
import time
import json
import psutil
from typing import Dict, List
import statistics

def get_ram_usage_percent():
    """Obtiene el porcentaje de uso de RAM"""
    return psutil.virtual_memory().percent

def test_individual_model_latency(model_id: str, server_url: str = "http://localhost:8082", iterations: int = 3):
    """
    Prueba la latencia de un modelo individual
    """
    print(f"🧪 Probando latencia de modelo: {model_id}")
    print("-" * 50)
    
    test_prompts = [
        "¿Qué es la inteligencia artificial?",
        "Explica qué es la atención Flash en transformers.",
        "Escribe una función en Python que calcule fibonacci."
    ]
    
    latencies = []
    tokens_per_second_values = []
    successful_requests = 0
    
    for i in range(min(iterations, len(test_prompts))):
        prompt = test_prompts[i]
        print(f"  Solicitud {i+1}/{min(iterations, len(test_prompts))}: '{prompt[:30]}...'")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 20,  # Limitar para prueba rápida
                    "temperature": 0.7
                },
                timeout=120
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                usage = result.get('usage', {})
                tokens_generated = usage.get('completion_tokens', 0)
                tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
                
                latencies.append(total_time)
                tokens_per_second_values.append(tokens_per_second)
                successful_requests += 1
                
                print(f"    ✅ {total_time:.2f}s ({tokens_per_second:.2f} tok/s)")
            else:
                print(f"    ❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        # Pequeño delay entre solicitudes para no sobrecargar
        time.sleep(2)
    
    results = {}
    if latencies:
        results = {
            "model_id": model_id,
            "latencies": latencies,
            "tokens_per_second_values": tokens_per_second_values,
            "successful_requests": successful_requests,
            "total_attempts": min(iterations, len(test_prompts)),
            "avg_latency": statistics.mean(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "std_dev_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "avg_tokens_per_second": statistics.mean(tokens_per_second_values) if tokens_per_second_values else 0
        }
    
    return results

def run_individual_model_latency_tests():
    """
    Ejecuta pruebas de latencia para cada modelo individual
    """
    print("🚀 INICIANDO PRUEBA DE LATENCIA INDIVIDUAL POR MODELO")
    print("="*80)
    
    initial_ram = get_ram_usage_percent()
    print(f"📊 RAM inicial: {initial_ram:.1f}%")
    print("")
    
    # Modelos disponibles
    models = [
        "phi4_fast", 
        "mistral_balanced", 
        "qwen_coder", 
        "gemma3_multimodal", 
        "aya_expanse_multilingual"
    ]
    
    all_results = []
    
    for model in models:
        print(f"\\n🔄 MODELO: {model}")
        print("="*80)
        
        # Verificar RAM antes de cada modelo
        current_ram = get_ram_usage_percent()
        print(f"📊 RAM antes de probar {model}: {current_ram:.1f}%")
        
        if current_ram > 95.0:
            print(f"⚠️  RAM críticamente alta, saltando modelo {model}")
            continue
        
        result = test_individual_model_latency(model)
        if result:
            all_results.append(result)
            print(f"\\n✅ RESULTADOS {model}:")
            print(f"   Promedio latencia: {result['avg_latency']:.2f}s")
            print(f"   Rango: {result['min_latency']:.2f}s - {result['max_latency']:.2f}s")
            print(f"   Estabilidad (std dev): {result['std_dev_latency']:.2f}s")
            print(f"   Velocidad promedio: {result['avg_tokens_per_second']:.2f} tok/s")
            print(f"   Éxito: {result['successful_requests']}/{result['total_attempts']}")
        else:
            print(f"❌ No se pudieron obtener resultados para {model}")
    
    final_ram = get_ram_usage_percent()
    print(f"\\n📊 RAM final: {final_ram:.1f}%")
    
    # Resumen final
    print("\\n" + "="*80)
    print("📋 RESUMEN DE PRUEBAS DE LATENCIA INDIVIDUAL")
    print("="*80)
    
    if all_results:
        print(f"Modelos probados: {len(all_results)}/{len(models)}")
        print(f"Total solicitudes exitosas: {sum(r['successful_requests'] for r in all_results)}")
        
        print("\\n📊 DETALLE POR MODELO:")
        print(f"{'Modelo':<25} {'Prom. Latencia':<15} {'Min-Max (s)':<15} {'Tok/s':<10} {'Éxito':<8}")
        print("-" * 80)
        
        for result in all_results:
            avg_lat = result['avg_latency']
            min_max = f"{result['min_latency']:.2f}-{result['max_latency']:.2f}"
            tps = result['avg_tokens_per_second']
            success_rate = f"{result['successful_requests']}/{result['total_attempts']}"
            print(f"{result['model_id']:<25} {avg_lat:<15.2f} {min_max:<15} {tps:<10.2f} {success_rate:<8}")
    else:
        print("❌ No se obtuvieron resultados de ningún modelo")
    
    return all_results

def main():
    print("🦫 Prueba de Latencia Individual - Sistema ARM-Axion")
    print("   Evaluando cada modelo experto de forma independiente")
    print("="*80)
    
    results = run_individual_model_latency_tests()
    
    # Guardar resultados en JSON
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"individual_model_latency_results_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\\n💾 Resultados guardados en: {filename}")
    
    print("\\n✅ Prueba de latencia individual completada")


if __name__ == "__main__":
    main()