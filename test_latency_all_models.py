#!/usr/bin/env python3
"""
Script para probar la latencia de todos los modelos en el sistema multimodelo ARM-Axion
"""

import time
import requests
import statistics
from typing import List, Dict, Tuple
import json

# Configuración
BASE_URL = "http://localhost:8080"
TEST_QUERIES = {
    "general": "¿Qué es la inteligencia artificial?",
    "technical": "Explica qué es un algoritmo de búsqueda binaria.",
    "coding": "Escribe una función en Python que calcule la serie de Fibonacci.",
    "multilingual": "Traduce 'Hello world' al español y francés.",
    "complex": "¿Cuáles son las ventajas del procesamiento en paralelo?"
}

def get_available_experts() -> List[str]:
    """Obtiene la lista de expertos disponibles"""
    try:
        response = requests.get(f"{BASE_URL}/experts", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [expert["expert_id"] for expert in data.get("experts", [])]
        else:
            print(f"❌ Error al obtener expertos: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return []

def test_model_latency(model_id: str, query: str, iterations: int = 3) -> Dict:
    """Prueba la latencia de un modelo específico"""
    latencies = []
    tokens_per_second_values = []
    success_count = 0
    
    print(f"   Probando {model_id} con '{query[:30]}...' ({iterations} iteraciones)")
    
    for i in range(iterations):
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": query}],
                    "temperature": 0.1,
                    "max_tokens": 50
                },
                headers={"Content-Type": "application/json"},
                timeout=60  # Aumentar timeout para permitir carga de modelo
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    latencies.append(total_time)
                    
                    # Calcular tokens por segundo
                    if "usage" in result:
                        completion_tokens = result["usage"].get("completion_tokens", 0)
                    else:
                        content = result["choices"][0]["message"]["content"]
                        completion_tokens = len(content.split())
                    
                    if total_time > 0:
                        tps = completion_tokens / total_time
                        tokens_per_second_values.append(tps)
                    
                    success_count += 1
                    print(f"     Iteración {i+1}: {total_time:.3f}s ({tps:.2f} t/s)")
                else:
                    print(f"     Iteración {i+1}: ❌ No choices in response")
            else:
                print(f"     Iteración {i+1}: ❌ HTTP {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            print(f"     Iteración {i+1}: ⏳ Timeout")
        except Exception as e:
            print(f"     Iteración {i+1}: ❌ Error: {e}")
    
    stats = {
        "model_id": model_id,
        "latencies": latencies,
        "tokens_per_second_values": tokens_per_second_values,
        "success_count": success_count,
        "total_attempts": iterations
    }
    
    if latencies:
        stats["avg_latency"] = statistics.mean(latencies) if latencies else 0
        stats["min_latency"] = min(latencies) if latencies else 0
        stats["max_latency"] = max(latencies) if latencies else 0
        stats["std_dev_latency"] = statistics.stdev(latencies) if len(latencies) > 1 else 0
    else:
        stats["avg_latency"] = 0
        stats["min_latency"] = 0
        stats["max_latency"] = 0
        stats["std_dev_latency"] = 0
    
    if tokens_per_second_values:
        stats["avg_tokens_per_second"] = statistics.mean(tokens_per_second_values)
    else:
        stats["avg_tokens_per_second"] = 0
    
    return stats

def run_latency_tests():
    """Ejecuta pruebas de latencia para todos los modelos"""
    print("🔍 Obteniendo lista de expertos disponibles...")
    experts = get_available_experts()
    
    if not experts:
        print("❌ No se pudieron obtener expertos. Asegúrate que el servidor esté corriendo.")
        return
    
    print(f"✅ {len(experts)} expertos encontrados:")
    for expert in experts:
        print(f"   - {expert}")
    print()
    
    # Test queries por dominio
    domain_queries = {
        "phi4_fast": TEST_QUERIES["general"],
        "mistral_balanced": TEST_QUERIES["technical"], 
        "qwen_coder": TEST_QUERIES["coding"],
        "gemma3_multimodal": TEST_QUERIES["complex"],
        "aya_expanse_multilingual": TEST_QUERIES["multilingual"]
    }
    
    all_results = []
    
    print("⏱️  Iniciando pruebas de latencia...")
    print("="*80)
    
    for expert in experts:
        print(f"\n🤖 Prueba de latencia para: {expert}")
        print("-" * 50)
        
        # Usar query específica si está definida, de lo contrario usar una general
        query = domain_queries.get(expert, TEST_QUERIES["general"])
        
        # Ejecutar pruebas de latencia
        result = test_model_latency(expert, query, iterations=3)
        all_results.append(result)
        
        print()
    
    # Mostrar resultados resumidos
    print("="*80)
    print("📊 RESULTADOS RESUMIDOS DE LATENCIA")
    print("="*80)
    
    print(f"{'Modelo':<25} {'Éxito':<6} {'Promedio':<10} {'Mínimo':<10} {'Máximo':<10} {'StdDev':<10} {'Tokens/s':<10}")
    print("-" * 80)
    
    for result in all_results:
        model = result["model_id"]
        success_rate = f"{result['success_count']}/{result['total_attempts']}"
        avg_lat = f"{result['avg_latency']:.3f}s" if result['avg_latency'] > 0 else "N/A"
        min_lat = f"{result['min_latency']:.3f}s" if result['min_latency'] > 0 else "N/A"
        max_lat = f"{result['max_latency']:.3f}s" if result['max_latency'] > 0 else "N/A"
        std_dev = f"{result['std_dev_latency']:.3f}s" if result['std_dev_latency'] > 0 else "N/A"
        tps = f"{result['avg_tokens_per_second']:.2f}" if result['avg_tokens_per_second'] > 0 else "N/A"
        
        print(f"{model:<25} {success_rate:<6} {avg_lat:<10} {min_lat:<10} {max_lat:<10} {std_dev:<10} {tps:<10}")
    
    print()
    print("="*80)
    print("📈 ANÁLISIS DETALLADO POR MODELO")
    print("="*80)
    
    for result in all_results:
        print(f"\n🤖 Modelo: {result['model_id']}")
        print(f"   Tasa de éxito: {result['success_count']}/{result['total_attempts']} ({result['success_count']/result['total_attempts']*100:.1f}%)")
        
        if result['latencies']:
            print(f"   Latencia - Promedio: {result['avg_latency']:.3f}s, Mínimo: {result['min_latency']:.3f}s, Máximo: {result['max_latency']:.3f}s")
            if result['std_dev_latency'] > 0:
                print(f"   Desviación estándar: {result['std_dev_latency']:.3f}s")
        
        if result['tokens_per_second_values']:
            print(f"   Velocidad promedio: {result['avg_tokens_per_second']:.2f} tokens/seg")
    
    return all_results

def main():
    print("🚀 Iniciando pruebas de latencia para todos los modelos...")
    print(f"   Servidor: {BASE_URL}")
    print()
    
    results = run_latency_tests()
    
    print("\n✅ Pruebas de latencia completadas!")
    
    # Guardar resultados en archivo
    try:
        with open("/home/elect/capibara6/latency_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("📄 Resultados guardados en latency_test_results.json")
    except Exception as e:
        print(f"⚠️  No se pudieron guardar los resultados: {e}")

if __name__ == "__main__":
    main()