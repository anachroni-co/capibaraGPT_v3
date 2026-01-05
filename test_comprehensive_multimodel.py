#!/usr/bin/env python3
"""
Script completo para probar el sistema multimodelo ARM-Axion
con pruebas de conexión, latencia, routing y rendimiento
"""

import requests
import time
import json
from typing import List, Dict, Any
import statistics

# Configuración
BASE_URL = "http://localhost:8080"

def check_server_health():
    """Verifica la salud del servidor"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Servidor saludable")
            print(f"      Modelos cargados: {health_data.get('models_loaded', 'unknown')}")
            print(f"      Modelos disponibles: {health_data.get('models_available', 'unknown')}")
            return True
        else:
            print(f"   ❌ Servidor responded con código {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error de conexión: {e}")
        return False

def get_available_models():
    """Obtiene la lista de modelos disponibles"""
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return models
        else:
            print(f"   ❌ Error al obtener modelos: {response.status_code}")
            return []
    except Exception as e:
        print(f"   ❌ Error de conexión: {e}")
        return []

def test_model_latency(model_id: str, query: str, iterations: int = 3):
    """Prueba la latencia de un modelo específico"""
    latencies = []
    tokens_per_second_values = []
    
    print(f"   Probando {model_id} ({iterations} iteraciones)...")
    
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
                timeout=30
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
                    
                    if i == 0:  # Solo mostrar detalles en la primera iteración
                        print(f"     Iteración {i+1}: {total_time:.3f}s ({tps:.2f} t/s)")
                    else:
                        print(f"     Iteración {i+1}: {total_time:.3f}s")
                else:
                    print(f"     Iteración {i+1}: ❌ No choices in response")
            else:
                print(f"     Iteración {i+1}: ❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"     Iteración {i+1}: ❌ Error: {e}")
    
    if latencies:
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        avg_tps = statistics.mean(tokens_per_second_values) if tokens_per_second_values else 0
        
        return {
            "model_id": model_id,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "avg_tokens_per_second": avg_tps,
            "success_count": len(latencies),
            "total_attempts": iterations
        }
    
    return {
        "model_id": model_id,
        "avg_latency": 0,
        "min_latency": 0,
        "max_latency": 0,
        "avg_tokens_per_second": 0,
        "success_count": 0,
        "total_attempts": iterations
    }

def test_semantic_routing(query: str):
    """Prueba el routing semántico sin especificar modelo"""
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": query}],  # Sin modelo específico
                "temperature": 0.1,
                "max_tokens": 50
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                
                # Extraer el modelo usado si está disponible en la respuesta
                model_used = result.get("model", "unknown")
                
                return {
                    "success": True,
                    "model_used": model_used,
                    "latency": total_time,
                    "response": content[:100] + "..." if len(content) > 100 else content
                }
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_comprehensive_tests():
    """Ejecuta todas las pruebas comprehensivas"""
    print("="*70)
    print("🚀 INICIANDO PRUEBAS COMPRENSIVAS DEL SISTEMA MULTIMODELO ARM-AXION")
    print("="*70)
    
    # 1. Verificar salud del servidor
    print("\n🔍 1. VERIFICANDO SALUD DEL SERVIDOR")
    print("-" * 40)
    server_healthy = check_server_health()
    
    if not server_healthy:
        print("❌ El servidor no está respondiendo. Asegúrate que esté corriendo.")
        return
    
    # 2. Obtener modelos disponibles
    print("\n📚 2. OBTENIENDO MODELOS DISPONIBLES")
    print("-" * 40)
    models = get_available_models()
    
    if not models:
        print("❌ No se encontraron modelos disponibles.")
        return
    
    print(f"✅ {len(models)} modelos encontrados:")
    for model in models:
        print(f"   - {model}")
    
    # 3. Probar latencia de cada modelo
    print("\n⏱️  3. PROBANDO LATENCIA DE MODELOS")
    print("-" * 40)
    
    test_queries = [
        ("general", "¿Qué es la inteligencia artificial?"),
        ("technical", "Explica brevemente qué es un algoritmo."),
        ("coding", "Escribe un hola mundo en Python.")
    ]
    
    all_latency_results = []
    
    for model_id in models:
        print(f"\n🤖 Modelo: {model_id}")
        print("   " + "-" * 30)
        
        for query_type, query in test_queries:
            print(f"   Tipo: {query_type}")
            result = test_model_latency(model_id, query, iterations=3)
            all_latency_results.append(result)
    
    # 4. Probar routing semántico
    print("\n🎯 4. PROBANDO ROUTING SEMÁNTICO")
    print("-" * 40)
    
    routing_tests = [
        ("Pregunta general", "¿Qué es el machine learning?"),
        ("Consulta técnica", "¿Cómo funciona un transformer en IA?"),
        ("Solicitud de código", "Escribe una función en Python para ordenar una lista"),
        ("Consulta multilingüe", "Traduce 'Hello' al español")
    ]
    
    routing_results = []
    for test_name, query in routing_tests:
        print(f"\n   Prueba: {test_name}")
        print(f"   Consulta: {query}")
        
        result = test_semantic_routing(query)
        if result["success"]:
            print(f"   ✅ Enrutado a: {result['model_used']}")
            print(f"   ⏱️  Latencia: {result['latency']:.3f}s")
            print(f"   📝 Respuesta: {result['response']}")
        else:
            print(f"   ❌ Error: {result['error']}")
        
        routing_results.append({
            "test_name": test_name,
            "query": query,
            "result": result
        })
    
    # 5. Mostrar resumen
    print("\n📊 5. RESUMEN DE RESULTADOS")
    print("-" * 40)
    
    # Resumen de latencias
    print("   Latencias por modelo:")
    model_summary = {}
    for result in all_latency_results:
        model_id = result["model_id"]
        if model_id not in model_summary:
            model_summary[model_id] = {
                "latencies": [],
                "tokens_per_sec": [],
                "success_count": 0,
                "total_attempts": 0
            }
        
        model_summary[model_id]["latencies"].append(result["avg_latency"])
        model_summary[model_id]["tokens_per_sec"].append(result["avg_tokens_per_second"])
        model_summary[model_id]["success_count"] += result["success_count"]
        model_summary[model_id]["total_attempts"] += result["total_attempts"]
    
    for model_id, data in model_summary.items():
        if data["latencies"]:
            avg_lat = statistics.mean([l for l in data["latencies"] if l > 0])
            avg_tps = statistics.mean([tps for tps in data["tokens_per_sec"] if tps > 0])
            success_rate = (data["success_count"] / data["total_attempts"]) * 100 if data["total_attempts"] > 0 else 0
            
            print(f"      {model_id}:")
            print(f"         Latencia promedio: {avg_lat:.3f}s")
            print(f"         Tokens/seg promedio: {avg_tps:.2f}")
            print(f"         Tasa de éxito: {success_rate:.1f}%")
    
    # Resumen de routing
    print(f"\n   Routing semántico:")
    successful_routes = sum(1 for r in routing_results if r["result"]["success"])
    print(f"      Rutas exitosas: {successful_routes}/{len(routing_results)}")
    
    # Identificar modelos usados en routing
    models_used_in_routing = {}
    for result in routing_results:
        if result["result"]["success"]:
            model_used = result["result"]["model_used"]
            if model_used not in models_used_in_routing:
                models_used_in_routing[model_used] = 0
            models_used_in_routing[model_used] += 1
    
    print("      Distribución de modelos en routing:")
    for model, count in models_used_in_routing.items():
        print(f"         {model}: {count} veces")
    
    print("\n" + "="*70)
    print("✅ PRUEBAS COMPRENSIVAS COMPLETADAS")
    print("="*70)

if __name__ == "__main__":
    run_comprehensive_tests()