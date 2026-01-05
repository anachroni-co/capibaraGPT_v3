#!/usr/bin/env python3
"""
Script para probar el router semántico y cómo enruta diferentes consultas
"""

import requests
import json
import time
from typing import List, Dict

# Configuración
BASE_URL = "http://localhost:8080"
ROUTER_ENABLED = True  # Asegúrate que el servidor esté configurado con routing

def test_model_with_explicit_selection(model_id: str, query: str):
    """Prueba un modelo específico con selección explícita"""
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": model_id,  # Selección explícita
                "messages": [{"role": "user", "content": query}],
                "temperature": 0.1,
                "max_tokens": 50
            },
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                tokens = result["usage"]["completion_tokens"] if "usage" in result else len(content.split())
                
                return {
                    "success": True,
                    "latency": elapsed_time,
                    "tokens": tokens,
                    "content": content,
                    "model_used": result.get("model", model_id)
                }
        return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_with_semantic_routing(query: str):
    """Prueba usando el router semántico (sin especificar modelo)"""
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
            timeout=60
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                tokens = result["usage"]["completion_tokens"] if "usage" in result else len(content.split())
                
                # Extraer información de routing si está disponible
                routed_to = result.get("model", "unknown")
                if "performance" in result:
                    routed_info = result["performance"].get("routed_to", routed_to)
                else:
                    routed_info = routed_to
                
                return {
                    "success": True,
                    "latency": elapsed_time,
                    "tokens": tokens,
                    "content": content,
                    "model_used": routed_info,
                    "routing_used": True
                }
        return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_router_accuracy():
    """Prueba cómo el router semántico enruta diferentes tipos de consultas"""
    
    test_cases = [
        {
            "query": "¿Cómo escribir una función en Python para calcular números primos?",
            "expected_domain": "coding",
            "type": "programming"
        },
        {
            "query": "Traduce 'Hello world' al español y francés",
            "expected_domain": "multilingual",
            "type": "translation"
        },
        {
            "query": "¿Qué es la inteligencia artificial?",
            "expected_domain": "general",
            "type": "general_knowledge"
        },
        {
            "query": "Explica cómo funciona un transformador en IA",
            "expected_domain": "expert",
            "type": "technical_explanation"
        }
    ]
    
    print("🔍 Probando el router semántico...")
    print(f"   Servidor: {BASE_URL}")
    print(f"   Consultas de prueba: {len(test_cases)}")
    print()
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"({i}/{len(test_cases)}) Prueba: {test_case['type']}")
        print(f"   Consulta: {test_case['query'][:60]}...")
        print(f"   Dominio esperado: {test_case['expected_domain']}")
        
        # Probar con routing semántico
        routing_result = test_with_semantic_routing(test_case['query'])
        
        if routing_result['success']:
            print(f"   ✅ Enrutado a: {routing_result['model_used']}")
            print(f"   ⏱️  Latencia: {routing_result['latency']:.3f}s")
            print(f"   📝 Tokens: {routing_result['tokens']}")
        else:
            print(f"   ❌ Error: {routing_result['error']}")
        
        print()
        results.append({
            "test_case": test_case,
            "result": routing_result
        })
    
    return results

def test_model_performance_comparison():
    """Compara el rendimiento de diferentes modelos con diferentes tipos de consultas"""
    
    queries_by_type = {
        "coding": [
            "Escribe una función en Python que calcule la serie de Fibonacci",
            "Cómo implementar un algoritmo de ordenamiento rápido"
        ],
        "general": [
            "¿Qué es la inteligencia artificial?",
            "Explica qué es el cambio climático"
        ],
        "multilingual": [
            "Traduce 'Hello, how are you?' al español y francés",
            "Cómo se dice 'good morning' en alemán e italiano"
        ]
    }
    
    print("🔍 Comparando rendimiento de modelos por tipo de consulta...")
    print()
    
    for query_type, queries in queries_by_type.items():
        print(f"📊 Tipo de consulta: {query_type}")
        print("-" * 40)
        
        for query in queries:
            print(f"   Consulta: {query[:50]}...")
            
            # Obtener respuesta con routing semántico
            routing_result = test_with_semantic_routing(query)
            if routing_result['success']:
                print(f"     Routing → {routing_result['model_used']}: {routing_result['latency']:.3f}s")
            else:
                print(f"     Routing → Error: {routing_result['error']}")
        
        print()

def main():
    print("🚀 Iniciando pruebas del router semántico y modelos...")
    print()
    
    # Probar el router semántico
    router_results = test_router_accuracy()
    
    # Comparar rendimiento
    test_model_performance_comparison()
    
    print("="*60)
    print("✅ Pruebas de router semántico completadas!")
    
    # Contar resultados exitosos
    successful_routes = sum(1 for r in router_results if r['result']['success'])
    print(f"   Rutas exitosas: {successful_routes}/{len(router_results)}")
    print("="*60)

if __name__ == "__main__":
    main()