#!/usr/bin/env python3
"""
Script para probar la conexión y latencia de cada modelo individualmente
en el sistema ARM-Axion multimodelo
"""

import time
import json
import requests
import asyncio
import aiohttp
from typing import Dict, List, Tuple
from dataclasses import dataclass
import statistics

# Configuración del servidor
BASE_URL = "http://localhost:8080"  # Ajusta según tu configuración
TEST_QUERY = "¿Cuál es el modelo de lenguaje más avanzado actualmente?"
TEST_QUERY_CODE = "Escribe una función en Python que calcule la serie de Fibonacci"
TEST_QUERY_MULTILINGUAL = "Traduce 'Hello, how are you?' al español y francés"

@dataclass
class ModelTestResult:
    model_id: str
    connection_success: bool
    latency: float
    tokens_per_second: float
    response_length: int
    error_message: str = ""


def test_model_connection(model_id: str, query: str = TEST_QUERY) -> ModelTestResult:
    """
    Prueba la conexión y latencia de un modelo específico
    """
    start_time = time.time()
    
    try:
        url = f"{BASE_URL}/v1/chat/completions"
        
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": query}
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60  # 60 segundos de timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            total_time = time.time() - start_time
            
            # Extraer información de la respuesta
            if "choices" in result and len(result["choices"]) > 0:
                response_text = result["choices"][0]["message"]["content"]
                response_length = len(response_text.split())
                
                # Calcular tokens por segundo
                if "usage" in result:
                    completion_tokens = result["usage"].get("completion_tokens", len(response_text.split()))
                else:
                    completion_tokens = len(response_text.split())
                
                tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
                
                return ModelTestResult(
                    model_id=model_id,
                    connection_success=True,
                    latency=total_time,
                    tokens_per_second=tokens_per_second,
                    response_length=response_length
                )
            else:
                return ModelTestResult(
                    model_id=model_id,
                    connection_success=False,
                    latency=0,
                    tokens_per_second=0,
                    response_length=0,
                    error_message="No choices in response"
                )
        else:
            return ModelTestResult(
                model_id=model_id,
                connection_success=False,
                latency=0,
                tokens_per_second=0,
                response_length=0,
                error_message=f"HTTP {response.status_code}: {response.text}"
            )
    
    except requests.exceptions.Timeout:
        return ModelTestResult(
            model_id=model_id,
            connection_success=False,
            latency=0,
            tokens_per_second=0,
            response_length=0,
            error_message="Timeout"
        )
    except requests.exceptions.ConnectionError:
        return ModelTestResult(
            model_id=model_id,
            connection_success=False,
            latency=0,
            tokens_per_second=0,
            response_length=0,
            error_message="Connection Error"
        )
    except Exception as e:
        return ModelTestResult(
            model_id=model_id,
            connection_success=False,
            latency=0,
            tokens_per_second=0,
            response_length=0,
            error_message=str(e)
        )


async def test_model_connection_async(session: aiohttp.ClientSession, model_id: str, query: str = TEST_QUERY) -> ModelTestResult:
    """
    Prueba la conexión y latencia de un modelo específico usando async
    """
    start_time = time.time()
    
    try:
        url = f"{BASE_URL}/v1/chat/completions"
        
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": query}
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        timeout = aiohttp.ClientTimeout(total=60)  # 60 segundos de timeout
        async with session.post(url, json=payload, timeout=timeout) as response:
            if response.status == 200:
                result = await response.json()
                total_time = time.time() - start_time
                
                # Extraer información de la respuesta
                if "choices" in result and len(result["choices"]) > 0:
                    response_text = result["choices"][0]["message"]["content"]
                    response_length = len(response_text.split())
                    
                    # Calcular tokens por segundo
                    if "usage" in result:
                        completion_tokens = result["usage"].get("completion_tokens", len(response_text.split()))
                    else:
                        completion_tokens = len(response_text.split())
                    
                    tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
                    
                    return ModelTestResult(
                        model_id=model_id,
                        connection_success=True,
                        latency=total_time,
                        tokens_per_second=tokens_per_second,
                        response_length=response_length
                    )
                else:
                    return ModelTestResult(
                        model_id=model_id,
                        connection_success=False,
                        latency=0,
                        tokens_per_second=0,
                        response_length=0,
                        error_message="No choices in response"
                    )
            else:
                error_text = await response.text()
                return ModelTestResult(
                    model_id=model_id,
                    connection_success=False,
                    latency=0,
                    tokens_per_second=0,
                    response_length=0,
                    error_message=f"HTTP {response.status}: {error_text}"
                )
    
    except asyncio.TimeoutError:
        return ModelTestResult(
            model_id=model_id,
            connection_success=False,
            latency=0,
            tokens_per_second=0,
            response_length=0,
            error_message="Timeout"
        )
    except aiohttp.ClientConnectorError:
        return ModelTestResult(
            model_id=model_id,
            connection_success=False,
            latency=0,
            tokens_per_second=0,
            response_length=0,
            error_message="Connection Error"
        )
    except Exception as e:
        return ModelTestResult(
            model_id=model_id,
            connection_success=False,
            latency=0,
            tokens_per_second=0,
            response_length=0,
            error_message=str(e)
        )


def get_available_models() -> List[str]:
    """
    Obtiene la lista de modelos disponibles desde el servidor
    """
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        else:
            print(f"Error al obtener modelos: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error al conectar con el servidor: {e}")
        return []


def run_single_model_tests(model_ids: List[str], test_queries: Dict[str, str]) -> List[ModelTestResult]:
    """
    Ejecuta pruebas de latencia para cada modelo individualmente
    """
    print(f"🔍 Probando {len(model_ids)} modelos...")
    print("="*80)
    
    all_results: List[ModelTestResult] = []
    
    for i, model_id in enumerate(model_ids):
        print(f"\n({i+1}/{len(model_ids)}) Probando modelo: {model_id}")
        print("-" * 50)
        
        for query_name, query in test_queries.items():
            print(f"   Prueba: {query_name}")
            
            # Ejecutar la prueba
            result = test_model_connection(model_id, query)
            
            if result.connection_success:
                print(f"   ✅ Conexión exitosa")
                print(f"   ⏱️  Latencia: {result.latency:.3f}s")
                print(f"   🚀 Tokens/seg: {result.tokens_per_second:.2f}")
                print(f"   📝 Tokens: {result.response_length}")
            else:
                print(f"   ❌ Error: {result.error_message}")
            
            all_results.append(result)
    
    return all_results


async def run_async_model_tests(model_ids: List[str], test_queries: Dict[str, str]) -> List[ModelTestResult]:
    """
    Ejecuta pruebas de latencia para cada modelo individualmente usando async
    """
    print(f"🔍 Probando {len(model_ids)} modelos (async)...")
    print("="*80)
    
    all_results: List[ModelTestResult] = []
    
    async with aiohttp.ClientSession() as session:
        for i, model_id in enumerate(model_ids):
            print(f"\n({i+1}/{len(model_ids)}) Probando modelo: {model_id}")
            print("-" * 50)
            
            for query_name, query in test_queries.items():
                print(f"   Prueba: {query_name}")
                
                # Ejecutar la prueba
                result = await test_model_connection_async(session, model_id, query)
                
                if result.connection_success:
                    print(f"   ✅ Conexión exitosa")
                    print(f"   ⏱️  Latencia: {result.latency:.3f}s")
                    print(f"   🚀 Tokens/seg: {result.tokens_per_second:.2f}")
                    print(f"   📝 Tokens: {result.response_length}")
                else:
                    print(f"   ❌ Error: {result.error_message}")
                
                all_results.append(result)
    
    return all_results


def analyze_results(results: List[ModelTestResult]):
    """
    Analiza y muestra estadísticas de las pruebas
    """
    print("\n" + "="*80)
    print("📊 RESULTADOS FINALES")
    print("="*80)
    
    # Separar resultados por modelo
    model_results = {}
    for result in results:
        if result.model_id not in model_results:
            model_results[result.model_id] = []
        model_results[result.model_id].append(result)
    
    # Mostrar estadísticas por modelo
    for model_id, model_results_list in model_results.items():
        successful_tests = [r for r in model_results_list if r.connection_success]
        
        print(f"\n🤖 Modelo: {model_id}")
        print(f"   Conexiones exitosas: {len(successful_tests)}/{len(model_results_list)}")
        
        if successful_tests:
            latencies = [r.latency for r in successful_tests]
            tokens_per_sec = [r.tokens_per_second for r in successful_tests if r.tokens_per_second > 0]
            response_lengths = [r.response_length for r in successful_tests]
            
            print(f"   Latencia - Min: {min(latencies):.3f}s, Max: {max(latencies):.3f}s, Promedio: {statistics.mean(latencies):.3f}s")
            if tokens_per_sec:
                print(f"   Tokens/seg - Min: {min(tokens_per_sec):.2f}, Max: {max(tokens_per_sec):.2f}, Promedio: {statistics.mean(tokens_per_sec):.2f}")
            print(f"   Tokens resp - Min: {min(response_lengths)}, Max: {max(response_lengths)}, Promedio: {statistics.mean(response_lengths):.0f}")
        else:
            failed_tests = [r for r in model_results_list if not r.connection_success]
            if failed_tests:
                print(f"   Errores: {[f.error_message for f in failed_tests]}")
    
    # Estadísticas generales
    all_successful = [r for r in results if r.connection_success]
    all_latencies = [r.latency for r in all_successful]
    
    print(f"\n📈 Estadísticas Generales:")
    print(f"   Total de pruebas: {len(results)}")
    print(f"   Pruebas exitosas: {len(all_successful)}")
    print(f"   Tasa de éxito: {(len(all_successful)/len(results)*100):.1f}%")
    
    if all_latencies:
        print(f"   Latencia general - Min: {min(all_latencies):.3f}s, Max: {max(all_latencies):.3f}s, Promedio: {statistics.mean(all_latencies):.3f}s")


async def main():
    """
    Función principal para ejecutar las pruebas
    """
    print("🚀 Iniciando pruebas de conexión y latencia de modelos...")
    print(f"   Servidor: {BASE_URL}")
    
    # Obtener modelos disponibles
    available_models = get_available_models()
    
    if not available_models:
        print("❌ No se pudieron obtener los modelos disponibles")
        print("   Asegúrate de que el servidor esté corriendo en", BASE_URL)
        return
    
    print(f"   Modelos disponibles: {available_models}")
    
    # Definir queries de prueba
    test_queries = {
        "general": TEST_QUERY,
        "code": TEST_QUERY_CODE,
        "multilingual": TEST_QUERY_MULTILINGUAL
    }
    
    # Ejecutar pruebas
    results = await run_async_model_tests(available_models, test_queries)
    
    # Analizar resultados
    analyze_results(results)
    
    print("\n✅ Pruebas completadas!")


if __name__ == "__main__":
    asyncio.run(main())