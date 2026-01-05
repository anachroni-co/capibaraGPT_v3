#!/usr/bin/env python3
"""
Script para diagnosticar problemas con los modelos del sistema ARM-Axion
"""

import time
import requests
import json
from typing import List, Dict
import sys
import traceback

BASE_URL = "http://localhost:8080"

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

def test_model_simple(model_id: str) -> Dict:
    """Prueba simple con timeout extendido"""
    try:
        start_time = time.time()
        
        # Hacer la solicitud con un timeout más largo
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.1,
                "max_tokens": 5
            },
            headers={"Content-Type": "application/json"},
            timeout=120  # 2 minutos de timeout
        )
        
        total_time = time.time() - start_time
        
        result = {
            "model_id": model_id,
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_time": total_time,
            "error_message": None
        }
        
        if response.status_code == 200:
            try:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    result["response_preview"] = data["choices"][0]["message"]["content"][:50]
                else:
                    result["response_preview"] = "No choices in response"
            except:
                result["response_preview"] = "Could not parse response"
        else:
            result["error_message"] = response.text[:200]  # Limitar longitud
            
        return result
    except requests.exceptions.Timeout:
        return {
            "model_id": model_id,
            "success": False,
            "status_code": "TIMEOUT",
            "response_time": 120,  # Max timeout
            "error_message": "Request timed out after 120 seconds"
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "success": False,
            "status_code": "EXCEPTION",
            "response_time": time.time() - start_time if 'start_time' in locals() else 0,
            "error_message": str(e)
        }

def get_server_stats():
    """Obtiene estadísticas del servidor"""
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"❌ Error obteniendo stats: {e}")
        return None

def main():
    print("🔍 Diagnosticando estado del sistema multimodelo ARM-Axion...")
    print(f"   Servidor: {BASE_URL}")
    print()
    
    # Verificar estado del servidor
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Servidor saludable: {health_data}")
        else:
            print(f"❌ Servidor reporta error: {health_response.status_code}")
            return
    except Exception as e:
        print(f"❌ No se puede conectar al servidor: {e}")
        return
    
    # Obtener expertos
    experts = get_available_experts()
    if not experts:
        print("❌ No se pudieron obtener expertos")
        return
    
    print(f"✅ {len(experts)} expertos disponibles:")
    for expert in experts:
        print(f"   - {expert}")
    print()
    
    # Probar cada modelo
    results = []
    for i, expert in enumerate(experts):
        print(f"({i+1}/{len(experts)}) Probando {expert}...")
        
        result = test_model_simple(expert)
        results.append(result)
        
        if result["success"]:
            print(f"   ✅ ÉXITO - Tiempo: {result['response_time']:.2f}s")
            print(f"      Respuesta: {result['response_preview']}")
        else:
            print(f"   ❌ FALLO - Código: {result['status_code']}, Tiempo: {result['response_time']:.2f}s")
            print(f"      Error: {result['error_message']}")
        print()
    
    # Mostrar resumen
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print("="*60)
    print("📊 RESUMEN DE DIAGNÓSTICO")
    print("="*60)
    print(f"Total modelos: {len(results)}")
    print(f"Exitosos: {len(successful)}")
    print(f"Fallidos: {len(failed)}")
    print()
    
    if successful:
        print("✅ Modelos funcionales:")
        for result in successful:
            print(f"   - {result['model_id']}: {result['response_time']:.2f}s")
    
    if failed:
        print("❌ Modelos no funcionales:")
        for result in failed:
            print(f"   - {result['model_id']}: {result['status_code']} - {result['error_message']}")
    
    # Obtener y mostrar estadísticas del servidor
    print("\n📈 ESTADÍSTICAS DEL SERVIDOR")
    print("-" * 40)
    stats = get_server_stats()
    if stats:
        print(f"Total requests: {stats.get('total_requests', 0)}")
        print(f"Número de expertos: {stats.get('router_stats', {}).get('num_experts', 0)}")
        experts_info = stats.get('experts', [])
        print("Estado de expertos:")
        for expert in experts_info:
            loaded = "SÍ" if expert.get('is_loaded', False) else "NO"
            print(f"   - {expert['expert_id']}: Cargado={loaded}, Requests={expert.get('total_requests', 0)}")
    
    return results

if __name__ == "__main__":
    results = main()