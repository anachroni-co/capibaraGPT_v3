#!/usr/bin/env python3
"""
Test rápido y seguro de latencia para el sistema ARM-Axion con monitoreo de RAM
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

def check_ram_usage(threshold: float = 90.0):
    """Verifica si el uso de RAM excede el umbral"""
    ram_percent = get_ram_usage_percent()
    if ram_percent > threshold:
        print(f"⚠️  RAM uso: {ram_percent:.1f}% - SUPERIOR AL LÍMITE DE {threshold}%")
        return True
    else:
        print(f"📊 RAM uso: {ram_percent:.1f}% - Seguro para continuar")
        return False

def is_server_responding(url: str) -> bool:
    """Verifica si el servidor está respondiendo"""
    try:
        response = requests.get(f"{url}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def quick_test_latency_safe(server_url: str = "http://localhost:8082"):
    """
    Test rápido de latencia con seguridad RAM
    """
    print("🚀 Iniciando TEST RÁPIDO de latencia con monitoreo de RAM...")
    print(f"   Servidor: {server_url}")
    print(f"   3 solicitudes como prueba rápida")
    print("="*60)

    if not is_server_responding(server_url):
        print(f"❌ Servidor no responde en {server_url}")
        return

    test_prompts = [
        "¿Qué es la inteligencia artificial?",
        "Explica qué hace un compilador", 
        "¿Cómo funciona la memoria en un sistema informático?"
    ]

    latencies = []
    tokens_per_second_values = []
    successful_requests = 0

    for i, prompt in enumerate(test_prompts):
        print(f"\\n--- Prueba {i+1}/3 ---")
        
        # Verificar RAM antes de la solicitud
        if check_ram_usage(90.0):
            print("⚠️  Prueba detenida por uso elevado de RAM")
            break

        print(f"📝 Prompt: '{prompt}'")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                json={
                    "model": "aya_expanse_multilingual",  # Modelo disponible
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 30,
                    "temperature": 0.7
                },
                timeout=60
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
                
                print(f"✅ Éxito: {total_time:.2f}s ({tokens_per_second:.2f} tok/s)")
                print(f"   Tokens: {tokens_generated}")
            else:
                print(f"❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

        # Pequeño delay para no sobrecargar
        time.sleep(2)

    # Resultados
    print("\\n" + "="*60)
    print("📊 RESULTADOS RÁPIDOS:")
    if latencies:
        print(f"✅ Solicitudes exitosas: {successful_requests}/3")
        print(f"   Promedio latencia: {statistics.mean(latencies):.2f}s")
        print(f"   Rango: {min(latencies):.2f}s - {max(latencies):.2f}s")
        if tokens_per_second_values:
            print(f"   Promedio vel.: {statistics.mean(tokens_per_second_values):.2f} tok/s")
    else:
        print("❌ No se completaron solicitudes exitosas")
    
    print(f"\\n💾 RAM final: {get_ram_usage_percent():.1f}%")
    print("✅ Test rápido completado")


if __name__ == "__main__":
    # Probar el servidor actualmente disponible
    server_url = "http://localhost:8082"  # Servidor estándar
    print(f"🔍 Probando servidor: {server_url}")
    quick_test_latency_safe(server_url)