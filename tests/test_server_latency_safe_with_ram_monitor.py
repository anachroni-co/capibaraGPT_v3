#!/usr/bin/env python3
"""
Versión segura de la prueba de latencia para cualquier servidor ARM-Axion
Monitoreo de RAM para prevenir bloqueos del servidor
"""

import requests
import time
import json
import psutil
import os
from typing import Dict, List, Optional
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

def test_server_latency_safe(
    server_url: str = "http://localhost:8082",
    max_requests: int = 10,  # Reducido para ser más seguro
    delay_between_requests: float = 3.0,
    ram_threshold: float = 90.0
):
    """
    Prueba de latencia segura que monitorea RAM para evitar bloqueos
    """
    print("🚀 Iniciando prueba de latencia SEGURA...")
    print(f"   Servidor: {server_url}")
    print(f"   Máximo de solicitudes: {max_requests}")
    print(f"   Retraso entre solicitudes: {delay_between_requests}s")
    print(f"   Límite de RAM: {ram_threshold}%")
    print("="*60)

    if not is_server_responding(server_url):
        print(f"❌ Servidor no responde en {server_url}")
        return

    # Test prompts para diferentes dominios
    test_prompts = [
        "¿Cómo funciona la atención Flash en ARM Axion?",
        "Explica brevemente el algoritmo de quicksort",
        "Escribe una función en Python que calcule números primos",
        "¿Cuál es la diferencia entre CPU e GPU para el procesamiento?",
        "Describe el concepto de optimización NEON en ARM"
    ]

    latencies = []
    tokens_per_second_values = []
    successful_requests = 0
    failed_requests = 0

    for i in range(max_requests):
        print(f"\\n--- Solicitud {i+1}/{max_requests} ---")
        
        # Verificar uso de RAM antes de cada solicitud
        if check_ram_usage(ram_threshold):
            print(f"⚠️  Prueba detenida temprano por uso elevado de RAM")
            break
            
        # Verificar si el servidor sigue respondiendo
        if not is_server_responding(server_url):
            print(f"❌ Servidor dejó de responder en solicitud {i+1}")
            failed_requests += 1
            break

        prompt = test_prompts[i % len(test_prompts)]
        print(f"📝 Prompt: '{prompt[:50]}...'")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 50,  # Reducido para ser más seguro
                    "temperature": 0.7,
                    "model": ""  # Dejar vacío para usar router automático (excepto en servidor básico)
                },
                timeout=120  # Tiempo de espera más largo
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
                
                # Obtener modelo usado si está disponible
                model_used = result.get('model', 'unknown')
                if model_used != 'unknown':
                    print(f"   Modelo usado: {model_used}")
                
            else:
                print(f"❌ HTTP {response.status_code}: {str(response.text)[:100]}")
                failed_requests += 1
                
        except requests.exceptions.Timeout:
            print("⏰ Timeout")
            failed_requests += 1
        except requests.exceptions.RequestException as e:
            print(f"💥 Error de red: {e}")
            failed_requests += 1
        except Exception as e:
            print(f"💥 Error: {e}")
            failed_requests += 1

        # Verificar RAM después de la solicitud
        if check_ram_usage(ram_threshold):
            print(f"⚠️  Prueba detenida por uso elevado de RAM después de solicitud {i+1}")
            break

        # Retraso entre solicitudes
        if i < max_requests - 1:  # No esperar después de la última solicitud
            print(f"⏳ Esperando {delay_between_requests}s antes de siguiente solicitud...")
            time.sleep(delay_between_requests)

    # Resultados finales
    print("\\n" + "="*60)
    print("📊 RESULTADOS FINALES DE PRUEBA DE LATENCIA SEGURA")
    print("="*60)
    
    if latencies:
        print(f"✅ Solicitudes exitosas: {successful_requests}")
        print(f"❌ Solicitudes fallidas: {failed_requests}")
        print(f"📊 Total de solicitudes intentadas: {successful_requests + failed_requests}")
        
        print(f"\\n⏱️  RENDIMIENTO:")
        if latencies:
            print(f"   Promedio latencia: {statistics.mean(latencies):.2f}s")
            print(f"   Mínimo latencia: {min(latencies):.2f}s")
            print(f"   Máximo latencia: {max(latencies):.2f}s")
            if len(latencies) > 1:
                print(f"   Desviación estándar: {statistics.stdev(latencies):.2f}s")
        
        if tokens_per_second_values:
            print(f"\\n⚡ VELOCIDAD:")
            print(f"   Promedio tokens/seg: {statistics.mean(tokens_per_second_values):.2f}")
            print(f"   Rango: {min(tokens_per_second_values):.2f} - {max(tokens_per_second_values):.2f}")
        
        print(f"\\n📈 EFICIENCIA:")
        total_attempts = successful_requests + failed_requests
        if total_attempts > 0:
            print(f"   Tasa de éxito: {(successful_requests/total_attempts*100):.1f}%")
    else:
        print("❌ No se completaron solicitudes exitosas")
    
    print(f"\\n💾 RAM final: {get_ram_usage_percent():.1f}%")
    print("✅ Prueba de latencia segura completada")


def main():
    """Función principal con selección de servidor"""
    print("🦫 Prueba Segura de Latencia - Sistema ARM-Axion")
    print("   Monitoreo de RAM para evitar bloqueos del servidor")
    print("   Límite: 90% de uso de RAM")
    print("="*70)
    
    # Lista de servidores disponibles
    servers = {
        "1": ("http://localhost:8082", "Servidor Estándar"),
        "2": ("http://localhost:8083", "Servidor con Streaming"), 
        "3": ("Seleccionar manualmente", "URL personalizada")
    }
    
    print("Selecciona un servidor para probar:")
    for key, (url, desc) in servers.items():
        print(f"  {key}. {desc} - {url}")
    
    choice = input("\\nIngresa tu elección (1-3): ").strip()
    
    if choice == "1":
        server_url = "http://localhost:8082"
    elif choice == "2":
        server_url = "http://localhost:8083"
    elif choice == "3":
        server_url = input("Ingresa la URL del servidor (ej. http://localhost:8082): ").strip()
        if not server_url:
            server_url = "http://localhost:8082"
    else:
        print("Opción inválida, usando servidor estándar...")
        server_url = "http://localhost:8082"
    
    print(f"\\n🔍 Verificando servidor en {server_url}...")
    
    if is_server_responding(server_url):
        print(f"✅ Servidor disponible: {servers.get(choice, ('', 'Servidor Personalizado'))[1]}")
        test_server_latency_safe(server_url)
    else:
        print(f"❌ Servidor no disponible en {server_url}")
        print("   Verifica que el servidor esté corriendo antes de ejecutar la prueba")


if __name__ == "__main__":
    main()