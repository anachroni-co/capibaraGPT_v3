#!/usr/bin/env python3
"""
Prueba segura de latencia para el sistema de consenso ARM-Axion
Con monitoreo de RAM para evitar bloqueos del servidor
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

def check_ram_usage():
    """Verifica si el uso de RAM excede el 90%"""
    ram_percent = get_ram_usage_percent()
    if ram_percent > 90.0:
        print(f"⚠️  RAM uso: {ram_percent:.1f}% - SUPERIOR AL LÍMITE DE 90%")
        print("   Terminando prueba para evitar bloqueo del servidor")
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

def test_consensus_latency(
    server_url: str = "http://localhost:8084",
    max_requests: int = 20,
    delay_between_requests: float = 2.0
):
    """
    Prueba de latencia segura para el sistema de consenso
    """
    print("🚀 Iniciando prueba de latencia para sistema de consenso...")
    print(f"   Servidor: {server_url}")
    print(f"   Máximo de solicitudes: {max_requests}")
    print(f"   Retraso entre solicitudes: {delay_between_requests}s")
    print("   Monitoreando uso de RAM para evitar bloqueos")
    print("="*60)

    if not is_server_responding(server_url):
        print(f"❌ Servidor no responde en {server_url}")
        return

    # Test prompts para diferentes dominios
    test_prompts = [
        "Explica cómo se implementan las optimizaciones ARM-Axion para mejorar el rendimiento.",
        "¿Cuál es la diferencia entre atención estándar y Flash Attention en ARM?",
        "Cómo se optimiza la memoria en un sistema con 5 modelos expertos.",
        "Cómo funciona el sistema de consenso entre diferentes modelos expertos.",
        "Qué ventajas tienen los kernels NEON para la inferencia en ARM Axion."
    ]

    latencies = []
    tokens_per_second_values = []
    successful_requests = 0
    failed_requests = 0

    for i in range(max_requests):
        print(f"\\n--- Solicitud {i+1}/{max_requests} ---")
        
        # Verificar uso de RAM antes de cada solicitud
        if check_ram_usage():
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
                    "max_tokens": 100,
                    "temperature": 0.7
                },
                timeout=60  # Tiempo de espera más largo para consenso
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                tokens_generated = result.get('usage', {}).get('completion_tokens', 0)
                tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
                
                latencies.append(total_time)
                tokens_per_second_values.append(tokens_per_second)
                successful_requests += 1
                
                print(f"✅ Éxito: {total_time:.2f}s ({tokens_per_second:.2f} tok/s)")
                
                # Mostrar modelo que respondió
                model = result.get('model', 'unknown')
                if model != 'consensus':
                    print(f"   Modelo: {model}")
                else:
                    print(f"   Consenso de múltiples expertos")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:100]}")
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
        if check_ram_usage():
            print(f"⚠️  Prueba detenida por uso elevado de RAM después de solicitud {i+1}")
            break

        # Retraso entre solicitudes
        if i < max_requests - 1:  # No esperar después de la última solicitud
            print(f"⏳ Esperando {delay_between_requests}s antes de siguiente solicitud...")
            time.sleep(delay_between_requests)

    # Resultados finales
    print("\\n" + "="*60)
    print("📊 RESULTADOS FINALES DE PRUEBA DE CONSENSO")
    print("="*60)
    
    if latencies:
        print(f"✅ Solicitudes exitosas: {successful_requests}")
        print(f"❌ Solicitudes fallidas: {failed_requests}")
        print(f"📊 Total de solicitudes intentadas: {successful_requests + failed_requests}")
        
        print(f"\\n⏱️  RENDIMIENTO:")
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
        print(f"   Tasa de éxito: {(successful_requests/(successful_requests+failed_requests)*100):.1f}%")
    else:
        print("❌ No se completaron solicitudes exitosas")
    
    print(f"\\n💾 RAM final: {get_ram_usage_percent():.1f}%")
    print("✅ Prueba de latencia para sistema de consenso completada")


def main():
    """Función principal"""
    print("🦫 Prueba Segura de Latencia - Sistema de Consenso ARM-Axion")
    print("   Monitoreo de RAM para evitar bloqueos del servidor")
    print("   Límite: 90% de uso de RAM")
    print("="*70)
    
    # Verificar que el servidor de consenso esté disponible
    server_url = "http://localhost:8084"
    print(f"🔍 Verificando servidor de consenso en {server_url}...")
    
    if is_server_responding(server_url):
        print("✅ Servidor de consenso detectado")
        test_consensus_latency(server_url)
    else:
        print(f"❌ Servidor de consenso no disponible en {server_url}")
        print("   Asegúrate de que el servidor esté corriendo en el puerto 8084")
        print("   Puedes iniciarlo con: bash start_consensus_server.sh")


if __name__ == "__main__":
    main()