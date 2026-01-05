#!/usr/bin/env python3
"""
Prueba de latencia ultrasegura para sistema ARM-Axion
Con monitoreo de RAM extremo para evitar bloqueos
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

def is_server_responding(url: str) -> bool:
    """Verifica si el servidor está respondiendo"""
    try:
        response = requests.get(f"{url}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def ultra_safe_test(server_url: str = "http://localhost:8082"):
    """
    Test ultra seguro - una sola solicitud pequeña para medir rendimiento sin riesgos
    """
    print("🚀 Iniciando PRUEBA ULTRA SEGURO de latencia...")
    print(f"   Servidor: {server_url}")
    print(f"   1 solicitud ligera para medir rendimiento")
    print(f"   Uso actual de RAM: {get_ram_usage_percent():.1f}%")
    print("="*60)

    if not is_server_responding(server_url):
        print(f"❌ Servidor no responde en {server_url}")
        return

    prompt = "¿Qué es ARM Axion?"

    print(f"\\n📝 Prompt: '{prompt}'")
    print(f"📊 RAM antes de solicitud: {get_ram_usage_percent():.1f}%")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "aya_expanse_multilingual",  # Modelo ya cargado
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 10,  # SOLAMENTE 10 tokens para ser ultra seguro
                "temperature": 0.7
            },
            timeout=30  # Tiempo de espera reducido
        )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            usage = result.get('usage', {})
            tokens_generated = usage.get('completion_tokens', 0)
            tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
            
            print(f"✅ Éxito: {total_time:.2f}s")
            print(f"   Tokens generados: {tokens_generated}")
            print(f"   Velocidad: {tokens_per_second:.2f} tok/s")
            print(f"   Modelo: {result.get('model', 'unknown')}")
            
            # Mostrar parte del contenido generado
            content = result['choices'][0]['message']['content']
            print(f"   Respuesta: '{content}'")
            
        else:
            print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

    print(f"\\n📊 RAM después de solicitud: {get_ram_usage_percent():.1f}%")
    print("✅ Prueba ultra segura completada")


if __name__ == "__main__":
    server_url = "http://localhost:8082"  # Servidor estándar
    print(f"🔍 Probando servidor: {server_url}")
    ultra_safe_test(server_url)