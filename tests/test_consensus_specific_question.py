#!/usr/bin/env python3
"""
Prueba específica para el sistema de consenso ARM-Axion
Con control de RAM para prevenir bloqueos del servidor
"""

import requests
import time
import json
import psutil
from typing import Dict, List

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

def test_consensus_question():
    """
    Prueba específica para el sistema de consenso con la pregunta:
    "¿Puede el ser humano ser completamente reemplazado por las nuevas IAS y por los robots inteligentes 
    en los próximos 20 años? ¿Que probabilidades hay de ese hecho?"
    """
    print("🚀 INICIANDO PRUEBA DE CONSENSO ESPECÍFICA")
    print("="*70)
    print("Pregunta: ¿Puede el ser humano ser completamente reemplazado por las nuevas IAS y")
    print("por los robots inteligentes en los próximos 20 años? ¿Qué probabilidades hay?")
    print("="*70)
    
    # Verificar uso de RAM antes de comenzar
    initial_ram = get_ram_usage_percent()
    print(f"📊 RAM inicial: {initial_ram:.1f}%")
    
    if initial_ram > 90.0:
        print(f"⚠️  ¡ADVERTENCIA! RAM está en {initial_ram:.1f}%, cerca del límite")
        print("   Procediendo con cuidado...")
    
    # Probar si el servidor de consenso (8084) está disponible
    consensus_servers = [
        ("http://localhost:8084", "Servidor de Consenso"),
        ("http://localhost:8082", "Servidor Estándar")
    ]
    
    selected_server = None
    selected_name = None
    
    for server_url, server_name in consensus_servers:
        print(f"🔍 Verificando {server_name} en {server_url}...")
        if is_server_responding(server_url):
            selected_server = server_url
            selected_name = server_name
            print(f"✅ {server_name} disponible")
            break
        else:
            print(f"❌ {server_name} no disponible")
    
    if not selected_server:
        print("❌ No hay servidores disponibles")
        return
    
    print(f"\\n🎯 Usando: {selected_name} ({selected_server})")
    
    # Verificar RAM antes de enviar la solicitud
    if check_ram_usage(90.0):
        print("❌ Prueba cancelada por uso elevado de RAM")
        return
    
    # La pregunta específica
    question = "¿Puede el ser humano ser completamente reemplazado por las nuevas IAS y por los robots inteligentes en los próximos 20 años? ¿Qué probabilidades hay de ese hecho?"
    
    print(f"\\n📝 Enviando pregunta: '{question[:60]}...'")
    
    start_time = time.time()
    
    try:
        # Enviar solicitud al servidor (usando modelo que ya esté cargado para ser seguro)
        response = requests.post(
            f"{selected_server}/v1/chat/completions",
            json={
                "model": "aya_expanse_multilingual",  # Modelo ya cargado
                "messages": [
                    {"role": "user", "content": question}
                ],
                "max_tokens": 150,  # Limitar tokens para ser seguro
                "temperature": 0.7
            },
            timeout=120  # Tiempo suficiente para procesamiento de consenso
        )
        
        total_time = time.time() - start_time
        
        final_ram = get_ram_usage_percent()
        print(f"\\n📊 RAM final: {final_ram:.1f}%")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            tokens_generated = result['usage']['completion_tokens']
            
            print(f"\\n✅ RESPUESTA OBTENIDA CON ÉXITO")
            print(f"⏱️  Tiempo total: {total_time:.2f}s")
            print(f"🔢 Tokens generados: {tokens_generated}")
            print(f"⚡ Velocidad: {tokens_generated/total_time:.2f} tokens/seg")
            
            print(f"\\n📖 RESPUESTA DEL SISTEMA:")
            print("-" * 50)
            print(content)
            print("-" * 50)
            
            # Mostrar información adicional si está disponible
            model_used = result.get('model', 'unknown')
            print(f"\\n🤖 Modelo utilizado: {model_used}")
            
        else:
            print(f"❌ Error HTTP {response.status_code}")
            print(f"   Detalles: {response.text[:200]}")
    
    except Exception as e:
        final_ram = get_ram_usage_percent()
        print(f"\\n📊 RAM final: {final_ram:.1f}%")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\\n🎯 Prueba de consenso completada con RAM final: {get_ram_usage_percent():.1f}%")
    print("✅ El servidor no se bloqueó durante la prueba")


def main():
    """Función principal"""
    print("🦫 Prueba de Consenso - Pregunta Específica")
    print("   Sistema ARM-Axion con control de RAM")
    print("   Pregunta: Sobre reemplazo humano por IA/robots en 20 años")
    print("="*70)
    
    test_consensus_question()


if __name__ == "__main__":
    main()