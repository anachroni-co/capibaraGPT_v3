#!/usr/bin/env python3
"""
Prueba LIGHT de la pregunta específica - versión ultra segura
"""

import requests
import time
import json
import psutil

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

def ultra_light_consensus_test():
    """
    Prueba ultra ligera para evitar problemas de RAM
    """
    print("🚀 INICIANDO PRUEBA ULTRA LIGERA DE PREGUNTA ESPECÍFICA")
    print("="*70)
    print("Pregunta: ¿Puede el ser humano ser completamente reemplazado por las nuevas IAS y")
    print("por los robots inteligentes en los próximos 20 años? ¿Qué probabilidades hay?")
    print("="*70)
    
    ram_before = get_ram_usage_percent()
    print(f"📊 RAM antes de prueba: {ram_before:.1f}%")
    
    # Intentar con servidor que esté disponible
    server_url = "http://localhost:8082"  # Este tiene un modelo ya cargado
    
    print(f"🔍 Usando servidor: {server_url}")
    
    if not is_server_responding(server_url):
        print(f"❌ Servidor no responde: {server_url}")
        return
    
    # Pregunta muy corta y objetivo claro para respuesta corta
    question = "¿Podrán las IAs reemplazar completamente a los humanos en 20 años? Porcentaje?"
    
    print(f"\\n📝 Pregunta optimizada: '{question}'")
    
    start_time = time.time()
    
    try:
        # Solicitud MUY LIGERA para evitar problemas de RAM
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "aya_expanse_multilingual",  # Modelo ya cargado
                "messages": [
                    {"role": "user", "content": question}
                ],
                "max_tokens": 25,  # MUY POQUITOS tokens para seguridad RAM
                "temperature": 0.7
            },
            timeout=45  # Tiempo razonable
        )
        
        total_time = time.time() - start_time
        ram_after = get_ram_usage_percent()
        
        print(f"\\n⏱️  Tiempo de respuesta: {total_time:.2f}s")
        print(f"📊 RAM después de prueba: {ram_after:.1f}%")
        print(f"📊 Cambio RAM: {ram_after - ram_before:+.1f}%")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            tokens_generated = result['usage']['completion_tokens']
            
            print(f"\\n✅ RESPUESTA OBTENIDA")
            print(f"🔢 Tokens generados: {tokens_generated}")
            print(f"⚡ Velocidad: {tokens_generated/total_time:.2f} tokens/seg")
            
            print(f"\\n📖 RESPUESTA BREVE:")
            print("-" * 40)
            print(content)
            print("-" * 40)
            
            model_used = result.get('model', 'unknown')
            print(f"\\n🤖 Modelo: {model_used}")
            
            # Mostrar uso de RAM final
            final_ram = get_ram_usage_percent()
            print(f"\\n📊 RAM final: {final_ram:.1f}%")
            
        else:
            print(f"❌ Error HTTP {response.status_code}")
            print(f"   Respuesta: {response.text[:200]}")
    
    except Exception as e:
        final_ram = get_ram_usage_percent()
        print(f"\\n📊 RAM final: {final_ram:.1f}%")
        print(f"❌ Error: {e}")

    print("\\n✅ Prueba ultra ligera completada con seguridad RAM")


if __name__ == "__main__":
    ultra_light_consensus_test()