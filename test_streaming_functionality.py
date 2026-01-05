#!/usr/bin/env python3
"""
Prueba simple de streaming para verificar que el servidor con streaming funciona correctamente
"""

import requests
import json
import time

def test_streaming():
    """
    Prueba del servidor con streaming verdadero
    """
    print("🚀 Probando servidor con streaming verdadero (puerto 8083)...")
    print("="*60)
    
    # Verificar que el servidor esté corriendo
    try:
        response = requests.get("http://localhost:8083/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Servidor de streaming saludable: {health}")
        else:
            print(f"❌ Servidor de streaming no responde: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error conectando con servidor de streaming: {e}")
        return
    
    # Probar streaming
    print("\n📝 Probando streaming con modelo aya_expanse_multilingual...")
    
    url = "http://localhost:8083/v1/chat/completions"
    
    payload = {
        "model": "aya_expanse_multilingual",
        "messages": [
            {"role": "user", "content": "¿Qué es la inteligencia artificial? Responde brevemente."}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True  # Importante: habilitar streaming
    }
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
        
        if response.status_code == 200:
            print(f"✅ Conexión establecida, recibiendo tokens...")
            print("-" * 40)
            
            full_response = ""
            token_count = 0
            start_time = time.time()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remover 'data: '
                        
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            # Extraer el contenido del token
                            if 'choices' in data and len(data['choices']) > 0:
                                choice = data['choices'][0]
                                
                                # Para chat completions
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    if content:
                                        print(content, end='', flush=True)
                                        full_response += content
                                        token_count += 1
                                
                                # Para completions
                                elif 'text' in choice:
                                    text = choice['text']
                                    if text:
                                        print(text, end='', flush=True)
                                        full_response += text
                                        token_count += 1
                                        
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n{'-' * 40}")
            print(f"✅ Streaming completado!")
            print(f"   Tokens recibidos: {token_count}")
            print(f"   Tiempo total: {total_time:.2f}s")
            print(f"   Longitud respuesta: {len(full_response)} caracteres")
        else:
            print(f"❌ Error en la solicitud: {response.status_code}")
            print(f"   Detalles: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Timeout esperando respuesta del servidor")
    except Exception as e:
        print(f"❌ Error durante la prueba de streaming: {e}")
        import traceback
        traceback.print_exc()

def test_comparison():
    """
    Compara el rendimiento entre servidor estándar y con streaming
    """
    print("\n🔄 Comparando rendimiento entre servidores...")
    print("="*60)
    
    # Probar servidor estándar
    print("\n📝 Servidor estándar (puerto 8082) - respuesta completa:")
    try:
        url = "http://localhost:8082/v1/chat/completions"
        payload = {
            "model": "aya_expanse_multilingual",
            "messages": [
                {"role": "user", "content": "¿Qué es el aprendizaje automático? Responde en una línea."}
            ],
            "max_tokens": 30,
            "temperature": 0.7,
            "stream": False  # Sin streaming
        }
        
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            response_time = end_time - start_time
            content = result['choices'][0]['message']['content']
            print(f"   ✅ Respuesta recibida en {response_time:.2f}s")
            print(f"   Contenido: {content[:60]}...")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Probar servidor con streaming (sin streaming para comparación justa)
    print("\n📝 Servidor con streaming (puerto 8083) - respuesta completa:")
    try:
        url = "http://localhost:8083/v1/chat/completions"
        payload = {
            "model": "aya_expanse_multilingual",
            "messages": [
                {"role": "user", "content": "¿Qué es el aprendizaje automático? Responde en una línea."}
            ],
            "max_tokens": 30,
            "temperature": 0.7,
            "stream": False  # Sin streaming para comparación justa
        }
        
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            response_time = end_time - start_time
            content = result['choices'][0]['message']['content']
            print(f"   ✅ Respuesta recibida en {response_time:.2f}s")
            print(f"   Contenido: {content[:60]}...")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Prueba de Servidor con Streaming Verdadero ARM-Axion")
    test_streaming()
    test_comparison()
    print("\n✅ Prueba completada")