#!/usr/bin/env python3
"""
Script de prueba para el endpoint /api/chat del servidor capibara6
"""

import requests
import json

def test_chat_endpoint():
    """Prueba el endpoint /api/chat para verificar que funciona correctamente"""
    
    # URL del endpoint
    url = "http://localhost:9002/api/chat"  # Puerto del gateway server actualizado
    
    # Datos de prueba
    payload = {
        "model": "aya_expanse_multilingual",
        "message": "Hola, ¿cómo estás?",
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("🧪 Probando endpoint /api/chat...")
    print(f"📍 Enviando solicitud a: {url}")
    print(f"📄 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        print(f"\n📈 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ ¡Éxito! El endpoint responde correctamente.")
            try:
                response_data = response.json()
                print(f"📄 Respuesta: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            except:
                print(f"📄 Respuesta (no JSON): {response.text}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Detalles: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Timeout: La solicitud tardó demasiado en responder.")
        print("💡 Posible causa: La VM de modelos (34.175.48.2:8082) no está accesible.")
    except requests.exceptions.ConnectionError:
        print("🔌 Error de conexión: No se puede conectar con el servidor local.")
        print("💡 Asegúrate de que el servidor Flask esté corriendo en el puerto 5001.")
    except Exception as e:
        print(f"💥 Error inesperado: {str(e)}")

if __name__ == "__main__":
    test_chat_endpoint()