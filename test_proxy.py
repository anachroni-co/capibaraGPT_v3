#!/usr/bin/env python3
"""
Test del proxy CORS local para Capibara6
"""

import requests
import sys

def test_proxy():
    print("🔍 Testing CORS Proxy for Capibara6...")
    print("")
    
    # Probar conexión con el proxy local
    proxy_url = "http://localhost:8001"
    
    print(f"📡 Probando conexión con el proxy local en {proxy_url}...")
    
    try:
        response = requests.get(proxy_url, timeout=5)
        if response.status_code == 200:
            print("✅ Proxy está corriendo")
            print(f"📋 Respuesta: {response.json()}")
        else:
            print(f"❌ Proxy respondió con código {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ No se pudo conectar con el proxy: {e}")
        print("   Asegúrate de que esté corriendo con: python3 cors_proxy_local.py")
        return False

    print("")
    
    # Probar endpoint de health
    print("🧪 Probando endpoint /api/health a través del proxy...")
    
    try:
        health_response = requests.get(f"{proxy_url}/api/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ Endpoint /api/health accesible a través del proxy")
            print(f"📋 Respuesta: {health_response.json()}")
        else:
            print(f"❌ Endpoint /api/health respondió con código {health_response.status_code}")
            print(f"   Contenido: {health_response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error probando /api/health: {e}")
        print("   El backend remoto puede no estar accesible o hay problemas de conexión")

    print("")
    
    # Probar endpoint de chat (debería fallar porque necesita POST)
    print("🧪 Probando endpoint /api/chat (debería dar 405 porque necesita POST)...")
    
    try:
        chat_response = requests.get(f"{proxy_url}/api/chat", timeout=10)
        if chat_response.status_code == 405:
            print("✅ Endpoint /api/chat accesible (devuelve 405 como es GET en lugar de POST)")
        elif chat_response.status_code == 200:
            print("✅ Endpoint /api/chat accesible")
        else:
            print(f"⚠️  Endpoint /api/chat respondió con código {chat_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error probando /api/chat: {e}")

    print("")
    print("🎉 Prueba de proxy completada!")
    return True

if __name__ == "__main__":
    test_proxy()