#!/usr/bin/env python3
"""
Script de inicio para el servidor GPT-OSS-20B
"""

import os
import sys
import subprocess
import time
import requests
from config_gptoss import get_gpt_oss_url, get_server_port

def check_gpt_oss_connection():
    """Verificar conexión con GPT-OSS-20B"""
    try:
        response = requests.get(f"{get_gpt_oss_url()}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Conexión con GPT-OSS-20B establecida")
            return True
        else:
            print(f"❌ Error en GPT-OSS-20B: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ No se puede conectar con GPT-OSS-20B: {e}")
        return False

def test_gpt_oss_model():
    """Probar el modelo con una petición simple"""
    try:
        payload = {
            "prompt": "Hola, ¿cómo estás?",
            "n_predict": 20,
            "temperature": 0.7,
            "stream": False
        }
        
        response = requests.post(
            f"{get_gpt_oss_url()}/completion",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('content', '').strip()
            print(f"✅ Modelo respondiendo correctamente: {content[:50]}...")
            return True
        else:
            print(f"❌ Error en el modelo: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error probando el modelo: {e}")
        return False

def start_server():
    """Iniciar el servidor Flask"""
    print("🚀 Iniciando servidor capibara6 con GPT-OSS-20B...")
    print(f"📡 URL del modelo: {get_gpt_oss_url()}")
    print(f"🌐 Puerto del servidor: {get_server_port()}")
    
    # Verificar conexión con el modelo
    if not check_gpt_oss_connection():
        print("⚠️  Advertencia: No se puede conectar con GPT-OSS-20B")
        print("   El servidor se iniciará pero las peticiones fallarán")
    
    # Probar el modelo
    if not test_gpt_oss_model():
        print("⚠️  Advertencia: El modelo no responde correctamente")
        print("   Verifica que el servidor GPT-OSS esté funcionando")
    
    # Iniciar servidor
    try:
        from server_gptoss import app
        print("🦫 Servidor iniciado correctamente")
        app.run(host='0.0.0.0', port=get_server_port(), debug=False)
    except Exception as e:
        print(f"❌ Error iniciando servidor: {e}")
        sys.exit(1)

if __name__ == '__main__':
    start_server()
