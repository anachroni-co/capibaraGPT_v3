#!/usr/bin/env python3
"""
Configuración para GPT-OSS-20B - Capibara6
"""

import os

# Configuración de la VM GPT-OSS-20B
GPT_OSS_CONFIG = {
    'url': 'http://34.175.215.109:8080',
    'timeout': 60,
    'max_tokens': 1000,
    'temperature': 0.7,
    'model_name': 'gpt-oss-20b'
}

# Configuración del servidor Flask
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.getenv('PORT', 5000)),
    'debug': False
}

# Configuración CORS
CORS_CONFIG = {
    'allowed_origins': [
        'http://localhost:8000',
        'http://127.0.0.1:8000',
        'https://capibara6.com',
        'http://capibara6.com'
    ]
}

def get_gpt_oss_url():
    """Obtener URL del modelo GPT-OSS"""
    return os.getenv('GPT_OSS_URL', GPT_OSS_CONFIG['url'])

def get_gpt_oss_timeout():
    """Obtener timeout para GPT-OSS"""
    return int(os.getenv('GPT_OSS_TIMEOUT', GPT_OSS_CONFIG['timeout']))

def get_server_port():
    """Obtener puerto del servidor"""
    return int(os.getenv('PORT', SERVER_CONFIG['port']))

if __name__ == '__main__':
    print("🤖 Configuración GPT-OSS-20B:")
    print(f"   URL: {get_gpt_oss_url()}")
    print(f"   Timeout: {get_gpt_oss_timeout()}s")
    print(f"   Puerto: {get_server_port()}")
    print(f"   Orígenes CORS: {CORS_CONFIG['allowed_origins']}")
