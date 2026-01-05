#!/usr/bin/env python3
"""
Configuración para producción - Capibara6
"""

import os

# Configuración de dominios
PRODUCTION_CONFIG = {
    'frontend_url': 'https://capibara6.com',
    'auth_server_url': 'https://api.capibara6.com',  # Cambiar por tu dominio del servidor de auth
    'allowed_origins': [
        'https://capibara6.com',
        'http://capibara6.com',  # Para desarrollo
        'http://localhost:8000',  # Para desarrollo local
        'http://127.0.0.1:8000'   # Para desarrollo local
    ]
}

# URLs de callback OAuth
OAUTH_CALLBACKS = {
    'github': f"{PRODUCTION_CONFIG['auth_server_url']}/auth/callback/github",
    'google': f"{PRODUCTION_CONFIG['auth_server_url']}/auth/callback/google"
}

def get_frontend_url():
    """Obtiene la URL del frontend según el entorno"""
    return os.environ.get('FRONTEND_URL', PRODUCTION_CONFIG['frontend_url'])

def get_auth_server_url():
    """Obtiene la URL del servidor de auth según el entorno"""
    return os.environ.get('AUTH_SERVER_URL', PRODUCTION_CONFIG['auth_server_url'])

def get_allowed_origins():
    """Obtiene los orígenes permitidos según el entorno"""
    return os.environ.get('ALLOWED_ORIGINS', PRODUCTION_CONFIG['allowed_origins'])

if __name__ == '__main__':
    print("🌐 Configuración de Producción:")
    print(f"   Frontend URL: {get_frontend_url()}")
    print(f"   Auth Server URL: {get_auth_server_url()}")
    print(f"   GitHub Callback: {OAUTH_CALLBACKS['github']}")
    print(f"   Google Callback: {OAUTH_CALLBACKS['google']}")
    print(f"   Allowed Origins: {get_allowed_origins()}")
