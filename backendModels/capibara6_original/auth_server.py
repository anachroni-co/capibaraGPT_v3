#!/usr/bin/env python3
"""
Servidor de autenticación OAuth para Capibara6
Maneja autenticación con GitHub y Google
"""

import os
import json
import secrets
import requests
from urllib.parse import urlencode, parse_qs
from flask import Flask, request, redirect, jsonify, session
from flask_cors import CORS
import jwt
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
# Configuración CORS - Desarrollo (localhost)
CORS(app, origins=['http://localhost:8000', 'http://127.0.0.1:8000', 'http://localhost:5500', 'http://127.0.0.1:5500'])

# Configuración CORS - Producción (comentado para activar más tarde)
# CORS(app, origins=['http://localhost:8000', 'http://127.0.0.1:8000', 'https://capibara6.com', 'http://capibara6.com'])

# Configuración OAuth
OAUTH_CONFIG = {
    'github': {
        'client_id': os.environ.get('GITHUB_CLIENT_ID', 'your_github_client_id'),
        'client_secret': os.environ.get('GITHUB_CLIENT_SECRET', 'your_github_client_secret'),
        'authorize_url': 'https://github.com/login/oauth/authorize',
        'token_url': 'https://github.com/login/oauth/access_token',
        'user_url': 'https://api.github.com/user'
    },
    'google': {
        'client_id': os.environ.get('GOOGLE_CLIENT_ID', 'your_google_client_id'),
        'client_secret': os.environ.get('GOOGLE_CLIENT_SECRET', 'your_google_client_secret'),
        'authorize_url': 'https://accounts.google.com/o/oauth2/v2/auth',
        'token_url': 'https://oauth2.googleapis.com/token',
        'user_url': 'https://www.googleapis.com/oauth2/v2/userinfo'
    }
}

# JWT Secret
JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_hex(32))

# ============================================
# Utility Functions
# ============================================
def generate_jwt_token(user_data):
    """Genera un token JWT para el usuario"""
    payload = {
        'user_id': user_data['id'],
        'email': user_data['email'],
        'name': user_data['name'],
        'provider': user_data['provider'],
        'exp': datetime.utcnow() + timedelta(days=7),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_jwt_token(token):
    """Verifica y decodifica un token JWT"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_user_from_github(token):
    """Obtiene información del usuario desde GitHub"""
    headers = {'Authorization': f'token {token}'}
    response = requests.get(OAUTH_CONFIG['github']['user_url'], headers=headers)
    
    if response.status_code == 200:
        user_data = response.json()
        return {
            'id': f"github_{user_data['id']}",
            'name': user_data['name'] or user_data['login'],
            'email': user_data.get('email', ''),
            'avatar': user_data['avatar_url'],
            'provider': 'github',
            'username': user_data['login']
        }
    return None

def get_user_from_google(token):
    """Obtiene información del usuario desde Google"""
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(OAUTH_CONFIG['google']['user_url'], headers=headers)
    
    if response.status_code == 200:
        user_data = response.json()
        return {
            'id': f"google_{user_data['id']}",
            'name': user_data['name'],
            'email': user_data['email'],
            'avatar': user_data['picture'],
            'provider': 'google'
        }
    return None

# ============================================
# Routes
# ============================================
@app.route('/auth/github')
def github_auth():
    """Inicia el flujo de autenticación con GitHub"""
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state
    
    params = {
        'client_id': OAUTH_CONFIG['github']['client_id'],
        'redirect_uri': f"{request.host_url}auth/callback/github",
        'scope': 'user:email',
        'state': state
    }
    
    auth_url = f"{OAUTH_CONFIG['github']['authorize_url']}?{urlencode(params)}"
    return redirect(auth_url)

@app.route('/auth/google')
def google_auth():
    """Inicia el flujo de autenticación con Google"""
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state
    
    params = {
        'client_id': OAUTH_CONFIG['google']['client_id'],
        'redirect_uri': f"{request.host_url}auth/callback/google",
        'response_type': 'code',
        'scope': 'openid email profile',
        'state': state
    }
    
    auth_url = f"{OAUTH_CONFIG['google']['authorize_url']}?{urlencode(params)}"
    return redirect(auth_url)

@app.route('/auth/callback/github')
def github_callback():
    """Maneja el callback de GitHub"""
    code = request.args.get('code')
    state = request.args.get('state')
    
    # Verificar state
    if state != session.get('oauth_state'):
        return jsonify({'error': 'Invalid state parameter'}), 400
    
    # Intercambiar código por token
    token_data = {
        'client_id': OAUTH_CONFIG['github']['client_id'],
        'client_secret': OAUTH_CONFIG['github']['client_secret'],
        'code': code
    }
    
    response = requests.post(
        OAUTH_CONFIG['github']['token_url'],
        data=token_data,
        headers={'Accept': 'application/json'}
    )
    
    if response.status_code == 200:
        token_info = response.json()
        access_token = token_info.get('access_token')
        
        if access_token:
            # Obtener información del usuario
            user_data = get_user_from_github(access_token)
            
            if user_data:
                # Generar JWT
                jwt_token = generate_jwt_token(user_data)
                
                # Redirigir al frontend con el token
                # Desarrollo (localhost)
                frontend_url = f"http://localhost:8000/auth/success?token={jwt_token}"
                
                # Producción (comentado para activar más tarde)
                # frontend_url = f"https://capibara6.com/auth/success?token={jwt_token}"
                
                return redirect(frontend_url)
    
    return jsonify({'error': 'Authentication failed'}), 400

@app.route('/auth/callback/google')
def google_callback():
    """Maneja el callback de Google"""
    code = request.args.get('code')
    state = request.args.get('state')
    
    # Verificar state
    if state != session.get('oauth_state'):
        return jsonify({'error': 'Invalid state parameter'}), 400
    
    # Intercambiar código por token
    token_data = {
        'client_id': OAUTH_CONFIG['google']['client_id'],
        'client_secret': OAUTH_CONFIG['google']['client_secret'],
        'code': code,
        'grant_type': 'authorization_code',
        'redirect_uri': f"{request.host_url}auth/callback/google"
    }
    
    response = requests.post(OAUTH_CONFIG['google']['token_url'], data=token_data)
    
    if response.status_code == 200:
        token_info = response.json()
        access_token = token_info.get('access_token')
        
        if access_token:
            # Obtener información del usuario
            user_data = get_user_from_google(access_token)
            
            if user_data:
                # Generar JWT
                jwt_token = generate_jwt_token(user_data)
                
                # Redirigir al frontend con el token
                # Desarrollo (localhost)
                frontend_url = f"http://localhost:8000/auth/success?token={jwt_token}"
                
                # Producción (comentado para activar más tarde)
                # frontend_url = f"https://capibara6.com/auth/success?token={jwt_token}"
                
                return redirect(frontend_url)
    
    return jsonify({'error': 'Authentication failed'}), 400

@app.route('/auth/verify', methods=['POST'])
def verify_token():
    """Verifica un token JWT"""
    data = request.get_json()
    token = data.get('token')
    
    if not token:
        return jsonify({'error': 'No token provided'}), 400
    
    payload = verify_jwt_token(token)
    
    if payload:
        return jsonify({
            'valid': True,
            'user': {
                'id': payload['user_id'],
                'email': payload['email'],
                'name': payload['name'],
                'provider': payload['provider']
            }
        })
    else:
        return jsonify({'valid': False}), 401

@app.route('/auth/logout', methods=['POST'])
def logout():
    """Cierra la sesión del usuario"""
    # En un sistema real, aquí invalidarías el token
    return jsonify({'message': 'Logged out successfully'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'capibara6-auth',
        'timestamp': datetime.utcnow().isoformat()
    })

# ============================================
# Error Handlers
# ============================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================
# Main
# ============================================
if __name__ == '__main__':
    print("🚀 Iniciando servidor de autenticación Capibara6...")
    print("📋 Configuración:")
    print(f"   - GitHub Client ID: {'✅ Configurado' if OAUTH_CONFIG['github']['client_id'] != 'your_github_client_id' else '❌ No configurado'}")
    print(f"   - Google Client ID: {'✅ Configurado' if OAUTH_CONFIG['google']['client_id'] != 'your_google_client_id' else '❌ No configurado'}")
    print(f"   - JWT Secret: {'✅ Configurado' if JWT_SECRET != 'your_jwt_secret' else '❌ No configurado'}")
    print("\n🔧 Para configurar OAuth:")
    print("   1. Crear apps en GitHub y Google")
    print("   2. Configurar variables de entorno:")
    print("      export GITHUB_CLIENT_ID='tu_client_id'")
    print("      export GITHUB_CLIENT_SECRET='tu_client_secret'")
    print("      export GOOGLE_CLIENT_ID='tu_client_id'")
    print("      export GOOGLE_CLIENT_SECRET='tu_client_secret'")
    print("      export JWT_SECRET='tu_jwt_secret'")
    print("\n🌐 Servidor ejecutándose en: http://localhost:5001")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
