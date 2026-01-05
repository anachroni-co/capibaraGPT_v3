#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de verificación de configuración para Capibara6
Verifica que todas las API keys necesarias estén configuradas correctamente
"""

import os
import sys
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
import requests
import json

# Cargar variables de entorno
load_dotenv()

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    """Imprimir encabezado del script"""
    print(f"{Colors.BLUE}{Colors.BOLD}")
    print("=" * 60)
    print("🦫 CAPIBARA6 - VERIFICADOR DE CONFIGURACIÓN")
    print("=" * 60)
    print(f"{Colors.END}")

def print_section(title):
    """Imprimir título de sección"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}📋 {title}{Colors.END}")
    print("-" * 40)

def check_env_var(var_name, required=True, description=""):
    """Verificar si una variable de entorno está configurada"""
    value = os.getenv(var_name)
    if value and value != f"tu_{var_name.lower()}" and not value.startswith("xxxxxxxx"):
        print(f"✅ {var_name}: {Colors.GREEN}Configurado{Colors.END}")
        return True
    else:
        status = f"{Colors.RED}No configurado{Colors.END}" if required else f"{Colors.YELLOW}Opcional{Colors.END}"
        print(f"❌ {var_name}: {status}")
        if description:
            print(f"   💡 {description}")
        return not required

def test_smtp():
    """Probar configuración SMTP"""
    print_section("CONFIGURACIÓN SMTP")
    
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = os.getenv('SMTP_PORT', '587')
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    
    if not all([smtp_server, smtp_user, smtp_password]):
        print("❌ Configuración SMTP incompleta")
        return False
    
    try:
        print("🔍 Probando conexión SMTP...")
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
        print("✅ Conexión SMTP exitosa")
        return True
    except Exception as e:
        print(f"❌ Error SMTP: {e}")
        return False

def test_openai():
    """Probar API de OpenAI"""
    print_section("OPENAI API")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key.startswith('sk-xxxxxxxx'):
        print("❌ OpenAI API Key no configurada")
        return False
    
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        response = requests.get('https://api.openai.com/v1/models', headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ OpenAI API conectada correctamente")
            return True
        else:
            print(f"❌ Error OpenAI API: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error conectando a OpenAI: {e}")
        return False

def test_anthropic():
    """Probar API de Anthropic"""
    print_section("ANTHROPIC API")
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key or api_key.startswith('sk-ant-xxxxxxxx'):
        print("❌ Anthropic API Key no configurada")
        return False
    
    try:
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json'
        }
        response = requests.get('https://api.anthropic.com/v1/messages', headers=headers, timeout=10)
        if response.status_code in [200, 400]:  # 400 es normal si no enviamos datos
            print("✅ Anthropic API conectada correctamente")
            return True
        else:
            print(f"❌ Error Anthropic API: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error conectando a Anthropic: {e}")
        return False

def test_google_ai():
    """Probar API de Google AI"""
    print_section("GOOGLE AI API")
    
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if not api_key or api_key.startswith('AIzaSyxxxxxxxx'):
        print("❌ Google AI API Key no configurada")
        return False
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("✅ Google AI API conectada correctamente")
            return True
        else:
            print(f"❌ Error Google AI API: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error conectando a Google AI: {e}")
        return False

def test_huggingface():
    """Probar API de Hugging Face"""
    print_section("HUGGING FACE API")
    
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    if not api_key or api_key.startswith('hf_xxxxxxxx'):
        print("❌ Hugging Face API Key no configurada")
        return False
    
    try:
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.get('https://huggingface.co/api/whoami', headers=headers, timeout=10)
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ Hugging Face API conectada como: {user_info.get('name', 'Usuario')}")
            return True
        else:
            print(f"❌ Error Hugging Face API: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error conectando a Hugging Face: {e}")
        return False

def test_pinecone():
    """Probar API de Pinecone"""
    print_section("PINECONE API")
    
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key or api_key.startswith('xxxxxxxx-xxxx'):
        print("❌ Pinecone API Key no configurada")
        return False
    
    try:
        headers = {'Api-Key': api_key}
        response = requests.get('https://api.pinecone.io/actions/whoami', headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ Pinecone API conectada correctamente")
            return True
        else:
            print(f"❌ Error Pinecone API: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error conectando a Pinecone: {e}")
        return False

def test_github():
    """Probar API de GitHub"""
    print_section("GITHUB API")
    
    token = os.getenv('GITHUB_TOKEN')
    if not token or token.startswith('ghp_xxxxxxxx'):
        print("❌ GitHub Token no configurado")
        return False
    
    try:
        headers = {'Authorization': f'token {token}'}
        response = requests.get('https://api.github.com/user', headers=headers, timeout=10)
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ GitHub API conectada como: {user_info.get('login', 'Usuario')}")
            return True
        else:
            print(f"❌ Error GitHub API: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error conectando a GitHub: {e}")
        return False

def check_required_config():
    """Verificar configuración requerida"""
    print_section("CONFIGURACIÓN REQUERIDA")
    
    required_vars = [
        ('SMTP_SERVER', 'Servidor SMTP para emails'),
        ('SMTP_PORT', 'Puerto SMTP'),
        ('SMTP_USER', 'Usuario SMTP'),
        ('SMTP_PASSWORD', 'Contraseña SMTP'),
        ('FROM_EMAIL', 'Email de origen')
    ]
    
    all_configured = True
    for var, desc in required_vars:
        if not check_env_var(var, required=True, description=desc):
            all_configured = False
    
    return all_configured

def check_optional_config():
    """Verificar configuración opcional"""
    print_section("CONFIGURACIÓN OPCIONAL")
    
    optional_vars = [
        ('OPENAI_API_KEY', 'API Key de OpenAI'),
        ('ANTHROPIC_API_KEY', 'API Key de Anthropic'),
        ('GOOGLE_AI_API_KEY', 'API Key de Google AI'),
        ('HUGGINGFACE_API_KEY', 'API Key de Hugging Face'),
        ('PINECONE_API_KEY', 'API Key de Pinecone'),
        ('WEAVIATE_URL', 'URL de Weaviate'),
        ('E2B_API_KEY', 'API Key de E2B'),
        ('GITHUB_TOKEN', 'Token de GitHub'),
        ('RAILWAY_TOKEN', 'Token de Railway'),
        ('VERCEL_TOKEN', 'Token de Vercel')
    ]
    
    configured_count = 0
    for var, desc in optional_vars:
        if check_env_var(var, required=False, description=desc):
            configured_count += 1
    
    print(f"\n📊 Configuración opcional: {configured_count}/{len(optional_vars)} servicios configurados")
    return configured_count

def main():
    """Función principal"""
    print_header()
    
    # Verificar que existe el archivo .env
    if not os.path.exists('.env'):
        print(f"{Colors.RED}❌ Archivo .env no encontrado{Colors.END}")
        print("💡 Copia .env.example a .env y configura tus claves:")
        print("   cp .env.example .env")
        sys.exit(1)
    
    print("✅ Archivo .env encontrado")
    
    # Verificar configuración requerida
    required_ok = check_required_config()
    
    # Verificar configuración opcional
    optional_count = check_optional_config()
    
    # Probar servicios configurados
    print_section("PRUEBAS DE CONECTIVIDAD")
    
    tests = [
        ("SMTP", test_smtp),
        ("OpenAI", test_openai),
        ("Anthropic", test_anthropic),
        ("Google AI", test_google_ai),
        ("Hugging Face", test_huggingface),
        ("Pinecone", test_pinecone),
        ("GitHub", test_github)
    ]
    
    passed_tests = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ Error en prueba {test_name}: {e}")
    
    # Resumen final
    print_section("RESUMEN")
    
    if required_ok:
        print(f"✅ {Colors.GREEN}Configuración requerida: COMPLETA{Colors.END}")
    else:
        print(f"❌ {Colors.RED}Configuración requerida: INCOMPLETA{Colors.END}")
    
    print(f"📊 Servicios opcionales configurados: {optional_count}")
    print(f"🧪 Pruebas de conectividad: {passed_tests}/{len(tests)}")
    
    if required_ok:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ¡Configuración básica lista!{Colors.END}")
        print("💡 Puedes ejecutar el backend con: cd backend && python server.py")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  Configuración incompleta{Colors.END}")
        print("💡 Revisa las variables de entorno requeridas")
    
    print(f"\n{Colors.BLUE}📚 Para más información, consulta API_KEYS_GUIDE.md{Colors.END}")

if __name__ == '__main__':
    main()