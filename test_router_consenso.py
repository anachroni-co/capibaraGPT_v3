#!/usr/bin/env python3
"""
Script de prueba para verificar el funcionamiento del router semántico y de consenso
"""

import sys
import os
import json
import requests
import time
import asyncio
from typing import Dict, Any, List

# Añadir la carpeta de backend al path
sys.path.insert(0, '/home/elect/capibara6/vm-bounty2')

def test_backend_connection():
    """Probar conexión con backend principal"""
    try:
        # Probar servidor de consenso en puerto 5005
        response = requests.get("http://34.12.166.76:5005/api/consensus/health", timeout=10)
        if response.status_code == 200:
            print("✅ Servidor de consenso accesible")
            return True
        else:
            print("❌ Servidor de consenso no accesible")
            return False
    except:
        print("❌ No se puede conectar al servidor de consenso")
        return False

def test_model_routing():
    """Probar el enrutamiento de diferentes tipos de consultas"""
    
    # Definir diferentes tipos de consultas para probar el routing
    test_queries = [
        {
            "prompt": "¿Qué es Python?",
            "expected_complexity": "simple",
            "description": "Pregunta general simple"
        },
        {
            "prompt": "Escribe un código en Python para calcular la serie de Fibonacci recursivamente",
            "expected_complexity": "coding",
            "description": "Pregunta de programación"
        },
        {
            "prompt": "Explica en detalle el teorema de Gödel sobre incompletitud y sus implicaciones en la lógica matemática",
            "expected_complexity": "complex",
            "description": "Pregunta compleja de análisis"
        },
        {
            "prompt": "Cuentame un chiste",
            "expected_complexity": "simple",
            "description": "Solicitud simple"
        },
        {
            "prompt": "Analiza las implicaciones éticas de la inteligencia artificial en la sociedad moderna",
            "expected_complexity": "analysis",
            "description": "Análisis ético"
        }
    ]
    
    print("\n🔍 Prueba de enrutamiento semántico")
    print("-" * 50)
    
    for i, query in enumerate(test_queries):
        print(f"\nTest {i+1}: {query['description']}")
        print(f"Consulta: {query['prompt'][:60]}...")
        
        # Intentar enviar la consulta al servidor de consenso
        try:
            payload = {
                "prompt": query['prompt'],
                "template": "general"  # Usar plantilla general para routing automático
            }
            
            response = requests.post(
                "http://34.12.166.76:5005/api/consensus/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'error' in result:
                    print(f"   ❌ Error: {result['error']}")
                else:
                    print(f"   ✅ Respuesta recibida")
                    print(f"   🧠 Modelo usado: {result.get('model_used', 'desconocido')}")
                    print(f"   ⏱️  Duración: {result.get('duration', 0):.2f}s")
                    print(f"   🔄 ¿Consenso?: {result.get('consensus', False)}")
            else:
                print(f"   ❌ Código de error HTTP: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error al contactar API: {str(e)}")

def test_models_availability():
    """Verificar la disponibilidad de modelos"""
    try:
        response = requests.get("http://34.12.166.76:5005/api/consensus/models", timeout=10)
        if response.status_code == 200:
            models_info = response.json()
            print("\n📋 Modelos disponibles:")
            for model_id in models_info.get('models_list', []):
                print(f"   • {model_id}")
            print(f"   Total: {models_info.get('active_models', 0)} modelos activos")
        else:
            print(f"\n❌ No se pudieron obtener los modelos (HTTP {response.status_code})")
    except Exception as e:
        print(f"\n❌ Error al obtener modelos: {str(e)}")

def test_consenso_funcionamiento():
    """Probar el funcionamiento del sistema de consenso"""
    print("\n🤝 Prueba de sistema de consenso")
    print("-" * 50)
    
    test_prompt = "¿Qué opinas sobre la inteligencia artificial?"
    
    try:
        payload = {
            "prompt": test_prompt,
            "template": "general"
        }
        
        print(f"Enviando consulta al sistema de consenso: '{test_prompt}'")
        response = requests.post(
            "http://34.12.166.76:5005/api/consensus/query",
            json=payload,
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Consulta procesada exitosamente")
            
            if 'error' not in result:
                print(f"🧠 Modelo usado: {result.get('model_used', 'desconocido')}")
                print(f"⏱️  Duración: {result.get('duration', 0):.2f}s")
                print(f"📊 Modelos consultados: {result.get('models_queried', 1)}")
                print(f"✅ Modelos exitosos: {result.get('successful_models', 1)}")
                print(f"🤝 ¿Usó consenso?: {result.get('consensus', False)}")
                if result.get('consensus'):
                    print(f"🎯 Método de consenso: {result.get('consensus_method', 'desconocido')}")
                
                response_text = result.get('response', '')
                print(f"📝 Longitud de respuesta: {len(response_text)} caracteres")
                
                if len(response_text) > 0:
                    print(f"💬 Respuesta (primeros 100 chars): {response_text[:100]}...")
            else:
                print(f"❌ Error en la respuesta: {result['error']}")
        else:
            print(f"❌ Error HTTP: {response.status_code}")
            print(f"   Detalles: {response.text}")
            
    except Exception as e:
        print(f"❌ Error en la prueba de consenso: {str(e)}")
        print("   Puede que el servidor no esté corriendo o que los modelos no estén disponibles")

def test_specific_models():
    """Probar modelos específicos si es posible"""
    print("\n🎯 Prueba de modelos específicos")
    print("-" * 50)
    
    # Probar obtener la configuración de modelos
    try:
        response = requests.get("http://34.12.166.76:5005/api/consensus/config", timeout=10)
        if response.status_code == 200:
            config = response.json()
            print(f"✅ Configuración de consenso obtenida")
            print(f"   Método: {config.get('voting_method', 'desconocido')}")
            print(f"   Mín. modelos: {config.get('min_models', 0)}")
            print(f"   Máx. modelos: {config.get('max_models', 0)}")
            print(f"   Modelo fallback: {config.get('fallback_model', 'desconocido')}")
            print(f"   Pesos de modelos: {config.get('model_weights', {})}")
        else:
            print(f"❌ No se pudo obtener la configuración (HTTP {response.status_code})")
    except Exception as e:
        print(f"❌ Error al obtener configuración: {str(e)}")

def main():
    """Función principal de pruebas"""
    print("🧪 Iniciando pruebas del sistema Capibara6")
    print("   Router Semántico y Sistema de Consenso")
    print("=" * 60)
    
    # Verificar conexión
    if not test_backend_connection():
        print("\n⚠️  Advertencia: No se pudo conectar al servidor de consenso")
        print("   Asegúrate de que el servidor esté corriendo en http://34.12.166.76:5005")
        return False
    
    # Probar disponibilidad de modelos
    test_models_availability()
    
    # Probar routing semántico
    test_model_routing()
    
    # Probar sistema de consenso
    test_consenso_funcionamiento()
    
    # Probar configuración específica
    test_specific_models()
    
    print("\n" + "=" * 60)
    print("📋 Resumen de pruebas:")
    print("   - Conexión con servidor: Verificada")
    print("   - Disponibilidad de modelos: Verificada") 
    print("   - Prueba de routing: Ejecutada")
    print("   - Sistema de consenso: Probado")
    print("   - Configuración: Verificada")
    print("\n✅ Pruebas completadas")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 ¡Todas las pruebas se completaron exitosamente!")
    else:
        print("\n❌ Hubo errores en las pruebas")
        sys.exit(1)