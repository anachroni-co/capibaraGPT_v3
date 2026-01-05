
#!/usr/bin/env python3
"""
Prueba de funcionalidad de consenso
"""

import requests
import time
import json

def test_consensus_functionality():
    """
    Prueba que el sistema de consenso esté funcionando
    """
    print("🧪 Probando funcionalidad de consenso...")
    
    # Verificar estado del servidor
    try:
        response = requests.get("http://localhost:8082/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Servidor estado: {stats}")
            
            # Verificar si el consenso está habilitado
            if "config" in stats:
                consensus_enabled = stats["config"].get("enable_consensus", False)
                print(f"📊 Consenso habilitado: {consensus_enabled}")
                
                if consensus_enabled:
                    print("✅ Sistema de consenso está activado")
                else:
                    print("⚠️  Sistema de consenso no está activado")
            else:
                print("⚠️  No se pudo verificar estado de consenso")
        else:
            print(f"❌ Error al obtener estado: {response.status_code}")
    except Exception as e:
        print(f"❌ Error al conectar con servidor: {e}")
    
    # Probar una solicitud simple
    try:
        print("\n📝 Enviando solicitud de prueba...")
        response = requests.post(
            "http://localhost:8082/v1/chat/completions",
            json={
                "model": "",  # Dejar vacío para usar router automático
                "messages": [
                    {"role": "user", "content": "¿Cuál es el modelo más apropiado para tareas de codificación?"}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Respuesta recibida: {len(result.get('choices', []))} opciones")
            if result.get('choices'):
                content = result['choices'][0].get('message', {}).get('content', '')[:100]
                print(f"📄 Contenido (primeros 100 chars): {content}...")
        else:
            print(f"❌ Error en la solicitud: {response.status_code}, {response.text}")
            
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")

if __name__ == "__main__":
    test_consensus_functionality()
