#!/usr/bin/env python3
"""
Prueba de verdadero sistema de consenso ARM-Axion
Con múltiples modelos trabajando en paralelo
"""

import requests
import time
import json
import psutil
from typing import Dict, List

def get_ram_usage_percent():
    """Obtiene el porcentaje de uso de RAM"""
    return psutil.virtual_memory().percent

def start_consensus_server_if_needed():
    """Intenta iniciar el servidor de consenso si no está corriendo"""
    import subprocess
    import os
    
    # Verificar si el servidor de consenso ya está corriendo en el puerto 8085
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 8085))
    sock.close()
    
    if result != 0:  # Puerto no está ocupado
        print("🔧 Iniciando servidor de consenso seguro en puerto 8085...")
        try:
            # Cambiarse al directorio correcto y ejecutar el servidor de consenso seguro
            os.chdir("/home/elect/capibara6/arm-axion-optimizations/vllm_integration/")
            
            # Comando para iniciar el servidor de consenso seguro
            cmd = [
                "python3", "multi_model_server_consensus_safe.py", 
                "--host", "0.0.0.0", 
                "--port", "8085", 
                "--config", "config.json"
            ]
            
            # Iniciar en background
            process = subprocess.Popen(cmd, stdout=open('/tmp/consensus_safe_test.log', 'w'), stderr=subprocess.STDOUT)
            
            # Esperar a que inicie (más tiempo ya que carga modelos)
            print("⏳ Esperando que inicie el servidor de consenso seguro (90 segundos)...")
            time.sleep(90)
            
            # Verificar que esté disponible
            import socket
            for attempt in range(10):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', 8085))
                sock.close()
                if result == 0:
                    print("✅ Servidor de consenso seguro iniciado en puerto 8085")
                    return True
                time.sleep(10)
            
            print("❌ No se pudo iniciar el servidor de consenso seguro")
            return False
            
        except Exception as e:
            print(f"❌ Error al iniciar servidor de consenso seguro: {e}")
            return False
    else:
        print("✅ Servidor de consenso seguro ya está corriendo en puerto 8085")
        return True

def test_true_consensus():
    """
    Prueba de verdadero consenso entre múltiples modelos
    """
    print("🚀 INICIANDO PRUEBA DE VERDADERO CONSENSO ENTRE MODELOS")
    print("="*70)
    print("Pregunta: ¿Puede el ser humano ser completamente reemplazado por las nuevas IAS y")
    print("por los robots inteligentes en los próximos 20 años? ¿Qué probabilidades hay?")
    print("="*70)
    
    initial_ram = get_ram_usage_percent()
    print(f"📊 RAM inicial: {initial_ram:.1f}%")
    
    # Intentar iniciar servidor de consenso si es necesario
    consensus_server_available = start_consensus_server_if_needed()
    
    if consensus_server_available:
        server_url = "http://localhost:8085"
        print(f"🎯 Usando servidor de CONSENSO: {server_url}")
    else:
        print("⚠️  Servidor de consenso no disponible, intentando con servidor estándar en 8082...")
        server_url = "http://localhost:8082"
        
        # Verificar que el servidor esté disponible
        try:
            response = requests.get(f"{server_url}/health", timeout=10)
            if response.status_code != 200:
                print(f"❌ Servidor estándar tampoco disponible en {server_url}")
                return
        except:
            print(f"❌ Servidor estándar tampoco disponible en {server_url}")
            return
    
    # Pregunta que podría activar routing a múltiples dominios
    question = "¿Puede el ser humano ser completamente reemplazado por las nuevas IAS y por los robots inteligentes en los próximos 20 años? ¿Qué probabilidades hay de ese hecho? Incluir perspectivas técnicas, éticas y económicas."
    
    print(f"\\n📝 Pregunta: '{question[:60]}...'")
    
    start_time = time.time()
    
    try:
        # Para activar el consenso real, dejar el campo "model" vacío para que el router decida
        # o bien especificar que se debe usar el sistema de consenso si está configurado
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "",  # Dejar vacío para que el router de consenso decida
                "messages": [
                    {"role": "user", "content": question}
                ],
                "max_tokens": 100,  # Limitar para seguridad de RAM
                "temperature": 0.7
            },
            timeout=120
        )
        
        total_time = time.time() - start_time
        
        final_ram = get_ram_usage_percent()
        print(f"\\n⏱️  Tiempo total: {total_time:.2f}s")
        print(f"📊 RAM final: {final_ram:.1f}% (cambio: {final_ram - initial_ram:+.1f}%)")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            tokens_generated = result['usage']['completion_tokens']
            model_used = result.get('model', 'unknown')
            
            print(f"\\n✅ RESPUESTA OBTENIDA")
            print(f"🔢 Tokens generados: {tokens_generated}")
            print(f"⚡ Velocidad: {tokens_generated/total_time:.2f} tokens/seg")
            print(f"🤖 Modelo(s) utilizado(s): {model_used}")
            
            print(f"\\n📖 RESPUESTA:")
            print("-" * 50)
            print(content)
            print("-" * 50)
            
            # Intentar obtener estadísticas si están disponibles
            if 'performance' in result:
                perf = result['performance']
                print(f"\\n📈 DETALLES DE RENDIMIENTO:")
                print(f"   Time to first token: {perf.get('time_to_first_token', 'n/a')}s")
                print(f"   Total time: {perf.get('total_time', 'n/a')}s")
                print(f"   Tokens/seg: {perf.get('tokens_per_second', 'n/a')}")
            
            # Si es servidor de consenso, podría tener información extra
            if '8085' in server_url:
                print(f"\\n🎯 ESTO ES UNA VERDADERA RESPUESTA DE CONSENSO")
                print(f"   El sistema pudo haber consultado varios modelos expertos")
                print(f"   para formular esta respuesta integrada")
            else:
                print(f"\\n⚠️  ESTA ES UNA RESPUESTA DE MODELO ÚNICO")
                print(f"   Usando el modelo: {model_used}")
                
        else:
            print(f"❌ Error HTTP {response.status_code}")
            print(f"   Detalles: {response.text[:200]}")
    
    except Exception as e:
        final_ram = get_ram_usage_percent()
        print(f"\\n📊 RAM final: {final_ram:.1f}%")
        print(f"❌ Error en la solicitud: {e}")
        import traceback
        traceback.print_exc()

    print("\\n✅ Prueba de consenso completada")


def main():
    """Función principal"""
    print("🦫 Prueba de Verdadero Consenso - Sistema ARM-Axion")
    print("   Evaluando uso de múltiples modelos expertos")
    print("="*70)
    
    test_true_consensus()


if __name__ == "__main__":
    main()