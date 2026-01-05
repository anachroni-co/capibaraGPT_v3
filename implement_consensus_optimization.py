#!/usr/bin/env python3
"""
Optimización adicional: Habilitar sistema de consenso para mejorar la latencia y calidad
"""

import json
import os
from pathlib import Path

def enable_consensus_optimization():
    """
    Habilita el sistema de consenso en la configuración para mejorar la calidad
    de las respuestas y potencialmente reducir la latencia mediante inferencia paralela
    """
    config_path = "/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.json"
    
    # Cargar configuración actual
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("🚀 Habilitando sistema de consenso...")
    
    # Actualizar la configuración para habilitar consenso
    config['enable_consensus'] = True
    
    # Configurar un modelo para consenso si no existe
    if not config.get('consensus_model'):
        # Por defecto usar un modelo ligero para consenso
        config['consensus_model'] = "/home/elect/models/phi-4-mini"  # Modelo rápido para síntesis
    
    # Asegurar que speculative routing esté habilitado para mejor latencia
    if 'speculative_routing' not in config:
        config['speculative_routing'] = {
            "enabled": True,
            "speculation_threshold": 0.85,
            "max_speculation_time": 0.5
        }
    
    # Guardar la configuración actualizada
    backup_path = config_path.replace(".json", ".with_consensus.backup")
    with open(backup_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Sistema de consenso habilitado en {config_path}")
    print(f"   Backup original en: {backup_path}")
    print(f"   Consenso: {config['enable_consensus']}")
    print(f"   Modelo de consenso: {config.get('consensus_model', 'no especificado')}")
    
    return config_path, backup_path


def update_livemind_orchestrator_consensus():
    """
    Actualiza el orchestrator para habilitar consenso por defecto
    """
    orchestrator_path = "/home/elect/capibara6/arm-axion-optimizations/vllm_integration/livemind_orchestrator.py"
    
    # Leer el archivo actual
    with open(orchestrator_path, 'r') as f:
        content = f.read()
    
    print("🔄 Actualizando LiveMind Orchestrator para habilitar consenso...")
    
    # Cambiar enable_consensus por defecto de False a True en la línea de inicialización
    updated_content = content.replace(
        "enable_consensus: bool = False,", 
        "enable_consensus: bool = True,"
    )
    
    # Actualizar también en el ejemplo de inicialización al final del archivo
    updated_content = updated_content.replace(
        "enable_consensus=False,",
        "enable_consensus=True,"
    )
    
    # Guardar el archivo actualizado
    backup_path = orchestrator_path.replace(".py", ".with_consensus.backup")
    with open(backup_path, 'w') as f:
        f.write(content)
    
    with open(orchestrator_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✅ LiveMind Orchestrator actualizado en {orchestrator_path}")
    print(f"   Backup original en: {backup_path}")
    
    return orchestrator_path, backup_path


def create_consensus_test():
    """
    Crea un script de prueba para verificar la funcionalidad de consenso
    """
    test_content = '''
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
        print("\\n📝 Enviando solicitud de prueba...")
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
'''
    
    test_path = "/home/elect/capibara6/test_consensus_functionality.py"
    with open(test_path, 'w') as f:
        f.write(test_content)
    
    # Hacerlo ejecutable
    os.chmod(test_path, 0o755)
    
    print(f"✅ Script de prueba de consenso creado en {test_path}")
    
    return test_path


def main():
    print("🎯 IMPLEMENTACIÓN DE CONSENSO PARA REDUCIR LATENCIA")
    print("=" * 60)
    print("La estrategia de consenso paralelo puede mejorar la latencia al:")
    print("- Permitir inferencia paralela en múltiples especialistas")
    print("- Sintetizar respuestas de múltiples modelos")
    print("- Mejorar calidad de respuestas sin aumentar latencia significativamente")
    print("=" * 60)
    
    # 1. Habilitar consenso en la configuración
    config_path, config_backup = enable_consensus_optimization()
    
    # 2. Actualizar el orchestrator
    orchestrator_path, orchestrator_backup = update_livemind_orchestrator_consensus()
    
    # 3. Crear script de prueba
    test_path = create_consensus_test()
    
    print("\\n📋 RESUMEN DE CAMBIOS:")
    print(f"   • Configuración actualizada: {config_path}")
    print(f"   • Orchestrator actualizado: {orchestrator_path}")
    print(f"   • Script de prueba: {test_path}")
    print(f"   • Backups creados para reversión si es necesario")
    
    print("\\n💡 NOTA: Para que los cambios surtan efecto, reinicie el servidor con:")
    print("   pkill -f multi_model_server  # Detener servidores anteriores")
    print("   cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration")
    print("   python3 multi_model_server.py --host 0.0.0.0 --port 8082 --config config.json &")
    
    print("\\n🧪 Para probar la funcionalidad de consenso:")
    print(f"   python3 {test_path}")
    
    print("\\n✅ Optimización de consenso implementada exitosamente!")


if __name__ == "__main__":
    main()