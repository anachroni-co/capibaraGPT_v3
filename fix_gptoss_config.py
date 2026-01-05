#!/usr/bin/env python3
"""
Script para corregir la configuración del modelo gpt-oss-20b
Actualiza la ruta del modelo para usar el directorio original en lugar del directorio incorrecto
"""

import json
import os

def update_model_config():
    config_path = "/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models.all_loaded.json"
    
    # Cargar configuración
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Configuración original cargada con {len(config['experts'])} expertos")
    
    # Encontrar el experto gptoss_complex y corregir su ruta
    updated = False
    for expert in config['experts']:
        if expert['expert_id'] == 'gptoss_complex':
            old_path = expert['model_path']
            new_path = "/home/elect/models/gpt-oss-20b/original"
            
            print(f"Corrigiendo ruta para {expert['expert_id']}:")
            print(f"  Antes: {old_path}")
            print(f"  Después: {new_path}")
            
            expert['model_path'] = new_path
            updated = True
            break
    
    if not updated:
        print("ERROR: No se encontró el experto gptoss_complex en la configuración")
        return False
    
    # Guardar configuración actualizada
    backup_path = config_path + ".corrected"
    with open(backup_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuración actualizada guardada en: {backup_path}")
    
    # Mostrar resumen
    print("\nResumen de modelos:")
    for expert in config['experts']:
        status = "CORRECTO" if not expert['expert_id'] == 'gptoss_complex' or 'original' in expert['model_path'] else "VERIFICAR"
        print(f"  - {expert['expert_id']}: {expert['model_path']} [{status}]")
    
    return True

if __name__ == "__main__":
    print("🔄 Corrigiendo configuración del modelo gpt-oss-20b...")
    
    if update_model_config():
        print("\n✅ Configuración actualizada correctamente!")
        print("\nInstrucciones para reiniciar el servicio:")
        print("1. Detener el servidor actual:")
        print("   pkill -f multi_model_server.py")
        print("2. Iniciar el servidor con la nueva configuración:")
        print("   cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration && source /home/elect/venv/bin/activate && nohup python3 multi_model_server.py --host 0.0.0.0 --port 8082 --config config.five_models.all_loaded.json > server.log 2>&1 &")
        print("3. Verificar que los 5 modelos estén disponibles:")
        print("   curl http://localhost:8082/v1/models")
    else:
        print("\n❌ Error al actualizar la configuración")