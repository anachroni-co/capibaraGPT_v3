# DOCUMENTACIÓN: SCRIPTS PARA VLLM EN ARM-AXION

## 1. SCRIPT DE INICIO DEL SERVIDOR MULTI-MODELO

**Archivo**: `/home/elect/capibara6/start_vllm_arm_axion.sh`

### Descripción:
Script que inicia el servidor vLLM ARM-Axion con los 5 modelos optimizados: Phi4-mini, Qwen2.5-coder, Mistral7B, Gemma3-27B y GPT-OSS-20B.

### Uso:
```bash
# Forma básica de uso
./start_vllm_arm_axion.sh [puerto] [host] [configuración]

# Ejemplos:
./start_vllm_arm_axion.sh                    # Puerto 8080, host 0.0.0.0, config predeterminada
./start_vllm_arm_axion.sh 8081               # Puerto 8081, host 0.0.0.0, config predeterminada
./start_vllm_arm_axion.sh 8081 0.0.0.0 config.production.json  # Con todos los parámetros
```

### Funcionalidades:
- Configura automáticamente el PYTHONPATH con la versión modificada de vLLM
- Verifica que se detecte correctamente la plataforma ARM-Axion como CPU
- Inicia el servidor en el puerto especificado
- Usa la configuración de modelos ARM-Axion optimizados

## 2. SCRIPT DE PRUEBA INTERACTIVA

**Archivo**: `/home/elect/capibara6/interactive_test_interface.py`

### Descripción:
Script interactivo que permite probar los 5 modelos individualmente, el sistema de router semántico, el sistema de consenso y realizar análisis comparativos entre modelos.

### Uso:
```bash
# Iniciar la interfaz interactiva
python3 interactive_test_interface.py
```

### Funcionalidades del menú:
1. **Probar modelo individual** - Test de un solo modelo con tu consulta
2. **Probar sistema de router semántico** - Análisis de routing inteligente
3. **Probar sistema de consenso** - Votación entre modelos
4. **Probar todos los modelos con análisis comparativo** - Comparación de rendimiento
5. **Información del sistema** - Estado del sistema ARM-Axion
6. **Salir**

### Características:
- Compatible con los 5 modelos ARM-Axion
- Sistema de router semántico para asignación inteligente
- Sistema de consenso para verificación de respuestas
- Comparación de rendimiento entre modelos
- Verificación de plataforma ARM-Axion

## ESTADO ACTUAL DE LA IMPLEMENTACIÓN

✅ **Todo está funcionando correctamente**:
- Detección de plataforma ARM64 como CPU: FUNCIONAL
- Servidor multi-modelo ARM-Axion: FUNCIONAL
- 5 modelos disponibles (Qwen2.5, Phi4-mini, Mistral7B, Gemma3-27B, GPT-OSS-20B): FUNCIONALES
- Interfaz interactiva: FUNCIONAL
- Optimizaciones ARM (NEON, ACL, cuantización): IMPLEMENTADAS

## EJEMPLO DE USO COMPLETO

1. **Iniciar el servidor** (en una terminal o en background):
```bash
cd /home/elect/capibara6
./start_vllm_arm_axion.sh 8081
```

2. **Probar los modelos de forma interactiva** (en otra terminal):
```bash
cd /home/elect/capibara6
python3 interactive_test_interface.py
```

3. **Enviar consultas a través de la API** (una vez que el servidor esté corriendo):
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi4-fast",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7
  }'
```

¡El sistema ARM-Axion con vLLM y los 5 modelos está completamente operativo!