# VM SERVICES - ESTADO ACTUAL

## Descripción
VM dedicada a servicios auxiliares: **TTS, MCP y N8n**
- **Nombre**: gpt-oss-20b
- **IP**: 34.175.136.104
- **Zona**: europe-southwest1-b
- **Arquitectura**: ARM-Axion (C4A-standard-32)

## Servicios Instalados

1. **TTS Kyutai** (puerto 5002)
   - Text-to-Speech con modelo Kyutai Moshi
   - Disponible en /home/elect/capibara6/vm-services/tts/

2. **MCP Server** (puerto 5003)
   - Model Context Protocol
   - Smart MCP v2.0 (detecta cuando agregar contexto y solo lo hace cuando es necesario)
   - Disponible en /home/elect/capibara6/vm-services/mcp/

3. **N8n** (puerto 5678)
   - Workflow automation engine
   - Instalado como servicio systemd
   - Accesible vía túnel SSH o VPN
   - Documentación en /home/elect/capibara6/docs/n8n/

## Scripts Disponibles

### Scripts de prueba interactiva:
- `/home/elect/capibara6/interactive_test_interface.py` - Interfaz completa de prueba
- `/home/elect/capibara6/interactive_test_interface_optimized.py` - Versión optimizada
- `/home/elect/capibara6/interactive_router_test.py` - Pruebas de router
- `/home/elect/capibara6/interactive_arm_axion_test.py` - Pruebas ARM-Axion específicas

### Scripts de inicio de servicios:
- `/home/elect/capibara6/start_vllm_arm_axion.sh` - Inicio del servidor vLLM ARM-Axion
- `/home/elect/capibara6/start_interactive_arm_axion.sh` - Inicio de interfaz interactiva
- `/home/elect/capibara6/scripts/start-all-services.sh` - Iniciar todos los servicios del sistema

## Acceso a Servicios Remotos

Según el archivo `/home/elect/capibara6/check_services_remote.sh`, los servicios en la VM están configurados para acceso remoto:

```bash
# Acceder a la VM de servicios
gcloud compute ssh gpt-oss-20b --zone=europe-southwest1-b --project=mamba-001

# Verificar servicios
curl http://34.175.136.104:5002/health  # TTS
curl http://34.175.136.104:5003/api/mcp/health  # MCP  
curl http://34.175.136.104:5678/healthz  # N8n
```

## Archivos de Documentación

- `/home/elect/capibara6/COMANDOS_VERIFICACION.md` - Guía de verificación del sistema completo
- `/home/elect/capibara6/VLLM_ARM_AXION_IMPLEMENTATION.md` - Documentación de implementación
- `/home/elect/capibara6/IMPLEMENTACION_ARM_AXION_EXITOSA.md` - Confirmación de implementación exitosa
- `/home/elect/capibara6/docs/n8n/` - Documentación específica de n8n

## Estado de Compilación y Despliegue

- **TTS**: Implementado con modelos Kyutai Moshi y Coqui
- **MCP**: Smart MCP v2.0 completamente funcional
- **N8n**: Instalado como servicio con endpoints API configurados
- **ARM-Axion**: Optimizaciones NEON, ACL, cuantización implementadas

## Acceso a N8n

Para acceder a la interfaz de N8n:

```bash
# Crear túnel SSH
ssh -L 5678:localhost:5678 elect@gpt-oss-20b.europe-southwest1-b.c.mamba-001.internal

# Acceder en navegador
open http://localhost:5678
```

## Conclusión

La VM de servicios (gpt-oss-20b) está completamente configurada con los tres servicios requeridos:
1. **TTS Kyutai** - Text-to-Speech
2. **MCP** - Model Context Protocol
3. **N8n** - Workflow Automation

Los scripts de inicio y verificación están disponibles en el sistema principal, y se pueden gestionar los servicios remotos a través de los comandos gcloud como se define en el script `check_services_remote.sh`.