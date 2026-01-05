# Arquitectura del Sistema Capibara6

## VM Bounty2 (esta máquina)

Esta VM aloja el servicio principal de chat para el modelo GPT-OSS-20B.

### Servicios Activos
- `server_gptoss.py` - Servidor del chat con el modelo GPT-OSS-20B (puerto 5001)
- `server_gptoss_CURRENT_WORKING.py` - Copia de seguridad del servidor funcionando
- `config_gptoss.py` - Configuración del servidor y modelo
- `start_gptoss_server.py` - Script de inicio del servidor
- `models_config.py` - Configuración de modelos
- `gpt_oss_optimized_config.py` - Configuración específica para GPT-OSS-20B

### Otros Servidores Disponibles (no activos actualmente)
- `capibara6_integrated_server.py` - Servidor integrado con soporte TOON (en otro directorio)
- Otros servidores especializados para funcionalidades específicas

## VM gpt-oss-20b (otra máquina)

Esta VM aloja otros servicios como:
- TTS (Text-to-Speech)
- MCP (Model Context Propagation)
- Sistema n8n
- Otros microservicios

## VM rag3 (otra máquina)

Esta VM aloja el sistema completo de RAG (Retrieval-Augmented Generation)

## Integración Futura

Los servicios de ambas VMs están destinados a integrarse para proporcionar una experiencia unificada.