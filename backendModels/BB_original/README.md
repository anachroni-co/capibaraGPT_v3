# Capibara6 - Plataforma de Inteligencia Artificial Distribuida

Proyecto de plataforma de IA avanzada con múltiples modelos, sistema de consenso y contexto inteligente.

## 🏗️ Arquitectura del Sistema

### 🖥️ Infraestructura en Google Cloud

**VM 1 - Modelos AI:**
- IP: `34.175.215.109:8080` y `34.175.104.187:8080`
- Modelos: GPT-OSS-20B, Gemma3-12B
- Servidor: llama-server

**VM 2 - Backend (Servidores):**
- IP: `34.175.215.109`
- Servidores backend alojados

## 🤖 Modelos de IA Configurados

1. **capibara6** (Gemma3-12B) - `http://34.175.104.187:8080`
2. **oss-120b** (OSS-120B en TPU) - `http://tpu-server:8080` 
3. **gpt_oss_20b** (GPT-OSS-20B) - `http://34.175.215.109:8080`

## 🔧 Servidores Backend

| Servidor | Archivo | Puerto | Estado | Función |
|----------|---------|--------|--------|---------|
| Servidor Integrado | `backend/capibara6_integrated_server.py` | 5001 | ✅ | Proxy principal + MCP + TTS + **TOON optimization** |
| Servidor Consensus | `backend/consensus_server.py` | 5003 | ✅ | Sistema de consenso multi-modelo (corregido de 5002) + **TOON optimization** |
| Servidor TTS | `backend/coqui_tts_server.py` | 5004 | ✅ | Síntesis de voz Coqui (corregido de 5002) |
| Smart MCP | `backend/smart_mcp_server.py` | 5010 | ✅ | Contexto inteligente standalone + **TOON optimization** |

## ⚡ Optimización de Tokens con TOON

El proyecto ahora incluye soporte para **Token-Oriented Object Notation (TOON)**, un formato que reduce significativamente el uso de tokens cuando se comunican datos estructurados con los modelos de IA:

- **Eficiencia**: Reduce 30-60% el uso de tokens para datos tabulares
- **Soporte en puntos críticos**: Consensus Server, Smart MCP Server, Integrated Server
- **Compatibilidad**: Totalmente compatible con JSON existente
- **Detección automática**: El sistema decide cuándo usar TOON vs JSON
- **Formato de intercambio**: Soporta negociación de contenido con headers `Accept` y `Content-Type`

### Uso de TOON en la API

- **Petición en TOON**: `Content-Type: application/toon` o `text/plain`
- **Respuesta en TOON**: `Accept: application/toon` o `text/plain`
- **Detección automática**: El sistema usa TOON cuando es más eficiente

### Ejemplo de conversión JSON ↔ TOON

**JSON**:
```json
{
  "users": [
    { "id": 1, "name": "Alice", "role": "admin" },
    { "id": 2, "name": "Bob", "role": "user" }
  ]
}
```

**TOON equivalente**:
```
users[2]{id,name,role}:
  1,Alice,admin
  2,Bob,user
```

## 🌐 Frontend (Vercel)

- Carpeta: `web/`
- Archivos principales: `chat.html`, `index.html`
- Scripts: `chatbot.js`, `script.js`, `neural-animation.js`

## 🔄 API Proxies (Vercel Functions)

```
api/
├── chat.js              → Proxy a VM:5001/api/chat
├── consensus/query.js   → Proxy a VM:5003/api/consensus/query
├── mcp/analyze.js       → Proxy a VM:5010/api/mcp/analyze
└── tts/speak.js         → Proxy a VM:5004/api/tts/speak
```

## 🚀 Estructura de Archivos

```
BB/
├── backend/                 # Servidores backend
│   ├── models_config.py     # Configuración de modelos
│   ├── toon_utils/          # Utilidades para formato TOON
│   │   ├── toon_converter.py # Conversor JSON ↔ TOON
│   │   └── format_manager.py # Gestor automático de formatos
│   ├── capibara6_integrated_server.py  # Servidor principal
│   ├── consensus_server.py  # Servidor de consenso
│   ├── coqui_tts_server.py  # Servidor TTS
│   └── smart_mcp_server.py  # Servidor MCP inteligente
├── web/                     # Frontend estático
│   ├── index.html           # Página principal
│   ├── chat.html            # Interfaz de chat
│   ├── style.css            # Estilos
│   ├── script.js            # Funciones generales
│   ├── chatbot.js           # Lógica del chatbot
│   └── neural-animation.js  # Animación de red neuronal
├── api/                     # API functions para Vercel
│   ├── chat.js
│   ├── consensus/
│   ├── mcp/
│   └── tts/
├── requirements.txt         # Dependencias Python
└── Dockerfile              # Contenedor para backend
```

## 📋 Problemas Resueltos

1. ✅ **Conflicto de Puerto 5002**: 
   - `consensus_server.py` ahora en puerto 5003
   - `coqui_tts_server.py` ahora en puerto 5004

2. ✅ **Modelos configurados**:
   - Configuración correcta de capibara6, oss-120b y gpt_oss_20b

3. ✅ **Duplicación MCP**:
   - Ambos servidores MCP documentados (integrado y standalone)
   - API proxy dirigido al standalone para separación de responsabilidades

4. ✅ **Optimización de Tokens**:
   - Implementado soporte para TOON en puntos críticos
   - Reducción significativa de uso de tokens
   - Compatibilidad completa con JSON existente

## 🚀 Despliegue

### Backend (VM 2)
```bash
# Instalar dependencias
pip install -r requirements.txt

# Iniciar servidores
python backend/capibara6_integrated_server.py  # Puerto 5001
python backend/consensus_server.py             # Puerto 5003
python backend/coqui_tts_server.py             # Puerto 5004
python backend/smart_mcp_server.py             # Puerto 5010
```

### Frontend (Vercel)
Desplegar la carpeta `web/` y las funciones API en Vercel.

## 🔐 Configuración de Seguridad

Los endpoints están configurados para aceptar conexiones desde los servidores designados. Asegúrate de configurar correctamente los firewalls y permisos de red en Google Cloud.