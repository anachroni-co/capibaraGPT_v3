# 🎯 Plan de Acción Prioritario - Capibara6

> **Fecha:** 2025-11-24  
> **Objetivo:** Validar migración vLLM y estabilizar el sistema  
> **Tiempo estimado:** 2-3 días

---

## 📋 Resumen de Prioridades

```
🔴 CRÍTICO (Hoy)          → Validar vLLM y corregir errores
🟡 IMPORTANTE (Esta semana) → Consolidar documentación y tests
🟢 MEJORAS (Este mes)      → Roadmap y optimizaciones
```

---

## 🔴 PRIORIDAD CRÍTICA (Hoy - 4 horas)

### ✅ Tarea 1: Validar Endpoints vLLM (30 min)

**Objetivo:** Confirmar que todos los modelos vLLM están respondiendo

**Pasos:**
1. Ejecutar comandos de verificación (ver [COMANDOS_VERIFICACION.md](COMANDOS_VERIFICACION.md))
2. Documentar resultados en checklist
3. Identificar modelos que no responden

**Comandos:**
```bash
# Copiar y ejecutar desde COMANDOS_VERIFICACION.md sección "Verificación Rápida"
curl http://34.12.166.76:8000/v1/models  # gpt-oss-20b
curl http://34.12.166.76:8001/v1/models  # phi4
curl http://34.12.166.76:8002/v1/models  # qwen2.5-coder
curl http://34.12.166.76:8003/v1/models  # mixtral
```

**Criterio de éxito:** ✅ Todos los modelos responden con lista de modelos

---

### ✅ Tarea 2: Probar Generación de Texto (45 min)

**Objetivo:** Validar que la generación de texto funciona con cada modelo

**Pasos:**
1. Ejecutar tests de funcionalidad (ver [COMANDOS_VERIFICACION.md](COMANDOS_VERIFICACION.md))
2. Probar con y sin streaming
3. Verificar latencias

**Tests mínimos:**
- Test con GPT-OSS-20B (complejo)
- Test con Phi4 (rápido)
- Test con Qwen2.5-coder (código)
- Test con streaming activado

**Criterio de éxito:** ✅ Todos los modelos generan texto correctamente

---

### ✅ Tarea 3: Validar Integración Frontend-Backend (1 hora)

**Objetivo:** Confirmar que el frontend puede comunicarse con vLLM a través del backend

**Pasos:**
1. Abrir frontend en navegador: https://www.capibara6.com
2. Enviar mensaje de prueba en el chat
3. Verificar que la respuesta llega correctamente
4. Probar con diferentes modelos (si hay selector)
5. Verificar streaming en tiempo real

**Casos de prueba:**
- ✅ Mensaje simple: "Hola, ¿cómo estás?"
- ✅ Pregunta técnica: "Explica qué es un API REST"
- ✅ Código: "Escribe una función Python para ordenar una lista"
- ✅ Streaming: Pregunta larga para ver tokens en tiempo real

**Criterio de éxito:** ✅ Chat funciona end-to-end con streaming

---

### ✅ Tarea 4: Actualizar Referencias Ollama → vLLM (1.5 horas)

**Objetivo:** Renombrar archivos y actualizar referencias

**Archivos a modificar:**

1. **Renombrar archivos:**
   ```bash
   cd backend/
   mv ollama_client.py vllm_client.py
   mv ollama_rag_integration.py vllm_rag_integration.py
   ```

2. **Actualizar imports en archivos que usan estos módulos:**
   ```bash
   # Buscar referencias
   grep -r "ollama_client" backend/
   grep -r "ollama_rag" backend/
   
   # Actualizar cada archivo encontrado
   # Cambiar: from ollama_client import ...
   # Por:     from vllm_client import ...
   ```

3. **Archivos probables a actualizar:**
   - `backend/api_server.py`
   - `backend/capibara6_integrated_server.py`
   - `backend/semantic_model_router.py`
   - `vm-bounty2/servers/server_gptoss.py`
   - `vm-bounty2/servers/consensus_server.py`

**Criterio de éxito:** ✅ No hay referencias a "ollama" en código activo

---

### ✅ Tarea 5: Verificar Servicios Críticos (30 min)

**Objetivo:** Confirmar que servicios esenciales están operativos

**Servicios a verificar:**
- ✅ Backend principal (5001)
- ✅ TTS Kyutai (5002)
- ✅ Auth server (5004)
- ✅ Sistema RAG (10.154.0.2:8000)
- ✅ Monitorización (Grafana, Prometheus)

**Comandos:**
```bash
# Ver COMANDOS_VERIFICACION.md sección "Verificar Servicios Backend"
curl http://34.12.166.76:5001/health
curl http://34.175.136.104:5002/health
curl http://34.12.166.76:5004/health
```

**Criterio de éxito:** ✅ Todos los servicios críticos responden "healthy"

---

### ✅ Tarea 6: Documentar Hallazgos (30 min)

**Objetivo:** Registrar el estado actual después de las verificaciones

**Pasos:**
1. Completar checklist en [COMANDOS_VERIFICACION.md](COMANDOS_VERIFICACION.md)
2. Actualizar [RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md) con estado real
3. Crear lista de problemas encontrados
4. Priorizar correcciones necesarias

**Criterio de éxito:** ✅ Documentación actualizada con estado real

---

## 🟡 PRIORIDAD IMPORTANTE (Esta semana - 12 horas)

### 📁 Tarea 7: Consolidar Documentación (4 horas)

**Objetivo:** Organizar los 120+ archivos .md en estructura clara

**Estructura propuesta:**
```
docs/
├── architecture/
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── VM_ARCHITECTURE_CONFIG.md
│   └── MODELS_REFERENCE.md
│
├── deployment/
│   ├── DEPLOYMENT_GUIDE.md
│   ├── SERVICES_SETUP.md
│   └── VM_SETUP_GUIDE.md
│
├── operations/
│   ├── CORE_OPERATIONS.md
│   ├── COMANDOS_VERIFICACION.md
│   └── START_SERVICES_VM.md
│
├── troubleshooting/
│   ├── SOLUCIÓN_ERRORES_404.md
│   ├── SOLUCION_CORS.md
│   └── FIX_CONNECTION_ISSUE.md
│
├── integrations/
│   ├── E2B_INTEGRATION_SUMMARY.md
│   ├── KYUTAI_TTS_INTEGRATION.md
│   ├── N8N_INTEGRATION_SUMMARY.md
│   └── TOON_RAG_INTEGRATION.md
│
└── vm-guides/
    ├── VM_BOUNTY2_GUIDE.md
    ├── VM_SERVICES_GUIDE.md
    └── VM_RAG3_GUIDE.md
```

**Pasos:**
1. Crear estructura de carpetas
2. Mover archivos a categorías apropiadas
3. Crear índice principal en `docs/README.md`
4. Actualizar enlaces en README principal
5. Eliminar duplicados

**Criterio de éxito:** ✅ Documentación organizada y accesible

---

### 🧪 Tarea 8: Crear Suite de Tests Básica (6 horas)

**Objetivo:** Implementar tests unitarios e integración

**Estructura:**
```
tests/
├── unit/
│   ├── test_vllm_client.py
│   ├── test_semantic_router.py
│   └── test_task_classifier.py
│
├── integration/
│   ├── test_backend_vllm.py
│   ├── test_rag_integration.py
│   └── test_tts_integration.py
│
└── e2e/
    ├── test_chat_flow.py
    └── test_streaming.py
```

**Tests prioritarios:**

1. **test_vllm_client.py** (1 hora)
   - Test de conexión a cada modelo
   - Test de generación simple
   - Test de streaming
   - Test de manejo de errores

2. **test_backend_vllm.py** (1.5 horas)
   - Test de endpoints API v1
   - Test de clasificación de tareas
   - Test de router semántico
   - Test de fallback

3. **test_rag_integration.py** (1.5 horas)
   - Test de búsqueda en Milvus
   - Test de queries en Nebula Graph
   - Test de Bridge API
   - Test de TOON optimization

4. **test_chat_flow.py** (2 horas)
   - Test end-to-end del chat
   - Test de multiidioma
   - Test de historial
   - Test de TTS

**Criterio de éxito:** ✅ Suite de tests ejecutándose con >80% de cobertura en componentes críticos

---

### 🔧 Tarea 9: Evaluar Servicios Deshabilitados (2 horas)

**Objetivo:** Decidir qué hacer con MCP, Consensus y Smart MCP

**Servicios a evaluar:**

1. **MCP Server (puerto 5003)** - Deshabilitado
   - ¿Por qué está deshabilitado?
   - ¿Es necesario para el sistema?
   - ¿Qué funcionalidad se pierde?
   - Decisión: Reactivar / Documentar / Eliminar

2. **Consensus Server (puerto 5005)** - Deshabilitado
   - Sistema multi-modelo configurado pero no activo
   - ¿Cuándo debería usarse?
   - ¿Cómo activarlo?
   - Decisión: Reactivar / Documentar uso

3. **Smart MCP (puerto 5010)** - No activo
   - Alternativa simplificada a MCP
   - ¿Es redundante?
   - Decisión: Reactivar / Eliminar

**Pasos:**
1. Revisar código de cada servicio
2. Revisar documentación existente
3. Probar activación en entorno de desarrollo
4. Documentar decisión y razones
5. Actualizar config.js con decisiones

**Criterio de éxito:** ✅ Decisión documentada para cada servicio

---

## 🟢 MEJORAS (Este mes - 20 horas)

### 🎨 Tarea 10: Completar Fase 2 del Roadmap (3 horas)

**Objetivo:** Terminar visualización de modelos (62.5% pendiente)

**Tareas pendientes:**
- [ ] Selector manual de modelo (20 min)
- [ ] Visualización de clasificación de tarea (25 min)
- [ ] Indicador de consenso multi-modelo (30 min)
- [ ] Gráfico de uso de modelos (35 min)
- [ ] Integración con backend (20 min)

**Referencia:** Ver [TODO.md](TODO.md) Fase 2

---

### 🖥️ Tarea 11: Iniciar Fase 3 - Panel E2B (5.5 horas)

**Objetivo:** Implementar visualización de sandboxes E2B

**Prioridad alta:**
- Reactivar módulo E2B desde archived
- Crear endpoints API para E2B
- Diseñar componente de terminal integrado
- Implementar panel de estado de sandbox

**Referencia:** Ver [TODO.md](TODO.md) Fase 3

---

### 🧹 Tarea 12: Limpieza de Código (4 horas)

**Objetivo:** Eliminar código duplicado y archivos temporales

**Acciones:**
1. Eliminar archivos `.backup` (30 min)
2. Consolidar scripts de verificación (1 hora)
3. Unificar `backend/` y `vm-bounty2/` (2 horas)
4. Eliminar código comentado (30 min)

---

### 📊 Tarea 13: Optimizar Monitorización (3 horas)

**Objetivo:** Mejorar dashboards y alertas

**Acciones:**
1. Revisar 18 dashboards de Grafana
2. Ajustar alertas de Prometheus
3. Configurar alertas para vLLM
4. Documentar métricas clave

---

### 🔐 Tarea 14: Auditoría de Seguridad (4.5 horas)

**Objetivo:** Verificar configuración de seguridad

**Áreas:**
1. Firewall rules (1 hora)
2. Autenticación OAuth (1 hora)
3. API keys y secrets (1 hora)
4. HTTPS/SSL (1 hora)
5. Documentar hallazgos (30 min)

---

## 📊 Cronograma Sugerido

### Día 1 (Hoy) - 4 horas
```
09:00 - 09:30  ✅ Tarea 1: Validar endpoints vLLM
09:30 - 10:15  ✅ Tarea 2: Probar generación de texto
10:15 - 10:30  ☕ Break
10:30 - 11:30  ✅ Tarea 3: Validar integración frontend-backend
11:30 - 13:00  ✅ Tarea 4: Actualizar referencias Ollama → vLLM
13:00 - 14:00  🍽️ Almuerzo
14:00 - 14:30  ✅ Tarea 5: Verificar servicios críticos
14:30 - 15:00  ✅ Tarea 6: Documentar hallazgos
```

### Día 2 - 6 horas
```
09:00 - 13:00  📁 Tarea 7: Consolidar documentación (4h)
13:00 - 14:00  🍽️ Almuerzo
14:00 - 16:00  🔧 Tarea 9: Evaluar servicios deshabilitados (2h)
```

### Día 3 - 6 horas
```
09:00 - 15:00  🧪 Tarea 8: Crear suite de tests (6h)
```

### Semana 2-4 - 20 horas
```
Distribuir tareas 10-14 según prioridad
```

---

## ✅ Checklist de Progreso

### Día 1 (Crítico)
- [ ] Tarea 1: Validar endpoints vLLM
- [ ] Tarea 2: Probar generación de texto
- [ ] Tarea 3: Validar integración frontend-backend
- [ ] Tarea 4: Actualizar referencias Ollama → vLLM
- [ ] Tarea 5: Verificar servicios críticos
- [ ] Tarea 6: Documentar hallazgos

### Semana 1 (Importante)
- [ ] Tarea 7: Consolidar documentación
- [ ] Tarea 8: Crear suite de tests
- [ ] Tarea 9: Evaluar servicios deshabilitados

### Mes 1 (Mejoras)
- [ ] Tarea 10: Completar Fase 2 roadmap
- [ ] Tarea 11: Iniciar Fase 3 - Panel E2B
- [ ] Tarea 12: Limpieza de código
- [ ] Tarea 13: Optimizar monitorización
- [ ] Tarea 14: Auditoría de seguridad

---

## 📝 Registro de Ejecución

**Fecha de inicio:** _____________

**Responsable:** _____________

### Día 1
- [ ] Completado
- Problemas encontrados: _____________
- Tiempo real: _____________

### Día 2
- [ ] Completado
- Problemas encontrados: _____________
- Tiempo real: _____________

### Día 3
- [ ] Completado
- Problemas encontrados: _____________
- Tiempo real: _____________

---

## 🔗 Referencias

- [RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md) - Estado actual
- [ANALISIS_ESTADO_ACTUAL.md](ANALISIS_ESTADO_ACTUAL.md) - Análisis completo
- [COMANDOS_VERIFICACION.md](COMANDOS_VERIFICACION.md) - Comandos de verificación
- [TODO.md](TODO.md) - Roadmap completo
- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Arquitectura
- [CORE_OPERATIONS.md](CORE_OPERATIONS.md) - Operaciones

---

**Documento creado:** 2025-11-24  
**Próxima revisión:** Después del Día 1
