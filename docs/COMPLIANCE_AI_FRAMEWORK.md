# Capibara Legal — Marco de Cumplimiento Normativo IA

**Referencia principal**: Convenio Marco del Consejo de Europa sobre Inteligencia Artificial
y Derechos Humanos, Democracia y Estado de Derecho — CM(2024)52-final, mayo 2024.
**Ámbito**: Tratado internacional vinculante para España y todos los Estados miembros del
Consejo de Europa. Aplicable a sistemas de IA en el sector privado que puedan afectar
derechos humanos.

---

## Por qué aplica a Capibara Legal

Capibara Legal ofrece orientación jurídica a ciudadanos y profesionales en jurisdicciones
del Consejo de Europa (España, latinoamérica hispanohablante con conexión CoE). Al
proporcionar análisis legal que puede influir en decisiones con consecuencias reales para
los usuarios, el sistema está comprendido en el ámbito de aplicación del Convenio.

No es un requisito optativo: España ratifica el tratado como obligación internacional.
Los sistemas de IA en producción deben cumplir antes del despliegue.

---

## Obligaciones por artículo — checklist de implementación

### Art. 6 — Transparencia (sistema)

**Obligación**: Los usuarios deben saber que interactúan con un sistema de IA y cuáles son
sus capacidades y limitaciones.

| Requisito | Implementación en Capibara | Estado |
|-----------|---------------------------|--------|
| Identificación como IA | Mensaje inicial explícito en cada sesión | [ ] Pendiente |
| Advertencia de limitaciones | Disclaimer en respuestas sobre materia legal | [ ] Pendiente |
| No fingir ser humano | Prohibido en configuración del sistema prompt | [x] Garantizado por diseño |
| Revelar naturaleza si se pregunta | Obligatorio en system prompt | [ ] Pendiente |

**Nota sobre think tags**: El stripping de `<think>...</think>` del output final
**no viola** el Art. 6 — el razonamiento interno no es "ocultamiento de naturaleza IA".
Sí sería violación usarlo para simular razonamiento humano al usuario.

```python
# Disclaimer mínimo a añadir en speculative_inference.py
SYSTEM_DISCLAIMER = (
    "Soy Capibara Legal, un asistente de inteligencia artificial. "
    "Mis respuestas son orientativas y no constituyen asesoramiento jurídico "
    "profesional. Consulta siempre con un abogado para decisiones con "
    "consecuencias legales concretas."
)
```

### Art. 8 — Transparencia (decisiones individuales)

**Obligación**: Cuando el sistema tome una decisión que afecte significativamente a una
persona, esta debe poder obtener explicación.

| Requisito | Implementación | Estado |
|-----------|----------------|--------|
| Explicabilidad de respuestas | RAG cita las fuentes normativas (BOE, CENDOJ) | [ ] Parcial |
| Fuentes citadas en respuestas legales | Formato `[Art. X, Ley Y]` en output | [ ] Pendiente |
| Log de qué datos se usaron | Registro de chunks RAG por sesión | [ ] Pendiente |

**Acción técnica**: Añadir a `rag_retriever.py` un campo `sources` en el output que
acompañe siempre a la respuesta, y formatear automáticamente la cita en el prompt.

### Art. 9 — Responsabilidad (accountability)

**Obligación**: Debe existir un responsable identificable de los impactos del sistema.

| Requisito | Implementación | Estado |
|-----------|----------------|--------|
| Titular responsable identificado | Persona física/jurídica definida | [ ] Definir |
| Política de uso aceptable | ToS con prohibiciones explícitas | [ ] Redactar |
| Registro de incidentes | Log estructurado de fallos y reclamaciones | [ ] Pendiente |
| Evaluación de impacto (DPIA) | Análisis de riesgos antes de producción | [ ] Pendiente |

**Nota**: No es suficiente "la IA decide" como exención de responsabilidad.
El operador del sistema es responsable de los outputs incorrectos o dañinos.

### Art. 11 — Privacidad y protección de datos

**Obligación**: Los datos personales procesados por el sistema deben tratarse conforme
al Convenio 108+ del CoE y el RGPD. Aplica directamente a la Mejora 5 de V4
(memoria persistente por usuario).

| Requisito | Implementación V3/V4 | Estado |
|-----------|---------------------|--------|
| Minimización de datos | Solo almacenar lo necesario para la sesión | [ ] Diseñar |
| Plazo de retención | Memoria V4 FAISS: máx. 90 días o hasta revocación | [ ] Definir |
| Derecho de supresión | API para borrar FAISS index de usuario | [ ] Pendiente |
| Derecho de acceso | Endpoint para exportar datos del usuario | [ ] Pendiente |
| Seudonimización | user_id hash, no datos identificativos en FAISS | [ ] Pendiente |
| Consentimiento explícito | Opt-in para memoria persistente (no por defecto) | [ ] Diseñar |

```python
# En persistent_memory.py (V4) — diseño mínimo de privacidad
class PersistentMemory:
    def __init__(self, user_id: str, retention_days: int = 90):
        self.user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        self.index_path = f"faiss/{self.user_hash}.index"
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(days=retention_days)

    def delete_user_data(self):
        """Derecho de supresión — RGPD Art. 17."""
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
```

### Art. 12 — Fiabilidad y seguridad

**Obligación**: Los sistemas de IA deben funcionar de manera fiable y segura,
incluyendo protecciones contra fallos y outputs dañinos.

| Requisito | Implementación | Estado |
|-----------|----------------|--------|
| Mitigación de alucinaciones | RAG con fuentes verificadas (BOE, CENDOJ) | [x] Diseñado |
| Dominio restringido | LoRA adapters especializados + system prompt | [x] Diseñado |
| Detección de consultas fuera de dominio | Clasificador de intención | [ ] Pendiente |
| No dar consejos en materia penal grave sin advertencia | Disclaimer específico | [ ] Pendiente |
| Pruebas antes de producción | Test suite con casos de referencia jurídica | [ ] Pendiente |

**Benchmark de fiabilidad mínimo antes de producción**:
```
- 50 casos de test con respuesta correcta conocida (resoluciones BOE/CENDOJ)
- Tasa de hallucination < 5% (respuesta cita norma inexistente o derogada)
- Tasa de respuesta fuera de dominio identificada > 90%
- Disclaimer mostrado en 100% de consultas legales con consecuencias reales
```

### Art. 13 — Innovación segura (sandboxing)

**Obligación**: Antes de desplegar en producción, los sistemas deben probarse en
entornos controlados.

| Fase | Entorno | Métricas de salida |
|------|---------|--------------------|
| V2 Small/Medium | Solo local, sin usuarios reales | Perplexity, benchmark interno |
| V2 Large beta | Beta cerrada ~10 usuarios | Feedback cualitativo, fallos |
| V3 desarrollo | Entorno aislado, sin datos de producción | Benchmarks V1→V3 |
| V3 producción | Phased rollout: 1% → 10% → 100% | Tasa de error, satisfacción |

### Art. 14 — Recursos y reparación

**Obligación**: Los usuarios deben tener mecanismos de recurso si el sistema les
causa un perjuicio.

| Requisito | Implementación | Estado |
|-----------|----------------|--------|
| Canal de reclamaciones | Email/formulario visible en UI | [ ] Pendiente |
| Tiempo de respuesta máximo | 30 días hábiles (alineado con RGPD) | [ ] Definir |
| Escalado a humano | Proceso para casos con impacto real | [ ] Definir |
| Registro de reclamaciones | Log estructurado con resolución | [ ] Pendiente |

### Art. 15 — Supervisión independiente

**Obligación**: Los sistemas de IA de alto riesgo deben estar sujetos a supervisión
independiente. La orientación legal automatizada puede clasificarse como alto riesgo.

| Requisito | Implementación | Estado |
|-----------|----------------|--------|
| Audit log de decisiones | Log con input, output, sources por sesión | [ ] Pendiente |
| Revisión periódica | Auditoría de outputs cada 6 meses | [ ] Planificar |
| Informe de transparencia | Publicación anual de métricas de uso y fallos | [ ] Planificar |

```python
# Estructura mínima del audit log
AUDIT_ENTRY = {
    "session_id": str,      # UUID de sesión, no identifica al usuario
    "timestamp": str,       # ISO 8601
    "query_hash": str,      # SHA256 del query, no el texto en claro
    "response_length": int, # longitud de respuesta
    "sources_cited": list,  # lista de fuentes RAG usadas
    "domain": str,          # subdominio legal detectado
    "disclaimer_shown": bool,
}
```

---

## Clasificación de riesgo del sistema

El Convenio distingue niveles de riesgo. Para Capibara Legal:

| Escenario de uso | Nivel de riesgo | Justificación |
|------------------|-----------------|---------------|
| Información legal general ("¿cuál es el plazo de prescripción?") | Bajo | No afecta directamente a derechos |
| Análisis de un caso concreto del usuario | **Alto** | Puede influir en decisiones con consecuencias reales |
| Redacción de documentos legales | **Alto** | Un error puede invalidar el documento |
| Uso por abogados para research | Medio | Profesional verifica antes de usar |
| Uso por ciudadanos sin asesoramiento adicional | **Alto** | Riesgo de dependencia sin verificación |

Para escenarios de riesgo alto: todos los artículos del Convenio aplican en su
versión más exigente. El sistema debe identificarlo y reforzar los disclaimers.

---

## Plan de implementación — antes de cada versión

### Antes de V2 en producción

- [ ] Añadir `SYSTEM_DISCLAIMER` en `speculative_inference.py`
- [ ] Añadir cita de fuentes RAG en respuestas (`[Fuente: BOE 2023/456]`)
- [ ] Redactar ToS mínimo con prohibiciones de uso
- [ ] Canal de reclamaciones operativo (email)
- [ ] Benchmark de fiabilidad: 50 casos de test

### Antes de V3 en producción

- [ ] Audit log implementado (`training/audit_logger.py`)
- [ ] Clasificador de riesgo por tipo de consulta
- [ ] DPIA (Data Protection Impact Assessment) completado
- [ ] Phased rollout plan documentado

### Antes de V4 en producción (memoria persistente)

- [ ] `PersistentMemory` con hash de usuario y expiración automática
- [ ] Endpoint de supresión de datos implementado y probado
- [ ] Consentimiento explícito para memoria: opt-in con información clara
- [ ] Auditoría de seguridad del almacenamiento FAISS

---

## Referencia normativa complementaria

| Norma | Relación con Capibara |
|-------|-----------------------|
| RGPD (Reglamento UE 2016/679) | Aplica en España — compatible con Art. 11 del Convenio |
| Ley Orgánica 3/2018 (LOPDGDD) | Implementación española del RGPD |
| Propuesta Reglamento IA UE (AI Act) | Capas adicionales para sistemas alto riesgo |
| Convenio 108+ del CoE | Marco de datos personales base del Art. 11 |
| AEPD (Agencia Española Protección de Datos) | Autoridad supervisora en España |

El Convenio Marco CoE es compatible con el AI Act UE — donde ambos aplican,
se elige el estándar más exigente.
