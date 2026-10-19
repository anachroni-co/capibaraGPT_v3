#!/usr/bin/env python3
"""Generate synthetic tool-use training data for the Capibara 'herramientas' LoRA adapter.

Each example teaches the model the PATTERN of tool invocation using the 0xFF (ÿ)
delimiter protocol of the capibaraGPT byte-level tokeniser (vocab=512):
    ÿTOOL:{"name":"<tool>", ...params...}ÿ      ← model emits this
    ÿRESULT:{"status":"ok","data":...}ÿ          ← injected result
    <natural-language answer>                     ← model continues

All tool-call examples use plausible-looking placeholder data — the model
learns the calling convention, not specific API results.

Output format (JSONL, one example per line, same schema as lora_finetune.py):
    {"prompt": "¿Qué dice...", "response": "ÿTOOL:{...}ÿ\\nÿRESULT:{...}ÿ\\n<answer>"}

Tools covered (~200 examples each, 1000 total):
    search_boe       — BOE open-data search (current legislation, reforms)
    search_cendoj    — CENDOJ jurisprudence (TS sentencias)
    search_web       — general web search (news, academic sources)
    get_article      — fetch exact article text from a consolidated law
    calculate_plazo  — compute procedural deadlines (días hábiles/naturales)

Example types per tool:
    1. Simple: single tool call → direct answer
    2. Multi:  two sequential tool calls (search_boe → get_article, etc.)
    3. Not-found: tool returns status "not_found" → polite "no encontré" reply

Usage
-----
    # Generate default 1000 examples
    python scripts/download_tool_data.py --output data/finetune/herramientas.jsonl

    # Custom count
    python scripts/download_tool_data.py --output data/finetune/herramientas.jsonl --count 1000

    # Then fine-tune the herramientas adapter
    python scripts/lora_finetune.py \\
        --base-ckpt checkpoints/axion_large_legal/soup_uniform.pkl \\
        --preset    large \\
        --data      data/finetune/herramientas.jsonl \\
        --specialty all \\
        --output    checkpoints/lora/large_herramientas \\
        --steps 3000 --rank 16 --lora-alpha 32
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("herramientas")

# ─── Protocol helpers ─────────────────────────────────────────────────────────

DELIM = "\xff"  # ÿ — byte 0xFF, delimiter in the capibaraGPT protocol


def _tool_call(name: str, **params) -> str:
    """Format a ÿTOOL:{...}ÿ call string."""
    payload = {"name": name, **params}
    return DELIM + "TOOL:" + json.dumps(payload, ensure_ascii=False) + DELIM


def _result(status: str, data) -> str:
    """Format a ÿRESULT:{...}ÿ string."""
    payload = {"status": status, "data": data}
    return DELIM + "RESULT:" + json.dumps(payload, ensure_ascii=False) + DELIM


def _not_found_result(msg: str = "No se encontraron resultados.") -> str:
    return DELIM + "RESULT:" + json.dumps({"status": "not_found", "data": msg}, ensure_ascii=False) + DELIM


# ─── Placeholder data pools ───────────────────────────────────────────────────

_BOE_RESULTS_POOL = [
    [
        {"id": "BOE-A-2023-15891", "titulo": "Real Decreto 5/2023, de 28 de junio, por el que se adoptan medidas urgentes...",
         "fecha": "2023-06-29", "url": "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2023-15891",
         "resumen": "Se regulan los plazos de prescripción en materia de responsabilidad contractual conforme a la reforma del Código Civil..."},
        {"id": "BOE-A-2022-8734", "titulo": "Ley 8/2022, de 14 de junio, de modificación del Código Penal en materia de delitos contra la libertad sexual",
         "fecha": "2022-06-15", "url": "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2022-8734",
         "resumen": "Se modifican los artículos 178 a 184 del Código Penal para adaptar la legislación al Convenio de Estambul..."},
    ],
    [
        {"id": "BOE-A-2021-3564", "titulo": "Ley 3/2021, de 28 de enero, por la que se regulan los fondos de inversión colectiva",
         "fecha": "2021-01-29", "url": "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2021-3564",
         "resumen": "La presente ley establece el régimen jurídico aplicable a los fondos de inversión en el mercado financiero español..."},
    ],
    [
        {"id": "BOE-A-2024-1122", "titulo": "Real Decreto-ley 2/2024, de 21 de enero, de medidas urgentes en materia de vivienda",
         "fecha": "2024-01-22", "url": "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2024-1122",
         "resumen": "Se adoptan medidas de contención de rentas de alquiler y ampliación de los derechos de los arrendatarios..."},
        {"id": "BOE-A-2023-22450", "titulo": "Ley 12/2023, de 24 de mayo, por el derecho a la vivienda",
         "fecha": "2023-05-25", "url": "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2023-22450",
         "resumen": "Primera ley estatal de vivienda que reconoce el derecho subjetivo a la vivienda digna y adecuada..."},
    ],
]

_CENDOJ_RESULTS_POOL = [
    [
        {"titulo": "STS 1234/2023 - Sala de lo Civil - Responsabilidad civil médica",
         "tribunal": "Tribunal Supremo", "fecha": "2023",
         "resumen": "El Tribunal Supremo establece que la responsabilidad civil médica requiere acreditar la relación de causalidad entre la actuación del profesional y el daño producido al paciente..."},
        {"titulo": "STS 876/2023 - Sala de lo Civil - Daños morales por negligencia médica",
         "tribunal": "Tribunal Supremo", "fecha": "2023",
         "resumen": "Se fija indemnización de 150.000 euros por daños morales derivados de error de diagnóstico con secuelas permanentes..."},
    ],
    [
        {"titulo": "STS 542/2022 - Sala de lo Penal - Delito de estafa",
         "tribunal": "Tribunal Supremo", "fecha": "2022",
         "resumen": "La Sala de lo Penal confirma que el engaño bastante en la estafa debe ser idóneo para inducir a error a una persona de diligencia media..."},
    ],
    [
        {"titulo": "STS 3301/2023 - Sala de lo Contencioso - Responsabilidad patrimonial de la Administración",
         "tribunal": "Tribunal Supremo", "fecha": "2023",
         "resumen": "Se reconoce el derecho a indemnización por funcionamiento anormal del servicio público sanitario con nexo causal acreditado mediante prueba pericial..."},
        {"titulo": "STS 1809/2022 - Sala de lo Contencioso - Silencio administrativo",
         "tribunal": "Tribunal Supremo", "fecha": "2022",
         "resumen": "El silencio administrativo positivo opera automáticamente por ministerio de la ley sin necesidad de resolución expresa..."},
    ],
]

_WEB_RESULTS_POOL = [
    [
        {"titulo": "Sentencia TS sobre daños morales en contratos: análisis doctrinal",
         "url": "https://www.elderecho.com/sentencia-ts-danos-morales-2024",
         "snippet": "El Tribunal Supremo consolida su doctrina sobre la cuantificación de los daños morales en el incumplimiento contractual..."},
        {"titulo": "Nueva jurisprudencia sobre responsabilidad civil extracontractual (2024)",
         "url": "https://www.abogacia.es/jurisprudencia-rc-extracontractual",
         "snippet": "Análisis de las sentencias más relevantes del TS en materia de responsabilidad civil durante el primer semestre de 2024..."},
    ],
    [
        {"titulo": "Estudio sobre el derecho de desistimiento en contratos de consumo | UAM",
         "url": "https://revistas.uam.es/derecho/article/view/12345",
         "snippet": "Análisis académico del régimen jurídico del desistimiento unilateral en contratos celebrados a distancia conforme al TRLGDCU..."},
    ],
    [
        {"titulo": "Últimas novedades legislativas en derecho laboral 2024 | BOE",
         "url": "https://www.boe.es/biblioteca_juridica/abrir_pdf.php?id=PUB-LF-2024-109",
         "snippet": "Recopilación de las principales reformas del Estatuto de los Trabajadores y normativa de Seguridad Social publicadas en el BOE en 2024..."},
    ],
]

_ARTICLE_TEXTS_POOL = {
    ("codigo_penal", "248"): (
        "Artículo 248. Cometen estafa los que, con ánimo de lucro, "
        "utilizaren engaño bastante para producir error en otro, induciéndole "
        "a realizar un acto de disposición en perjuicio propio o ajeno."
    ),
    ("codigo_penal", "197"): (
        "Artículo 197. El que, para descubrir los secretos o vulnerar la "
        "intimidad de otro, sin su consentimiento, se apodere de sus papeles, "
        "cartas, mensajes de correo electrónico o cualesquiera otros documentos "
        "o efectos personales, intercepte sus telecomunicaciones..."
    ),
    ("codigo_penal", "138"): (
        "Artículo 138. El que matare a otro será castigado, como reo de homicidio, "
        "con la pena de prisión de diez a quince años."
    ),
    ("constitucion", "24"): (
        "Artículo 24. 1. Todas las personas tienen derecho a obtener la tutela "
        "efectiva de los jueces y tribunales en el ejercicio de sus derechos e "
        "intereses legítimos, sin que, en ningún caso, pueda producirse indefensión."
    ),
    ("constitucion", "18"): (
        "Artículo 18. 1. Se garantiza el derecho al honor, a la intimidad "
        "personal y familiar y a la propia imagen. 2. El domicilio es inviolable..."
    ),
    ("lec", "400"): (
        "Artículo 400. Preclusión de la alegación de hechos y fundamentos jurídicos. "
        "1. Cuando lo que se pida en la demanda pueda fundarse en diferentes hechos "
        "o en distintos fundamentos o títulos jurídicos, habrán de aducirse en ella "
        "cuantos resulten conocidos o puedan invocarse al tiempo de interponerla..."
    ),
    ("lec", "217"): (
        "Artículo 217. Carga de la prueba. 1. Cuando, al tiempo de dictar sentencia "
        "o resolución semejante, el tribunal considerase dudosos unos hechos relevantes "
        "para la decisión, desestimará las pretensiones del actor o del reconviniente, "
        "o las del demandado o reconvenido, según corresponda a unos u otros la carga "
        "de probar los hechos que permanezcan inciertos..."
    ),
    ("estatuto_trabajadores", "55"): (
        "Artículo 55. Forma y efectos del despido disciplinario. 1. El despido deberá "
        "ser notificado por escrito al trabajador, haciendo figurar los hechos que lo "
        "motivan y la fecha en que tendrá efectos. 5. Será nulo el despido que tenga "
        "por móvil alguna de las causas de discriminación prohibidas en la Constitución..."
    ),
    ("estatuto_trabajadores", "37"): (
        "Artículo 37. Descanso semanal, fiestas y permisos. 1. Los trabajadores "
        "tendrán derecho a un descanso mínimo semanal, acumulable por períodos de "
        "hasta catorce días, de día y medio ininterrumpido que, como regla general, "
        "comprenderá la tarde del sábado o, en su caso, la mañana del lunes y el día completo del domingo..."
    ),
}

_PLAZO_RESULTS_POOL = [
    {"fecha_fin": "2025-03-29", "dias_totales": "20", "tipo": "hábiles",  "detalle": "sábado 29 de marzo de 2025"},
    {"fecha_fin": "2025-04-04", "dias_totales": "34", "tipo": "hábiles",  "detalle": "viernes 4 de abril de 2025"},
    {"fecha_fin": "2025-02-10", "dias_totales": "10", "tipo": "naturales","detalle": "lunes 10 de febrero de 2025"},
    {"fecha_fin": "2025-05-20", "dias_totales": "30", "tipo": "hábiles",  "detalle": "martes 20 de mayo de 2025"},
    {"fecha_fin": "2025-06-15", "dias_totales": "15", "tipo": "naturales","detalle": "domingo 15 de junio de 2025"},
    {"fecha_fin": "2025-08-26", "dias_totales": "40", "tipo": "hábiles",  "detalle": "martes 26 de agosto de 2025"},
    {"fecha_fin": "2025-12-20", "dias_totales": "60", "tipo": "hábiles",  "detalle": "sábado 20 de diciembre de 2025"},
    {"fecha_fin": "2025-09-30", "dias_totales": "30", "tipo": "naturales","detalle": "martes 30 de septiembre de 2025"},
]

# ─── Prompt / answer phrasing pools ───────────────────────────────────────────

# ── search_boe ────────────────────────────────────────────────────────────────
_BOE_TOPICS = [
    ("contrato de arrendamiento urbano", "arrendamiento urbano", "vivienda"),
    ("despido disciplinario improcedente", "despido disciplinario", "laboral"),
    ("responsabilidad civil extracontractual", "responsabilidad civil", "daños"),
    ("delitos contra la hacienda pública", "fraude fiscal", "hacienda pública"),
    ("protección de datos personales", "RGPD", "datos personales"),
    ("ley de transparencia", "transparencia administrativa", "acceso a la información"),
    ("concurso de acreedores", "insolvencia empresarial", "concurso"),
    ("derecho de desistimiento", "desistimiento contratos consumo", "consumidores"),
    ("violencia de género", "violencia doméstica", "ley orgánica género"),
    ("tráfico de drogas", "tráfico de estupefacientes", "narcotráfico"),
    ("privacidad en el trabajo", "control laboral videovigilancia", "videovigilancia"),
    ("derecho de huelga", "huelga servicios esenciales", "huelga"),
]

_BOE_PROMPT_TEMPLATES = [
    "¿Qué dice la ley sobre {topic}?",
    "¿Se ha modificado recientemente la normativa sobre {topic}?",
    "Busca la normativa vigente sobre {topic}.",
    "¿Cuál es la regulación actual de {topic} en España?",
    "Necesito información legal sobre {topic}. ¿Qué dice el BOE?",
    "¿Existe alguna reforma reciente sobre {topic}?",
    "Dame las últimas novedades legislativas sobre {topic}.",
    "¿Cómo regula la legislación española {topic}?",
    "Busca leyes y decretos sobre {topic} en el BOE.",
    "¿Qué normativa regula {topic} en España actualmente?",
]

_BOE_ANSWER_TEMPLATES = [
    "He encontrado {n} resultado(s) en el BOE sobre {topic}:\n\n{items}\n\nLa normativa vigente regula esta materia principalmente a través de {primera_norma}.",
    "Según el BOE, la regulación de {topic} se recoge en las siguientes normas:\n\n{items}\n\nDestaca especialmente {primera_norma}, que establece el marco jurídico básico.",
    "He localizado la siguiente normativa sobre {topic} en el BOE:\n\n{items}",
    "La legislación vigente sobre {topic} incluye:\n\n{items}\n\nTe recomiendo consultar especialmente {primera_norma} para una visión completa de la regulación.",
]


def _fmt_boe_items(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. **{h['titulo']}** ({h['fecha']})\n   {h['resumen'][:200]}")
    return "\n\n".join(lines)


# ── search_cendoj ─────────────────────────────────────────────────────────────
_CENDOJ_TOPICS = [
    ("responsabilidad civil médica", "negligencia médica"),
    ("daños morales en contratos", "daños morales"),
    ("despido nulo por maternidad", "despido discriminatorio"),
    ("prescripción de la acción penal", "prescripción penal"),
    ("responsabilidad patrimonial de la Administración", "responsabilidad administrativa"),
    ("cláusulas abusivas en hipotecas", "cláusulas abusivas hipotecarias"),
    ("delito de estafa mediante engaño", "estafa"),
    ("guarda y custodia compartida", "custodia compartida"),
    ("derecho al olvido en internet", "derecho al olvido"),
    ("accidente de trabajo en subcontrata", "accidente laboral subcontrata"),
    ("contrato de franquicia resolución", "franquicia"),
    ("herencia intestada extranjeros", "herencia internacional"),
]

_CENDOJ_PROMPT_TEMPLATES = [
    "¿Cómo ha fallado el Tribunal Supremo en casos de {topic}?",
    "Jurisprudencia del Tribunal Supremo sobre {topic}.",
    "¿Qué criterio sigue el TS para {topic}?",
    "¿Cuál es la doctrina del Tribunal Supremo sobre {topic}?",
    "Busca sentencias del TS sobre {topic}.",
    "¿Cómo resuelve el Tribunal Supremo los casos de {topic}?",
    "¿Existe jurisprudencia consolidada del TS sobre {topic}?",
    "Necesito conocer el criterio jurisprudencial del TS en materia de {topic}.",
    "¿Qué dice el Tribunal Supremo sobre {topic}?",
    "Dame jurisprudencia reciente del TS sobre {topic}.",
]

_CENDOJ_ANSWER_TEMPLATES = [
    "El Tribunal Supremo ha sentado la siguiente jurisprudencia sobre {topic}:\n\n{items}\n\nEn síntesis, el TS establece que {resumen_breve}.",
    "He encontrado las siguientes sentencias del TS sobre {topic}:\n\n{items}",
    "La doctrina del Tribunal Supremo sobre {topic} puede resumirse así:\n\n{items}\n\nEste criterio jurisprudencial es de aplicación general.",
    "Según la jurisprudencia del TS, en materia de {topic}:\n\n{items}",
]


def _fmt_cendoj_items(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. **{h['titulo']}** ({h['fecha']})\n   {h['resumen'][:200]}")
    return "\n\n".join(lines)


# ── search_web ────────────────────────────────────────────────────────────────
_WEB_QUERIES = [
    ("sentencias recientes del TS sobre indemnización por despido", "despido"),
    ("estudios académicos sobre derecho de la competencia España 2024", "competencia"),
    ("últimas novedades en derecho de familia España 2024", "familia"),
    ("reforma procesal penal España noticias 2024", "procesal penal"),
    ("jurisprudencia reciente menores no acompañados", "menores migrantes"),
    ("novedades LOPD y RGPD España empresas 2024", "protección datos"),
    ("sentencias recientes sobre responsabilidad de redes sociales", "redes sociales"),
    ("estudios sobre eficacia de mediación civil España", "mediación"),
    ("últimas reformas derecho concursal España 2024", "concursal"),
    ("artículos académicos blanqueo de capitales compliance", "blanqueo capitales"),
    ("noticias ley de inteligencia artificial regulación Europa", "IA regulación"),
    ("jurisprudencia reciente acoso laboral mobbing", "acoso laboral"),
]

_WEB_PROMPT_TEMPLATES = [
    "¿Hay sentencias recientes sobre {topic}?",
    "Busca estudios académicos sobre {topic}.",
    "Últimas novedades en {topic}.",
    "¿Qué se ha publicado recientemente sobre {topic}?",
    "Busca información actualizada sobre {topic}.",
    "¿Existen publicaciones académicas recientes sobre {topic}?",
    "Necesito los últimos desarrollos doctrinales sobre {topic}.",
    "¿Qué novedades hay en {topic} en 2024?",
    "Busca noticias jurídicas recientes sobre {topic}.",
    "Dame información actualizada sobre {topic}.",
]

_WEB_ANSWER_TEMPLATES = [
    "He encontrado los siguientes recursos sobre {topic}:\n\n{items}\n\nEn resumen, los últimos desarrollos apuntan a {idea}.",
    "La búsqueda sobre {topic} ha devuelto:\n\n{items}",
    "Sobre {topic}, he localizado los siguientes recursos actualizados:\n\n{items}\n\nTe recomiendo especialmente el primer resultado para una visión completa.",
    "Aquí tienes información reciente sobre {topic}:\n\n{items}",
]


def _fmt_web_items(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. [{h['titulo']}]({h['url']})\n   {h['snippet']}")
    return "\n\n".join(lines)


# ── get_article ───────────────────────────────────────────────────────────────
_ARTICLE_REQUESTS = [
    ("codigo_penal",          "248",  "Código Penal",                     "estafa"),
    ("codigo_penal",          "197",  "Código Penal",                     "descubrimiento de secretos"),
    ("codigo_penal",          "138",  "Código Penal",                     "homicidio"),
    ("constitucion",          "24",   "Constitución Española",            "tutela judicial efectiva"),
    ("constitucion",          "18",   "Constitución Española",            "derecho al honor e intimidad"),
    ("lec",                   "400",  "Ley de Enjuiciamiento Civil",       "preclusión"),
    ("lec",                   "217",  "Ley de Enjuiciamiento Civil",       "carga de la prueba"),
    ("estatuto_trabajadores", "55",   "Estatuto de los Trabajadores",      "despido disciplinario"),
    ("estatuto_trabajadores", "37",   "Estatuto de los Trabajadores",      "descanso semanal y permisos"),
]

_ARTICLE_PROMPT_TEMPLATES = [
    "¿Qué dice exactamente el artículo {art} de {ley_nombre}?",
    "Dame el texto del artículo {art} de {ley_nombre}.",
    "¿Cuál es la redacción actual del artículo {art} de {ley_nombre}?",
    "¿Cuál es el contenido del artículo {art} del {ley_nombre}?",
    "Necesito el texto literal del artículo {art} de {ley_nombre}.",
    "¿Cómo está redactado el artículo {art} de {ley_nombre}?",
    "¿Puedes mostrarme el artículo {art} de {ley_nombre}?",
    "¿Qué establece el artículo {art} de {ley_nombre}?",
    "Texto del artículo {art} de {ley_nombre}.",
    "¿Cuál es el contenido exacto del artículo {art} del {ley_nombre}?",
]

_ARTICLE_ANSWER_TEMPLATES = [
    "El artículo {art} de {ley_nombre} dispone lo siguiente:\n\n{texto}",
    "El texto del artículo {art} de {ley_nombre} es el siguiente:\n\n{texto}",
    "La redacción actual del artículo {art} de {ley_nombre} es:\n\n{texto}",
    "El artículo {art} de {ley_nombre} establece:\n\n{texto}",
]


# ── calculate_plazo ───────────────────────────────────────────────────────────
_PLAZO_CASES = [
    ("2025-03-01", 20, "habiles",   "20 días hábiles desde el 1 de marzo de 2025"),
    ("2025-01-15", 15, "naturales", "15 días naturales desde el 15 de enero de 2025"),
    ("2025-06-10", 30, "habiles",   "30 días hábiles desde el 10 de junio de 2025"),
    ("2025-04-01", 10, "habiles",   "10 días hábiles desde el 1 de abril de 2025"),
    ("2025-09-01", 30, "naturales", "30 días naturales desde el 1 de septiembre de 2025"),
    ("2025-02-01", 40, "habiles",   "40 días hábiles desde el 1 de febrero de 2025"),
    ("2025-11-01", 60, "habiles",   "60 días hábiles desde el 1 de noviembre de 2025"),
    ("2025-07-15", 15, "naturales", "15 días naturales desde el 15 de julio de 2025"),
]

_PLAZO_PROMPT_TEMPLATES = [
    "¿Cuándo vence el plazo de {dias} días {tipo} desde el {fecha_es}?",
    "Tengo {dias} días {tipo} desde el {fecha_es}, ¿cuándo es el último día?",
    "¿En qué fecha caduca el recurso si tengo {dias} días {tipo} desde el {fecha_es}?",
    "Calcula el plazo de {dias} días {tipo} a partir del {fecha_es}.",
    "El plazo para recurrir es de {dias} días {tipo} desde el {fecha_es}. ¿Cuándo vence?",
    "¿Cuál es la fecha límite si el plazo es de {dias} días {tipo} desde el {fecha_es}?",
    "Necesito saber cuándo expiran los {dias} días {tipo} contados desde el {fecha_es}.",
    "Desde el {fecha_es} tengo {dias} días {tipo} para presentar el escrito. ¿Cuándo es la fecha tope?",
]

_PLAZO_ANSWER_TEMPLATES = [
    "Contando {dias} días {tipo} desde el {fecha_es}, el plazo vence el **{fecha_fin}** ({detalle}).",
    "El plazo de {dias} días {tipo} desde el {fecha_es} expira el **{fecha_fin}** ({detalle}). En total han transcurrido {dias_totales} días de calendario.",
    "La fecha límite es el **{fecha_fin}** ({detalle}), que resulta de contar {dias} días {tipo} a partir del {fecha_es}.",
    "El {dias}º día {tipo} desde el {fecha_es} cae en el **{fecha_fin}** ({detalle}).",
]

_FECHA_ES_LABELS = {
    "2025-03-01": "1 de marzo de 2025",
    "2025-01-15": "15 de enero de 2025",
    "2025-06-10": "10 de junio de 2025",
    "2025-04-01": "1 de abril de 2025",
    "2025-09-01": "1 de septiembre de 2025",
    "2025-02-01": "1 de febrero de 2025",
    "2025-11-01": "1 de noviembre de 2025",
    "2025-07-15": "15 de julio de 2025",
}

# ─── Example generators ────────────────────────────────────────────────────────

ExampleList = list[dict[str, str]]


def _make_search_boe_examples(rng: random.Random, target: int) -> ExampleList:
    examples: ExampleList = []
    while len(examples) < target:
        topic_tuple = rng.choice(_BOE_TOPICS)
        topic_label, query, keyword = topic_tuple
        prompt = rng.choice(_BOE_PROMPT_TEMPLATES).format(topic=topic_label)

        example_type = rng.choices(["simple", "not_found"], weights=[85, 15])[0]

        if example_type == "not_found":
            tool_str = _tool_call("search_boe", query=query)
            result_str = _not_found_result(f"No se encontraron normas recientes sobre '{query}' en el BOE.")
            answer = (
                f"Lo siento, no he encontrado información legislativa reciente sobre {topic_label} "
                f"en el BOE. Te recomiendo consultar directamente boe.es o contactar con un profesional jurídico."
            )
            response = f"{tool_str}\n{result_str}\n{answer}"
        else:
            hits = rng.choice(_BOE_RESULTS_POOL)
            tool_str = _tool_call("search_boe", query=query)
            result_str = _result("ok", hits)
            items_text = _fmt_boe_items(hits)
            primera_norma = hits[0]["titulo"][:80]
            answer = rng.choice(_BOE_ANSWER_TEMPLATES).format(
                n=len(hits),
                topic=topic_label,
                items=items_text,
                primera_norma=primera_norma,
            )
            response = f"{tool_str}\n{result_str}\n{answer}"

        examples.append({"prompt": prompt, "response": response})
    return examples


def _make_search_cendoj_examples(rng: random.Random, target: int) -> ExampleList:
    examples: ExampleList = []
    while len(examples) < target:
        topic_tuple = rng.choice(_CENDOJ_TOPICS)
        topic_label, query = topic_tuple
        prompt = rng.choice(_CENDOJ_PROMPT_TEMPLATES).format(topic=topic_label)

        example_type = rng.choices(["simple", "not_found"], weights=[80, 20])[0]

        if example_type == "not_found":
            tool_str = _tool_call("search_cendoj", query=query)
            result_str = _not_found_result(f"No se encontraron sentencias sobre '{query}' en CENDOJ.")
            answer = (
                f"No he podido localizar jurisprudencia del Tribunal Supremo sobre {topic_label} "
                f"en CENDOJ. Puede que la materia sea muy específica o que las sentencias no estén "
                f"indexadas con esos términos. Te recomiendo buscar directamente en poderjudicial.es."
            )
            response = f"{tool_str}\n{result_str}\n{answer}"
        else:
            hits = rng.choice(_CENDOJ_RESULTS_POOL)
            tool_str = _tool_call("search_cendoj", query=query)
            result_str = _result("ok", hits)
            items_text = _fmt_cendoj_items(hits)
            resumen_breve = hits[0]["resumen"][:100]
            answer = rng.choice(_CENDOJ_ANSWER_TEMPLATES).format(
                topic=topic_label,
                items=items_text,
                resumen_breve=resumen_breve,
            )
            response = f"{tool_str}\n{result_str}\n{answer}"

        examples.append({"prompt": prompt, "response": response})
    return examples


def _make_search_web_examples(rng: random.Random, target: int) -> ExampleList:
    examples: ExampleList = []
    while len(examples) < target:
        query_tuple = rng.choice(_WEB_QUERIES)
        query, topic = query_tuple
        prompt = rng.choice(_WEB_PROMPT_TEMPLATES).format(topic=topic)

        example_type = rng.choices(["simple", "not_found"], weights=[85, 15])[0]

        if example_type == "not_found":
            tool_str = _tool_call("search_web", query=query)
            result_str = _not_found_result(f"No se encontraron resultados web para '{query}'.")
            answer = (
                f"La búsqueda no ha devuelto resultados relevantes sobre {topic}. "
                f"Te sugiero reformular la consulta o buscar directamente en bases de datos "
                f"jurídicas especializadas como Aranzadi, La Ley o Westlaw."
            )
            response = f"{tool_str}\n{result_str}\n{answer}"
        else:
            hits = rng.choice(_WEB_RESULTS_POOL)
            tool_str = _tool_call("search_web", query=query)
            result_str = _result("ok", hits)
            items_text = _fmt_web_items(hits)
            idea = hits[0]["snippet"][:80]
            answer = rng.choice(_WEB_ANSWER_TEMPLATES).format(
                topic=topic,
                items=items_text,
                idea=idea,
            )
            response = f"{tool_str}\n{result_str}\n{answer}"

        examples.append({"prompt": prompt, "response": response})
    return examples


def _make_get_article_examples(rng: random.Random, target: int) -> ExampleList:
    examples: ExampleList = []
    while len(examples) < target:
        req = rng.choice(_ARTICLE_REQUESTS)
        ley_key, art, ley_nombre, materia = req
        prompt = rng.choice(_ARTICLE_PROMPT_TEMPLATES).format(art=art, ley_nombre=ley_nombre)

        example_type = rng.choices(["simple", "not_found"], weights=[85, 15])[0]

        if example_type == "not_found":
            tool_str = _tool_call("get_article", ley=ley_key, articulo=art)
            result_str = _not_found_result(f"No se encontró el artículo {art} en {ley_key}.")
            answer = (
                f"No he podido recuperar el texto del artículo {art} de {ley_nombre}. "
                f"Puede que la versión consolidada no esté disponible en este momento. "
                f"Consulta la versión actualizada en boe.es."
            )
            response = f"{tool_str}\n{result_str}\n{answer}"
        else:
            texto = _ARTICLE_TEXTS_POOL.get((ley_key, art), f"Artículo {art}. [Texto de {ley_nombre}].")
            tool_str = _tool_call("get_article", ley=ley_key, articulo=art)
            result_str = _result("ok", texto)
            answer = rng.choice(_ARTICLE_ANSWER_TEMPLATES).format(
                art=art, ley_nombre=ley_nombre, texto=texto
            )
            response = f"{tool_str}\n{result_str}\n{answer}"

        examples.append({"prompt": prompt, "response": response})
    return examples


def _make_calculate_plazo_examples(rng: random.Random, target: int) -> ExampleList:
    examples: ExampleList = []
    while len(examples) < target:
        case = rng.choice(_PLAZO_CASES)
        fecha_iso, dias, tipo, desc = case
        fecha_es = _FECHA_ES_LABELS.get(fecha_iso, fecha_iso)
        tipo_label = "hábiles" if tipo == "habiles" else "naturales"

        prompt = rng.choice(_PLAZO_PROMPT_TEMPLATES).format(
            dias=dias, tipo=tipo_label, fecha_es=fecha_es
        )

        example_type = rng.choices(["simple", "not_found"], weights=[90, 10])[0]

        if example_type == "not_found":
            tool_str = _tool_call("calculate_plazo", fecha_inicio=fecha_iso, dias=dias, tipo=tipo)
            result_str = _not_found_result("Fecha de inicio inválida o fuera de rango.")
            answer = (
                f"No he podido calcular el plazo. "
                f"Comprueba que la fecha de inicio esté en formato YYYY-MM-DD y que el número de días sea positivo."
            )
            response = f"{tool_str}\n{result_str}\n{answer}"
        else:
            plazo_data = rng.choice(_PLAZO_RESULTS_POOL)
            tool_str = _tool_call("calculate_plazo", fecha_inicio=fecha_iso, dias=dias, tipo=tipo)
            result_str = _result("ok", plazo_data)
            answer = rng.choice(_PLAZO_ANSWER_TEMPLATES).format(
                dias=dias,
                tipo=tipo_label,
                fecha_es=fecha_es,
                fecha_fin=plazo_data["fecha_fin"],
                detalle=plazo_data["detalle"],
                dias_totales=plazo_data["dias_totales"],
            )
            response = f"{tool_str}\n{result_str}\n{answer}"

        examples.append({"prompt": prompt, "response": response})
    return examples


# ─── Multi-tool examples ──────────────────────────────────────────────────────

def _make_multi_tool_examples(rng: random.Random, n_per_tool: int) -> ExampleList:
    """Generate multi-step examples: BOE search → get_article, CENDOJ → BOE, etc."""
    examples: ExampleList = []

    # Pattern A: search_boe → get_article
    pattern_a_prompts = [
        "¿Qué dice la ley de enjuiciamiento civil sobre la carga de la prueba? Dame el texto exacto del artículo.",
        "Busca la normativa sobre despido disciplinario y muéstrame el artículo del ET que lo regula.",
        "¿Cómo regula el Código Penal la estafa? Quiero ver el artículo exacto.",
        "Localiza la regulación del derecho a la tutela judicial y muéstrame el artículo de la Constitución.",
        "Busca información sobre el despido nulo y luego dame el texto del artículo 55 del ET.",
    ]
    for _ in range(n_per_tool):
        prompt = rng.choice(pattern_a_prompts)
        # Step 1: search_boe
        hits = rng.choice(_BOE_RESULTS_POOL)
        t1 = _tool_call("search_boe", query="carga de la prueba enjuiciamiento civil")
        r1 = _result("ok", hits)
        # Step 2: get_article
        req = rng.choice(_ARTICLE_REQUESTS)
        ley_key, art, ley_nombre, _ = req
        texto = _ARTICLE_TEXTS_POOL.get((ley_key, art), f"Artículo {art}. [Texto de ejemplo].")
        t2 = _tool_call("get_article", ley=ley_key, articulo=art)
        r2 = _result("ok", texto)
        answer = (
            f"He localizado la normativa en el BOE y obtenido el texto exacto del artículo {art} de {ley_nombre}:\n\n"
            f"{texto}\n\n"
            f"Como puedes ver, {ley_nombre} establece en este artículo los requisitos formales y sustantivos aplicables."
        )
        response = f"{t1}\n{r1}\n{t2}\n{r2}\n{answer}"
        examples.append({"prompt": prompt, "response": response})

    # Pattern B: search_cendoj → search_boe
    pattern_b_prompts = [
        "Quiero conocer la jurisprudencia del TS sobre las cláusulas abusivas y también qué dice la normativa del BOE.",
        "¿Cómo resuelve el TS los casos de responsabilidad patrimonial? Y además, ¿hay normativa nueva en el BOE?",
        "Dame jurisprudencia del TS sobre accidentes laborales y la normativa vigente en el BOE.",
    ]
    for _ in range(n_per_tool):
        prompt = rng.choice(pattern_b_prompts)
        cj_hits = rng.choice(_CENDOJ_RESULTS_POOL)
        boe_hits = rng.choice(_BOE_RESULTS_POOL)
        t1 = _tool_call("search_cendoj", query="cláusulas abusivas hipotecas")
        r1 = _result("ok", cj_hits)
        t2 = _tool_call("search_boe", query="cláusulas abusivas protección consumidores")
        r2 = _result("ok", boe_hits)
        answer = (
            "He consultado tanto la jurisprudencia del Tribunal Supremo como la normativa del BOE:\n\n"
            f"**Jurisprudencia TS:**\n{_fmt_cendoj_items(cj_hits)}\n\n"
            f"**Normativa BOE:**\n{_fmt_boe_items(boe_hits)}\n\n"
            "En conjunto, la regulación y la jurisprudencia ofrecen un marco completo sobre esta materia."
        )
        response = f"{t1}\n{r1}\n{t2}\n{r2}\n{answer}"
        examples.append({"prompt": prompt, "response": response})

    # Pattern C: calculate_plazo with context (web search for deadline rules)
    pattern_c_prompts = [
        "¿Cuándo vence el plazo para recurrir si tengo 20 días hábiles desde el 1 de marzo? ¿Hay jurisprudencia sobre cómputo de plazos?",
        "Calcula mi plazo de 15 días naturales desde el 15 de enero y busca si el TS tiene doctrina sobre plazos de recurso.",
    ]
    for _ in range(n_per_tool):
        prompt = rng.choice(pattern_c_prompts)
        case = rng.choice(_PLAZO_CASES)
        fecha_iso, dias, tipo, _ = case
        plazo_data = rng.choice(_PLAZO_RESULTS_POOL)
        cj_hits = rng.choice(_CENDOJ_RESULTS_POOL)
        t1 = _tool_call("calculate_plazo", fecha_inicio=fecha_iso, dias=dias, tipo=tipo)
        r1 = _result("ok", plazo_data)
        t2 = _tool_call("search_cendoj", query="cómputo de plazos procesales días hábiles")
        r2 = _result("ok", cj_hits)
        tipo_label = "hábiles" if tipo == "habiles" else "naturales"
        answer = (
            f"El plazo de {dias} días {tipo_label} desde {_FECHA_ES_LABELS.get(fecha_iso, fecha_iso)} "
            f"vence el **{plazo_data['fecha_fin']}** ({plazo_data['detalle']}).\n\n"
            f"Sobre el cómputo de plazos procesales, el TS ha establecido:\n\n{_fmt_cendoj_items(cj_hits)}"
        )
        response = f"{t1}\n{r1}\n{t2}\n{r2}\n{answer}"
        examples.append({"prompt": prompt, "response": response})

    return examples


# ─── Generation orchestrator ──────────────────────────────────────────────────

_GENERATORS: list[tuple[str, Callable[[random.Random, int], ExampleList]]] = [
    ("search_boe",       _make_search_boe_examples),
    ("search_cendoj",    _make_search_cendoj_examples),
    ("search_web",       _make_search_web_examples),
    ("get_article",      _make_get_article_examples),
    ("calculate_plazo",  _make_calculate_plazo_examples),
]


def generate_examples(total: int = 1000, seed: int = 42) -> list[dict[str, str]]:
    """Generate *total* synthetic tool-use training examples.

    The examples are distributed roughly evenly across all five tools, with a
    small portion reserved for multi-tool (chain) examples.

    Parameters
    ----------
    total: Total number of examples to generate (default 1000).
    seed:  Random seed for reproducibility (default 42).

    Returns a shuffled list of {"prompt": ..., "response": ...} dicts.
    """
    rng = random.Random(seed)
    # Reserve ~10% for multi-tool chains
    multi_total = max(10, total // 10)
    single_total = total - multi_total
    per_tool = single_total // len(_GENERATORS)
    remainder = single_total % len(_GENERATORS)

    all_examples: list[dict[str, str]] = []

    for idx, (name, gen_fn) in enumerate(_GENERATORS):
        n = per_tool + (1 if idx < remainder else 0)
        logger.info("Generating %d examples for tool '%s'", n, name)
        batch = gen_fn(rng, n)
        all_examples.extend(batch)
        logger.info("  → %d examples generated for '%s'", len(batch), name)

    # Multi-tool examples (distributed across patterns)
    n_per_pattern = multi_total // 3
    logger.info("Generating %d multi-tool chain examples", multi_total)
    multi = _make_multi_tool_examples(rng, n_per_pattern)
    all_examples.extend(multi)
    logger.info("  → %d multi-tool examples generated", len(multi))

    rng.shuffle(all_examples)
    logger.info("Total examples generated: %d", len(all_examples))
    return all_examples


# ─── JSONL writer ─────────────────────────────────────────────────────────────

def write_jsonl(examples: list[dict[str, str]], output_path: Path) -> None:
    """Write *examples* as JSONL to *output_path*, creating parent dirs as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    size_mb = output_path.stat().st_size / 1e6
    logger.info("Wrote %d examples to %s (%.2f MB)", len(examples), output_path, size_mb)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate synthetic tool-use training data for the herramientas LoRA adapter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/download_tool_data.py --output data/finetune/herramientas.jsonl
  python scripts/download_tool_data.py --output data/finetune/herramientas.jsonl --count 1000
  python scripts/download_tool_data.py --output data/finetune/herramientas.jsonl --count 500 --seed 7
""",
    )
    p.add_argument(
        "--output", "-o",
        default="data/finetune/herramientas.jsonl",
        help="Output JSONL file path (default: data/finetune/herramientas.jsonl)",
    )
    p.add_argument(
        "--count", "-n",
        type=int,
        default=1000,
        help="Total number of examples to generate (default: 1000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="Print N example(s) to stdout and exit without writing file",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.count < 5:
        logger.error("--count must be at least 5")
        sys.exit(1)

    examples = generate_examples(total=args.count, seed=args.seed)

    if args.preview > 0:
        for ex in examples[: args.preview]:
            print("─" * 60)
            print("PROMPT:", ex["prompt"])
            print("RESPONSE:", ex["response"])
        print(f"\n(showing {min(args.preview, len(examples))} of {len(examples)} examples)")
        return

    output_path = Path(args.output)
    write_jsonl(examples, output_path)
    logger.info("Done. Fine-tune with: python scripts/lora_finetune.py --data %s --specialty all", output_path)


if __name__ == "__main__":
    main()
