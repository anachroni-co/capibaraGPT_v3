#!/usr/bin/env python3
"""Download and format instruction-tuning datasets for LoRA fine-tuning.

Adapters produced
-----------------
  instruccion  — General instruction following (Alpaca-es, OpenAssistant-es, Dolly-es)
  qa           — Question answering over context (SQuAD-es, XQuAD, MLQA)
  extraccion   — Named entity extraction (CoNLL-2002, WikiNEuRal)
  redaccion    — Formal legal writing (synthetic templates)
  dialogo      — Multi-turn conversation (OpenAssistant-es multi-turn chains)
  razonamiento — Step-by-step reasoning (MGSM-es, mCoT)
  traduccion   — Translation Spanish ↔ English/Catalan (OPUS-100, Europarl)
  all          — All of the above

Each adapter writes:
  data/finetune/<adapter>.jsonl    — {"prompt": …, "response": …} pairs

Usage
-----
    # All adapters
    python scripts/download_instruction_data.py --source all

    # Single adapter
    python scripts/download_instruction_data.py --source qa

    # Limit examples per dataset (useful for smoke tests)
    python scripts/download_instruction_data.py --source all --max-examples 5000

    # Then fine-tune any adapter
    python scripts/lora_finetune.py \\
        --base-ckpt checkpoints/axion_large_legal/soup_uniform.pkl \\
        --preset    large \\
        --data      data/finetune/instruccion.jsonl \\
        --specialty instruccion \\
        --output    checkpoints/lora/large_instruccion \\
        --steps 2000 --rank 16 --lora-alpha 32 --dtype bf16
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("instruct")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_jsonl(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    mb = path.stat().st_size / 1e6
    logger.info("  → %s  (%d pairs, %.1f MB)", path, len(pairs), mb)


def _load_hf(dataset_id: str, config=None, split="train", streaming=True):
    """Load a HuggingFace dataset, trying multiple split names."""
    try:
        import datasets as hf
    except ImportError:
        logger.error("pip install datasets huggingface_hub")
        return None

    splits_to_try = [split, "train", "train[:100%]"]
    kwargs = dict(streaming=streaming)
    if config:
        kwargs["name"] = config

    for s in splits_to_try:
        try:
            ds = hf.load_dataset(dataset_id, split=s, **kwargs)
            logger.info("  Loaded %s (config=%s, split=%s)", dataset_id, config, s)
            return ds
        except Exception:
            continue
    logger.warning("  Could not load %s (config=%s) — skipping", dataset_id, config)
    return None


# ── Adapter: instruccion ───────────────────────────────────────────────────────

def download_instruccion(finetune_dir: Path, max_ex: int) -> None:
    """Alpaca-es + OpenAssistant Spanish + Dolly-es → general instruction following."""
    pairs: list[dict] = []

    # ── Alpaca Spanish ──
    for dataset_id in (
        "bertin-project/alpaca-es",
        "argilla/alpaca-cleaned-es",
        "GRJR/alpaca-spanish",
    ):
        ds = _load_hf(dataset_id, streaming=True)
        if ds is None:
            continue
        count = 0
        for row in ds:
            if max_ex and count >= max_ex // 3:
                break
            inst  = (row.get("instruction") or "").strip()
            inp   = (row.get("input") or "").strip()
            out   = (row.get("output") or row.get("response") or "").strip()
            if not inst or not out or len(out) < 10:
                continue
            prompt = f"{inst}\n\n{inp}".strip() if inp else inst
            pairs.append({"prompt": prompt, "response": out})
            count += 1
        logger.info("  Alpaca-es (%s): %d examples", dataset_id, count)
        break  # use first that works

    # ── Dolly-es ──
    for dataset_id in (
        "argilla/databricks-dolly-15k-curated-es",
        "brainsphere/dolly-15k-spanish",
        "HuggingFaceH4/databricks_dolly_15k",   # fallback: English Dolly
    ):
        ds = _load_hf(dataset_id, streaming=True)
        if ds is None:
            continue
        count = 0
        for row in ds:
            if max_ex and count >= max_ex // 3:
                break
            # Spanish curated: 'instruction', 'context', 'response'
            # English fallback: 'instruction', 'context', 'response'
            inst    = (row.get("instruction") or "").strip()
            context = (row.get("context") or "").strip()
            resp    = (row.get("response") or row.get("output") or "").strip()
            # Language filter for English fallback
            lang = row.get("lang", "es")
            if "databricks_dolly_15k" in dataset_id and lang not in ("es", ""):
                continue
            if not inst or not resp or len(resp) < 10:
                continue
            prompt = f"{inst}\n\nContexto: {context}".strip() if context else inst
            pairs.append({"prompt": prompt, "response": resp})
            count += 1
        logger.info("  Dolly-es (%s): %d examples", dataset_id, count)
        break

    # ── OpenAssistant Spanish (single-turn only; multi-turn → dialogo) ──
    ds = _load_hf("OpenAssistant/oasst1", streaming=False)
    if ds is not None:
        # Build {message_id: row} map and extract root-level human→assistant pairs
        by_id: dict[str, dict] = {}
        for row in ds:
            by_id[row["message_id"]] = row

        count = 0
        for mid, row in by_id.items():
            if max_ex and count >= max_ex // 3:
                break
            if row.get("lang") != "es":
                continue
            if row.get("role") != "prompter":
                continue
            if row.get("parent_id") is not None:
                continue  # not root — skip (used in dialogo)
            # Find the best-ranked assistant reply
            replies = [
                r for r in by_id.values()
                if r.get("parent_id") == mid and r.get("role") == "assistant"
            ]
            if not replies:
                continue
            best = min(replies, key=lambda r: r.get("rank") or 99)
            prompt = row["text"].strip()
            resp   = best["text"].strip()
            if len(prompt) < 10 or len(resp) < 10:
                continue
            pairs.append({"prompt": prompt, "response": resp})
            count += 1
        logger.info("  OpenAssistant-es (root pairs): %d examples", count)

    _write_jsonl(pairs, finetune_dir / "instruccion.jsonl")
    logger.info("instruccion total: %d pairs", len(pairs))


# ── Adapter: qa ───────────────────────────────────────────────────────────────

_QA_TEMPLATES = [
    "Basándote en el siguiente texto, responde la pregunta.\n\nTexto:\n{context}\n\nPregunta: {question}",
    "Lee el siguiente pasaje y responde:\n\n{context}\n\n¿{question}",
    "Usando el texto como referencia, ¿{question}\n\nTexto:\n{context}",
    "Texto de referencia:\n{context}\n\nPregunta: {question}",
]


def _qa_prompt(context: str, question: str, idx: int) -> str:
    t = _QA_TEMPLATES[idx % len(_QA_TEMPLATES)]
    return t.format(context=context.strip(), question=question.strip())


def download_qa(finetune_dir: Path, max_ex: int) -> None:
    """SQuAD-es + XQuAD-es + MLQA-es → reading comprehension Q&A."""
    pairs: list[dict] = []

    sources = [
        # (dataset_id, config, context_field, question_field, answers_field)
        ("xquad",        "xquad.es",    "context", "question", "answers"),
        ("mlqa",         "mlqa.es.es",  "context", "question", "answers"),
        ("squad_es",     None,          "context", "question", "answers"),
        ("rajpurkar/squad_es", None,    "context", "question", "answers"),
    ]

    per_src = max_ex // len(sources) if max_ex else 0

    for ds_id, cfg, ctx_f, q_f, ans_f in sources:
        ds = _load_hf(ds_id, config=cfg, streaming=True)
        if ds is None:
            continue
        count = 0
        for idx, row in enumerate(ds):
            if per_src and count >= per_src:
                break
            context  = (row.get(ctx_f) or "").strip()
            question = (row.get(q_f)   or "").strip()
            answers  = row.get(ans_f, {})

            if isinstance(answers, dict):
                ans_list = answers.get("text", [])
            elif isinstance(answers, list):
                ans_list = answers
            else:
                continue

            if not context or not question or not ans_list:
                continue
            answer = ans_list[0].strip() if isinstance(ans_list[0], str) else ""
            if not answer:
                continue

            if len(context) > 2000:
                context = context[:2000]

            pairs.append({
                "prompt":   _qa_prompt(context, question, idx),
                "response": answer,
            })
            count += 1

        logger.info("  %s (config=%s): %d examples", ds_id, cfg, count)
        if count > 0 and per_src and len(pairs) >= max_ex:
            break

    _write_jsonl(pairs, finetune_dir / "qa.jsonl")
    logger.info("qa total: %d pairs", len(pairs))


# ── Adapter: extraccion ────────────────────────────────────────────────────────

# CoNLL-2002 / WikiNEuRal tag indices → label name
_CONLL_TAGS = {
    0: "O", 1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
    7: "B-MISC", 8: "I-MISC",
}
_TYPE_ES = {
    "PER":  "Persona",
    "ORG":  "Organización",
    "LOC":  "Lugar",
    "MISC": "Otro",
}
_EXTRAC_TEMPLATES = [
    "Identifica y clasifica las entidades nombradas en el siguiente texto:\n\n{text}",
    "Extrae todas las entidades (personas, organizaciones, lugares) del texto:\n\n{text}",
    "¿Qué entidades nombradas aparecen en este texto?\n\n{text}",
    "Lista todas las personas, organizaciones y lugares mencionados en:\n\n{text}",
]


def _ner_to_pairs(tokens: list[str], tags: list[int], idx: int) -> dict | None:
    """Convert a NER-tagged sentence into an instruction pair."""
    text = " ".join(tokens)
    if len(text) < 20:
        return None

    # Reconstruct entity spans
    entities: list[dict] = []
    current: list[str] = []
    current_type = ""

    for tok, tag_id in zip(tokens, tags):
        label = _CONLL_TAGS.get(tag_id, "O")
        if label.startswith("B-"):
            if current:
                entities.append({"texto": " ".join(current), "tipo": current_type})
            current = [tok]
            current_type = label[2:]
        elif label.startswith("I-") and current:
            current.append(tok)
        else:
            if current:
                entities.append({"texto": " ".join(current), "tipo": current_type})
            current, current_type = [], ""

    if current:
        entities.append({"texto": " ".join(current), "tipo": current_type})

    if not entities:
        return None

    # Format response as readable list
    lines = [
        f"- {_TYPE_ES.get(e['tipo'], e['tipo'])}: {e['texto']}"
        for e in entities
    ]
    response = "\n".join(lines)

    template = _EXTRAC_TEMPLATES[idx % len(_EXTRAC_TEMPLATES)]
    return {"prompt": template.format(text=text), "response": response}


def download_extraccion(finetune_dir: Path, max_ex: int) -> None:
    """CoNLL-2002 Spanish NER + WikiNEuRal → entity extraction."""
    pairs: list[dict] = []

    sources = [
        ("conll2002",             "es"),
        ("Babelscape/wikineural", "es"),
    ]

    per_src = max_ex // len(sources) if max_ex else 0

    for ds_id, cfg in sources:
        ds = _load_hf(ds_id, config=cfg, streaming=False)
        if ds is None:
            continue
        count = 0
        for idx, row in enumerate(ds):
            if per_src and count >= per_src:
                break
            tokens = row.get("tokens") or row.get("words") or []
            tags   = row.get("ner_tags") or row.get("tags") or []
            if not tokens or not tags or len(tokens) != len(tags):
                continue
            pair = _ner_to_pairs(tokens, tags, idx)
            if pair:
                pairs.append(pair)
                count += 1

        logger.info("  %s (config=%s): %d examples", ds_id, cfg, count)

    _write_jsonl(pairs, finetune_dir / "extraccion.jsonl")
    logger.info("extraccion total: %d pairs", len(pairs))


# ── Adapter: redaccion (synthetic templates) ───────────────────────────────────

# Each template: (prompt_template, response_template)
# Variables in {MAYUSCULAS} are filled randomly from the lists below.

_NOMBRES = [
    "María García López", "Carlos Martínez Ruiz", "Ana Fernández Soto",
    "José Rodríguez Pérez", "Isabel Sánchez Torres", "Luis González Mora",
    "Elena Jiménez Vega", "Antonio López Blanco", "Carmen Díaz Ortega",
    "Francisco Romero Gil",
]
_CIUDADES = [
    "Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza",
    "Málaga", "Bilbao", "Alicante", "Córdoba", "Valladolid",
]
_CALLES = [
    "Calle Mayor, 15, 3.º A", "Avenida de la Constitución, 42, 1.º B",
    "Paseo de Gracia, 88, 2.º", "Calle del Sol, 7, bajo",
    "Gran Vía, 23, 4.º C", "Plaza de España, 3, entresuelo",
]
_IMPORTES = ["650", "800", "950", "1.100", "1.250", "700", "875"]
_DURACIONES = ["un año", "dos años", "tres años", "un año prorrogable"]
_JUZGADOS = [
    "Juzgado de Primera Instancia n.º 3 de Madrid",
    "Juzgado de lo Social n.º 1 de Barcelona",
    "Juzgado de lo Contencioso-Administrativo n.º 2 de Sevilla",
    "Juzgado de Instrucción n.º 5 de Valencia",
]
_PROCURADOR_NOMBRES = [
    "Dña. Raquel Moreno Serrano", "D. Javier Navarro Ibáñez",
    "Dña. Patricia Suárez Blanco", "D. Manuel Castro Reyes",
]

_REDACCION_TEMPLATES = [
    # ── Contrato de arrendamiento ──
    (
        "Redacta un contrato de arrendamiento de vivienda con los siguientes datos:\n"
        "- Arrendador: {ARRENDADOR}\n"
        "- Arrendatario: {ARRENDATARIO}\n"
        "- Inmueble: {DIRECCION}, {CIUDAD}\n"
        "- Renta mensual: {IMPORTE} €\n"
        "- Duración: {DURACION}\n"
        "- Fecha de inicio: 1 de enero de 2025",

        "CONTRATO DE ARRENDAMIENTO DE VIVIENDA\n\n"
        "En {CIUDAD}, a 1 de enero de 2025.\n\n"
        "REUNIDOS\n\n"
        "De una parte, D./Dña. {ARRENDADOR}, mayor de edad, con DNI __________, "
        "en adelante el ARRENDADOR.\n\n"
        "De otra parte, D./Dña. {ARRENDATARIO}, mayor de edad, con DNI __________, "
        "en adelante el ARRENDATARIO.\n\n"
        "MANIFIESTAN\n\n"
        "Que el ARRENDADOR es propietario del inmueble sito en {DIRECCION}, {CIUDAD}, "
        "y que ambas partes acuerdan celebrar el presente contrato de arrendamiento "
        "conforme a la Ley 29/1994, de 24 de noviembre, de Arrendamientos Urbanos, "
        "con arreglo a las siguientes\n\n"
        "CLÁUSULAS\n\n"
        "PRIMERA. El ARRENDADOR cede en arrendamiento al ARRENDATARIO el inmueble "
        "descrito para su uso como vivienda habitual y permanente.\n\n"
        "SEGUNDA. La duración del presente contrato será de {DURACION}, con inicio "
        "el día 1 de enero de 2025. Transcurrido dicho plazo, el contrato se "
        "prorrogará automáticamente por períodos anuales salvo denuncia de cualquiera "
        "de las partes con un preaviso mínimo de 30 días.\n\n"
        "TERCERA. La renta mensual se fija en {IMPORTE} euros (€), pagaderos por "
        "adelantado dentro de los cinco primeros días de cada mes mediante "
        "transferencia bancaria a la cuenta que designe el ARRENDADOR.\n\n"
        "CUARTA. El ARRENDATARIO entrega en concepto de fianza una mensualidad de "
        "renta ({IMPORTE} €), en cumplimiento del artículo 36 LAU, cuya devolución "
        "se efectuará a la finalización del contrato previo estado satisfactorio "
        "del inmueble.\n\n"
        "QUINTA. Los gastos de comunidad, IBI y demás tributos que recaigan sobre "
        "el inmueble serán a cargo del ARRENDADOR, salvo pacto en contrario.\n\n"
        "SEXTA. El ARRENDATARIO no podrá subarrendar ni ceder el contrato sin "
        "consentimiento escrito del ARRENDADOR.\n\n"
        "Y en prueba de conformidad, las partes firman el presente contrato en el "
        "lugar y fecha indicados.\n\n"
        "EL ARRENDADOR                    EL ARRENDATARIO\n\n"
        "_____________________            _____________________\n"
        "{ARRENDADOR}                     {ARRENDATARIO}",
    ),

    # ── Escrito de demanda civil ──
    (
        "Redacta el encabezado y el cuerpo de una demanda civil de reclamación de "
        "cantidad por {IMPORTE} euros contra {DEMANDADO}, presentada ante el "
        "{JUZGADO} por {DEMANDANTE}.",

        "AL {JUZGADO}\n\n"
        "D./Dña. {DEMANDANTE}, Procurador/a de los Tribunales, en nombre y "
        "representación de D./Dña. {DEMANDANTE_REAL}, según escritura de poder "
        "que se acompaña, bajo la dirección letrada de D./Dña. __________, "
        "ante este Juzgado comparezco y, como mejor proceda en Derecho, DIGO:\n\n"
        "Que por medio del presente escrito, y en ejercicio de la acción de "
        "reclamación de cantidad, formulo DEMANDA DE JUICIO ORDINARIO contra "
        "D./Dña. {DEMANDADO}, con domicilio a efectos de notificaciones en "
        "__________, en base a los siguientes\n\n"
        "HECHOS\n\n"
        "PRIMERO. Mi representado/a y el/la demandado/a suscribieron con fecha "
        "__________ un contrato por el que el/la demandado/a quedó obligado/a a "
        "abonar la cantidad de {IMPORTE} euros (€).\n\n"
        "SEGUNDO. Vencida la obligación, el/la demandado/a ha incumplido "
        "reiteradamente su obligación de pago pese a los requerimientos "
        "fehacientes efectuados, sin que hasta la fecha se haya satisfecho "
        "cantidad alguna.\n\n"
        "FUNDAMENTOS DE DERECHO\n\n"
        "I. COMPETENCIA Y JURISDICCIÓN. Corresponde a los Juzgados de Primera "
        "Instancia el conocimiento de los asuntos civiles conforme a los "
        "artículos 45 y 50 LEC.\n\n"
        "II. LEGITIMACIÓN. Mi representado/a ostenta plena legitimación activa "
        "por ser titular del crédito reclamado.\n\n"
        "III. FONDO. Conforme al artículo 1.091 CC, las obligaciones nacidas de "
        "los contratos tienen fuerza de ley entre las partes contratantes. "
        "El artículo 1.101 CC prevé la indemnización de daños y perjuicios "
        "causada por el incumplimiento.\n\n"
        "SUPLICO AL JUZGADO que, teniendo por presentada esta demanda, se sirva "
        "admitirla y, previos los trámites legales oportunos, dicte Sentencia "
        "condenando al/a la demandado/a a abonar a mi representado/a la cantidad "
        "de {IMPORTE} euros, más los intereses legales y las costas procesales.\n\n"
        "OTROSÍ DIGO que se acompaña poder para pleitos, documentos contractuales "
        "y justificantes de requerimiento.\n\n"
        "En {CIUDAD}, a __ de __________ de 2025.\n\n"
        "FIRMA DEL PROCURADOR        FIRMA DEL LETRADO\n\n"
        "_____________________       _____________________",
    ),

    # ── Recurso de alzada ──
    (
        "Redacta un recurso de alzada contra una resolución administrativa que "
        "impone una multa de {IMPORTE} euros a {RECURRENTE} por infracción de "
        "tráfico en {CIUDAD}.",

        "RECURSO DE ALZADA\n\n"
        "D./Dña. {RECURRENTE}, con DNI __________, domicilio a efectos de "
        "notificaciones en __________, ante el órgano superior competente "
        "COMPARECE y, como mejor proceda en Derecho, EXPONE:\n\n"
        "Que con fecha __________ le fue notificada Resolución de la Dirección "
        "General de Tráfico por la que se impone sanción de multa por importe de "
        "{IMPORTE} euros (€), por supuesta comisión de infracción consistente en "
        "__________.\n\n"
        "No estando conforme con la citada resolución, al amparo de los artículos "
        "112 y siguientes de la Ley 39/2015, de 1 de octubre, del Procedimiento "
        "Administrativo Común de las Administraciones Públicas, interpone el "
        "presente RECURSO DE ALZADA en base a los siguientes\n\n"
        "MOTIVOS DE IMPUGNACIÓN\n\n"
        "PRIMERO. DEFECTO DE MOTIVACIÓN. La resolución impugnada incurre en falta "
        "de motivación suficiente, vulnerando el artículo 35 LPAC, al no "
        "especificar con detalle los hechos en que se fundamenta la infracción.\n\n"
        "SEGUNDO. FALTA DE PRUEBA. La Administración no ha acreditado "
        "suficientemente los hechos constitutivos de la infracción, por lo que "
        "procede la aplicación del principio in dubio pro reo reconocido en el "
        "artículo 53.2 CE.\n\n"
        "TERCERO. PROPORCIONALIDAD. En caso de considerarse acreditada la "
        "infracción, la sanción impuesta resulta desproporcionada en relación "
        "con la gravedad de los hechos, vulnerando el principio de "
        "proporcionalidad del artículo 29.3 LRJSP.\n\n"
        "Por lo expuesto,\n\n"
        "SOLICITA que se tenga por interpuesto en tiempo y forma el presente "
        "recurso de alzada y, previo los trámites legales, se dicte resolución "
        "estimando el mismo y anulando la sanción impugnada o, subsidiariamente, "
        "reduciéndola a su grado mínimo.\n\n"
        "En {CIUDAD}, a __ de __________ de 2025.\n\n"
        "Firma: _____________________\n"
        "{RECURRENTE}",
    ),

    # ── Carta de requerimiento de pago ──
    (
        "Redacta una carta de requerimiento de pago para reclamar {IMPORTE} euros "
        "a {DEUDOR} con plazo de 15 días antes de iniciar acciones judiciales.",

        "REQUERIMIENTO DE PAGO\n\n"
        "Lugar y fecha: {CIUDAD}, __ de __________ de 2025\n\n"
        "A la atención de:\nD./Dña. {DEUDOR}\n__________ [dirección]\n\n"
        "Estimado/a Sr./Sra. {DEUDOR}:\n\n"
        "Por medio de la presente, y en nombre de mi representado/a, me dirijo a "
        "Vd. para REQUERIRLE formal y fehacientemente el pago de la cantidad de "
        "{IMPORTE} EUROS ({IMPORTE} €), que adeuda a mi representado/a en virtud "
        "de __________.\n\n"
        "Dicho importe se encuentra vencido y es exigible desde el pasado "
        "__________, sin que hasta la fecha haya procedido a su abono pese a las "
        "gestiones amistosas realizadas.\n\n"
        "En consecuencia, le otorgo un plazo IMPRORROGABLE DE QUINCE (15) DÍAS "
        "NATURALES desde la recepción de la presente para que proceda al pago "
        "íntegro del importe adeudado mediante transferencia bancaria a la cuenta "
        "IBAN ES__________ __________.\n\n"
        "Transcurrido dicho plazo sin haber recibido el pago, mi representado/a "
        "se verá en la obligación de iniciar las oportunas acciones judiciales "
        "para la reclamación de la deuda, siendo los gastos y costas procesales "
        "que se generen enteramente a su cargo, conforme al artículo 1.101 CC.\n\n"
        "Confiando en que dará a la presente la atención que merece y evitará "
        "mayores perjuicios para ambas partes, quedo a su disposición para "
        "cualquier aclaración.\n\n"
        "Atentamente,\n\n"
        "_____________________\n"
        "[Nombre del Abogado]\n"
        "Letrado del Ilustre Colegio de Abogados de {CIUDAD}\n"
        "Colegiado n.º __________",
    ),

    # ── Escrito de contestación a demanda ──
    (
        "Redacta el encabezado de un escrito de contestación a una demanda civil "
        "presentado por {DEMANDADO} ante el {JUZGADO}.",

        "AL {JUZGADO}\n\n"
        "D./Dña. {PROCURADOR}, Procurador/a de los Tribunales, en nombre y "
        "representación de D./Dña. {DEMANDADO}, según escritura de poder que se "
        "acompaña, bajo la dirección letrada de D./Dña. __________, en los autos "
        "de Juicio Ordinario n.º __________ seguidos a instancia de "
        "D./Dña. __________, ante este Juzgado comparezco y, como mejor proceda "
        "en Derecho, DIGO:\n\n"
        "Que por medio del presente escrito, dentro del plazo legal, formulo "
        "CONTESTACIÓN A LA DEMANDA en base a los siguientes\n\n"
        "HECHOS\n\n"
        "PRIMERO. Se niegan todos y cada uno de los hechos de la demanda que no "
        "sean expresamente reconocidos en el presente escrito.\n\n"
        "SEGUNDO. [Exponer los hechos propios de la defensa con claridad "
        "y numerados.]\n\n"
        "FUNDAMENTOS DE DERECHO\n\n"
        "I. [Fundamento procesal y sustantivo de la oposición.]\n\n"
        "SUPLICO AL JUZGADO que, teniendo por presentado este escrito y admitido "
        "el mismo, se sirva dictar Sentencia desestimando íntegramente la demanda "
        "con expresa condena en costas a la parte actora.\n\n"
        "En {CIUDAD}, a __ de __________ de 2025.\n\n"
        "FIRMA DEL PROCURADOR        FIRMA DEL LETRADO\n\n"
        "_____________________       _____________________",
    ),
]


def _fill(template: str, rng: random.Random) -> str:
    subs = {
        "ARRENDADOR":    rng.choice(_NOMBRES),
        "ARRENDATARIO":  rng.choice(_NOMBRES),
        "DEMANDANTE":    rng.choice(_PROCURADOR_NOMBRES),
        "DEMANDANTE_REAL": rng.choice(_NOMBRES),
        "DEMANDADO":     rng.choice(_NOMBRES),
        "RECURRENTE":    rng.choice(_NOMBRES),
        "DEUDOR":        rng.choice(_NOMBRES),
        "PROCURADOR":    rng.choice(_PROCURADOR_NOMBRES),
        "DIRECCION":     rng.choice(_CALLES),
        "CIUDAD":        rng.choice(_CIUDADES),
        "IMPORTE":       rng.choice(_IMPORTES),
        "DURACION":      rng.choice(_DURACIONES),
        "JUZGADO":       rng.choice(_JUZGADOS),
    }
    result = template
    for key, val in subs.items():
        result = result.replace("{" + key + "}", val)
    return result


def download_redaccion(finetune_dir: Path, max_ex: int) -> None:
    """
    Synthetic legal writing templates.
    Each base template is instantiated with randomised names/places/amounts
    to produce distinct training examples.
    """
    rng   = random.Random(42)
    pairs: list[dict] = []

    # How many copies per template
    n_copies = max(50, (max_ex or 500) // len(_REDACCION_TEMPLATES))

    for prompt_tpl, resp_tpl in _REDACCION_TEMPLATES:
        for _ in range(n_copies):
            pairs.append({
                "prompt":   _fill(prompt_tpl, rng),
                "response": _fill(resp_tpl,   rng),
            })

    rng.shuffle(pairs)
    _write_jsonl(pairs, finetune_dir / "redaccion.jsonl")
    logger.info("redaccion total: %d synthetic pairs (%d templates × %d copies)",
                len(pairs), len(_REDACCION_TEMPLATES), n_copies)


# ── Adapter: dialogo ───────────────────────────────────────────────────────────

def download_dialogo(finetune_dir: Path, max_ex: int) -> None:
    """
    OpenAssistant multi-turn Spanish conversations.
    Each training example contains the full conversation history as the
    prompt and the next assistant turn as the response.
    """
    ds = _load_hf("OpenAssistant/oasst1", streaming=False)
    if ds is None:
        logger.warning("Could not load OpenAssistant — skipping dialogo")
        return

    by_id: dict[str, dict] = {}
    for row in ds:
        by_id[row["message_id"]] = row

    pairs: list[dict] = []

    def _collect_chain(node_id: str, history: list[str]) -> None:
        if max_ex and len(pairs) >= max_ex:
            return
        row = by_id.get(node_id)
        if row is None or row.get("lang") != "es":
            return

        text = row["text"].strip()
        role = row.get("role", "")

        if role == "assistant" and len(history) >= 1:
            # Format: alternate Human/Asistente turns
            prompt_parts = []
            for i, h in enumerate(history):
                prefix = "Human:" if i % 2 == 0 else "Asistente:"
                prompt_parts.append(f"{prefix} {h}")
            pairs.append({
                "prompt":   "\n".join(prompt_parts),
                "response": text,
            })

        new_history = history + [text]
        replies = [
            r["message_id"] for r in by_id.values()
            if r.get("parent_id") == node_id
        ]
        for child_id in replies:
            _collect_chain(child_id, new_history)

    # Start from root prompter nodes in Spanish
    roots = [
        mid for mid, row in by_id.items()
        if row.get("role") == "prompter"
        and row.get("parent_id") is None
        and row.get("lang") == "es"
    ]
    for root_id in roots:
        if max_ex and len(pairs) >= max_ex:
            break
        _collect_chain(root_id, [])

    _write_jsonl(pairs, finetune_dir / "dialogo.jsonl")
    logger.info("dialogo total: %d pairs", len(pairs))


# ── Adapter: razonamiento ──────────────────────────────────────────────────────

_REASON_TEMPLATES = [
    "Resuelve el siguiente problema paso a paso:\n\n{question}",
    "Explica tu razonamiento y da la respuesta:\n\n{question}",
    "Piensa detenidamente y responde:\n\n{question}",
]


def download_razonamiento(finetune_dir: Path, max_ex: int) -> None:
    """MGSM Spanish + translated GSM8K → chain-of-thought reasoning."""
    pairs: list[dict] = []

    sources = [
        ("juletxara/mgsm_es", None,  "question", "answer"),
        ("mgsm",              "es",  "question", "answer"),
        ("gsm8k",             "main","question", "answer"),   # fallback: English
    ]

    for ds_id, cfg, q_f, a_f in sources:
        ds = _load_hf(ds_id, config=cfg, streaming=True)
        if ds is None:
            continue
        count = 0
        for idx, row in enumerate(ds):
            if max_ex and count >= max_ex:
                break
            question = (row.get(q_f) or "").strip()
            answer   = (row.get(a_f) or "").strip()
            if not question or not answer:
                continue
            template = _REASON_TEMPLATES[idx % len(_REASON_TEMPLATES)]
            pairs.append({"prompt": template.format(question=question), "response": answer})
            count += 1
        logger.info("  %s (config=%s): %d examples", ds_id, cfg, count)
        if count > 0 and (not max_ex or len(pairs) >= max_ex):
            break

    _write_jsonl(pairs, finetune_dir / "razonamiento.jsonl")
    logger.info("razonamiento total: %d pairs", len(pairs))


# ── Adapter: traduccion ────────────────────────────────────────────────────────

_TRANS_TEMPLATES_ES_EN = [
    "Traduce al inglés:\n\n{source}",
    "Translate the following Spanish text to English:\n\n{source}",
    "Versión en inglés del siguiente texto en español:\n\n{source}",
]
_TRANS_TEMPLATES_EN_ES = [
    "Traduce al español:\n\n{source}",
    "Translate the following English text to Spanish:\n\n{source}",
    "Versión en español del siguiente texto en inglés:\n\n{source}",
]
_TRANS_TEMPLATES_ES_CA = [
    "Tradueix al català:\n\n{source}",
    "Traduce al catalán:\n\n{source}",
    "Versió en català del text en castellà:\n\n{source}",
]


def download_traduccion(finetune_dir: Path, max_ex: int) -> None:
    """OPUS-100 + Europarl Spanish↔English/Catalan translation pairs."""
    pairs: list[dict] = []
    per_src = max_ex // 3 if max_ex else 0

    # ── Spanish ↔ English (OPUS-100) ──
    for ds_id, cfg in [("Helsinki-NLP/opus-100", "es-en"),
                       ("opus_books",            "es-en")]:
        ds = _load_hf(ds_id, config=cfg, streaming=True)
        if ds is None:
            continue
        count = 0
        for idx, row in enumerate(ds):
            if per_src and count >= per_src:
                break
            trans = row.get("translation") or {}
            es = (trans.get("es") or row.get("es") or "").strip()
            en = (trans.get("en") or row.get("en") or "").strip()
            if not es or not en or len(es) < 20:
                continue
            # Add both directions
            tpl_es = _TRANS_TEMPLATES_ES_EN[idx % len(_TRANS_TEMPLATES_ES_EN)]
            tpl_en = _TRANS_TEMPLATES_EN_ES[idx % len(_TRANS_TEMPLATES_EN_ES)]
            pairs.append({"prompt": tpl_es.format(source=es), "response": en})
            pairs.append({"prompt": tpl_en.format(source=en), "response": es})
            count += 2
        logger.info("  %s (es-en): %d examples", ds_id, count)
        if count > 0:
            break

    # ── Spanish ↔ Catalan (OPUS-100) ──
    for ds_id, cfg in [("Helsinki-NLP/opus-100", "ca-es"),
                       ("Helsinki-NLP/opus-100", "es-ca")]:
        ds = _load_hf(ds_id, config=cfg, streaming=True)
        if ds is None:
            continue
        count = 0
        for idx, row in enumerate(ds):
            if per_src and count >= per_src:
                break
            trans = row.get("translation") or {}
            es = (trans.get("es") or "").strip()
            ca = (trans.get("ca") or "").strip()
            if not es or not ca or len(es) < 20:
                continue
            tpl = _TRANS_TEMPLATES_ES_CA[idx % len(_TRANS_TEMPLATES_ES_CA)]
            pairs.append({"prompt": tpl.format(source=es), "response": ca})
            count += 1
        logger.info("  %s (ca-es): %d examples", ds_id, count)
        if count > 0:
            break

    _write_jsonl(pairs, finetune_dir / "traduccion.jsonl")
    logger.info("traduccion total: %d pairs", len(pairs))


# ── CLI ────────────────────────────────────────────────────────────────────────

ALL_SOURCES = ["instruccion", "qa", "extraccion", "redaccion",
               "dialogo", "razonamiento", "traduccion"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=ALL_SOURCES + ["all"],
        default="all",
        help="Adapter to prepare (default: all)",
    )
    parser.add_argument(
        "--finetune-dir", default="data/finetune",
        help="Output directory for JSONL files (default: data/finetune)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=0,
        help="Max examples per source, 0=unlimited (default: 0)",
    )
    args = parser.parse_args()

    ftdir = Path(args.finetune_dir)
    ftdir.mkdir(parents=True, exist_ok=True)

    sources = ALL_SOURCES if args.source == "all" else [args.source]

    for src in sources:
        logger.info("=" * 60)
        logger.info("Adapter: %s", src)
        logger.info("=" * 60)
        fn = {
            "instruccion": download_instruccion,
            "qa":          download_qa,
            "extraccion":  download_extraccion,
            "redaccion":   download_redaccion,
            "dialogo":     download_dialogo,
            "razonamiento":download_razonamiento,
            "traduccion":  download_traduccion,
        }[src]
        if src == "redaccion":
            fn(ftdir, args.max_examples)
        else:
            fn(ftdir, args.max_examples)

    logger.info("=" * 60)
    logger.info("All done → %s", ftdir)
    logger.info("")
    logger.info("Fine-tune any adapter:")
    logger.info("  python scripts/lora_finetune.py \\")
    logger.info("      --base-ckpt checkpoints/axion_large_legal/soup_uniform.pkl \\")
    logger.info("      --preset    large \\")
    logger.info("      --data      %s/<adapter>.jsonl \\", ftdir)
    logger.info("      --specialty <adapter> \\")
    logger.info("      --output    checkpoints/lora/large_<adapter> \\")
    logger.info("      --steps 2000 --rank 16 --lora-alpha 32 --dtype bf16")


if __name__ == "__main__":
    main()
