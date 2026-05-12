#!/usr/bin/env python3
"""Generate RLM training trajectories from BOE/CENDOJ documents.

Produces ~1,000 (documento, pregunta, code, respuesta) examples for
fine-tuning Slim200M to use the RLM scaffold (arXiv:2512.24601v2).

Usage:
    python scripts/generate_rlm_trajectories.py \
        --corpus-dir data/raw/boe/ \
        --output data/finetune/rlm_trajectories.jsonl \
        --n 1000
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import textwrap
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("rlm_gen")

CHUNK_TOKENS = 1024

SYSTEM_PROMPT = """\
Eres un asistente legal español. Tienes acceso a un REPL Python con las \
siguientes variables:
  - documento: str — fragmento del documento actual
  - chunks: list[str] — todos los fragmentos del documento
  - depth: int — profundidad de recursión actual
  - sub_llm(pregunta, fragmento) -> str — llama al modelo sobre un fragmento
  - RESPUESTA_FINAL: str — asigna aquí la respuesta final cuando termines

Escribe código Python elemental (split, loops, comparaciones de strings) para \
responder la pregunta. Cuando tengas la respuesta, asigna: RESPUESTA_FINAL = "..."
No uses imports. No uses funciones externas salvo sub_llm().
"""

QUESTION_TEMPLATES = {
    "objeto": ["¿Cuál es el objeto principal de este texto legal?", "¿Qué regula esta norma?", "¿Cuál es la materia que regula este artículo?"],
    "sancion": ["¿Qué sanciones o penas establece este texto?", "¿Cuáles son las consecuencias por incumplimiento?", "¿Qué infracciones y sus sanciones se mencionan?"],
    "aplicacion": ["¿A quién se aplica esta norma?", "¿Cuál es el ámbito de aplicación?", "¿Qué sujetos están obligados por esta disposición?"],
    "fecha": ["¿Cuándo entra en vigor esta disposición?", "¿Qué plazos se mencionan en el texto?", "¿Qué fechas o períodos son relevantes?"],
    "resumen": ["Resume el contenido principal en una frase.", "¿Cuál es la idea central de este fragmento?", "Extrae los puntos clave de este texto legal."],
}

CODE_TEMPLATES = {
    "objeto": textwrap.dedent("""\
        keywords = ["objeto", "regula", "tiene por objeto", "ámbito", "materia"]
        lineas = documento.split("\\n")
        encontradas = []
        for linea in lineas:
            if any(kw in linea.lower() for kw in keywords):
                encontradas.append(linea.strip())
        RESPUESTA_FINAL = encontradas[0] if encontradas else documento.split(".")[0].strip()
        """),
    "sancion": textwrap.dedent("""\
        keywords = ["sanción", "multa", "pena", "infracción", "euros", "prisión", "inhabilitación"]
        lineas = documento.split("\\n")
        sanciones = [l.strip() for l in lineas if any(kw in l.lower() for kw in keywords)]
        RESPUESTA_FINAL = " | ".join(sanciones[:3]) if sanciones else "No se mencionan sanciones explícitas."
        """),
    "aplicacion": textwrap.dedent("""\
        keywords = ["aplicará", "aplicable", "sujetos", "personas", "quienes", "empresas", "ciudadanos"]
        lineas = documento.split("\\n")
        aplicacion = [l.strip() for l in lineas if any(kw in l.lower() for kw in keywords)]
        RESPUESTA_FINAL = aplicacion[0] if aplicacion else "Ámbito de aplicación no especificado."
        """),
    "fecha": textwrap.dedent("""\
        meses = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"]
        palabras = documento.split()
        fechas = []
        for i, p in enumerate(palabras):
            if any(mes in p.lower() for mes in meses):
                fechas.append(" ".join(palabras[max(0,i-2):min(len(palabras),i+3)]))
        RESPUESTA_FINAL = " | ".join(fechas[:3]) if fechas else "No se identifican fechas explícitas."
        """),
    "resumen": textwrap.dedent("""\
        oraciones = [s.strip() for s in documento.replace("\\n"," ").split(".") if len(s.strip()) > 20]
        if len(oraciones) >= 2:
            RESPUESTA_FINAL = oraciones[0] + ". " + oraciones[1] + "."
        elif oraciones:
            RESPUESTA_FINAL = oraciones[0] + "."
        else:
            RESPUESTA_FINAL = documento[:300].strip()
        """),
}


def _load_documents(corpus_dir: Path, doc_type: str) -> list[tuple[str, str]]:
    docs = []
    for pattern in ["*.txt", "*.json"]:
        for p in sorted(corpus_dir.glob(pattern))[:5000]:
            try:
                if pattern == "*.json":
                    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                    text = data.get("texto", data.get("text", data.get("content", "")))
                    doc_id = data.get("id", p.stem)
                else:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    doc_id = p.stem
                if len(text.split()) >= 50:
                    docs.append((doc_id, text))
            except Exception:
                continue
    logger.info("Loaded %d documents from %s", len(docs), corpus_dir)
    return docs


def _split_chunks(text: str, chunk_size: int = CHUNK_TOKENS) -> list[str]:
    words = text.split()
    chunks, buf = [], []
    for w in words:
        buf.append(w)
        if len(buf) >= chunk_size:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _extract_answer(code: str, chunk: str, chunks: list[str]) -> str:
    state: dict = {
        "documento": chunk, "chunks": chunks, "depth": 0, "RESPUESTA_FINAL": None,
        "__builtins__": {"len": len, "range": range, "str": str, "int": int,
                         "list": list, "max": max, "min": min, "any": any, "all": all,
                         "enumerate": enumerate, "zip": zip, "print": print,
                         "True": True, "False": False, "None": None},
    }
    try:
        exec(compile(code, "<gen>", "exec"), state)  # noqa: S102
    except Exception:
        pass
    return str(state.get("RESPUESTA_FINAL") or chunk[:200].strip())


def generate_trajectories(docs: list[tuple[str, str]], n: int, rng: random.Random) -> list[dict]:
    trajectories = []
    q_types = list(QUESTION_TEMPLATES.keys())
    while len(trajectories) < n and docs:
        doc_id, text = rng.choice(docs)
        chunks = _split_chunks(text)
        if not chunks:
            continue
        chunk = rng.choice(chunks)
        q_type = rng.choice(q_types)
        pregunta = rng.choice(QUESTION_TEMPLATES[q_type])
        code_template = CODE_TEMPLATES[q_type]
        metadata = f"# Documento: {len(chunks)} fragmentos | fragmento actual ({len(chunk.split())} palabras) | depth=0"
        prompt = f"{metadata}\nPregunta: {pregunta}"
        respuesta = _extract_answer(code_template, chunk, chunks)
        trajectories.append({
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "response": f"# Fragmento actual en 'documento'\n{code_template.rstrip()}",
            "doc_source": doc_id, "pregunta_type": q_type,
            "pregunta": pregunta, "respuesta_esperada": respuesta[:500],
        })
        if len(trajectories) % 100 == 0:
            logger.info("Generated %d/%d trajectories", len(trajectories), n)
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", default="data/raw/boe/")
    parser.add_argument("--output", default="data/finetune/rlm_trajectories.jsonl")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--doc-type", choices=["ley", "sentencia", "resolucion"], default="ley")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        logger.warning("Corpus dir not found: %s — using synthetic docs", corpus_dir)
        docs = [(f"doc_{i:04d}", " ".join(["Artículo", str(i), ".",
            "Esta disposición regula el régimen de sanciones aplicable.",
            "Las infracciones graves se sancionarán con multa de hasta 60.000 euros.",
            "El ámbito de aplicación comprende a todas las empresas del sector."] * 50)) for i in range(200)]
    else:
        docs = _load_documents(corpus_dir, args.doc_type)

    if not docs:
        logger.error("No documents found — aborting")
        return

    rng = random.Random(args.seed)
    trajectories = generate_trajectories(docs, args.n, rng)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    logger.info("Saved %d trajectories -> %s", len(trajectories), out_path)
    by_type: dict[str, int] = {}
    for t in trajectories:
        k = t["pregunta_type"]; by_type[k] = by_type.get(k, 0) + 1
    for k, v in sorted(by_type.items()):
        logger.info("  %-12s %d", k, v)


if __name__ == "__main__":
    main()
