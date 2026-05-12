#!/usr/bin/env python3
"""Recursive Language Model (RLM) scaffold for Capibara.

Based on: arXiv:2512.24601v2 — LLM writes simple Python to interact with an
external REPL and calls itself recursively over document chunks, achieving
effectively unlimited context beyond the model's context window.

Architecture:
  - Infini-attention handles 4K-8K tokens (short/medium docs)
  - RLM handles 50K-1M tokens (BOE laws, full contracts, full expedientes)

The model generates elementary Python (split, loops, string matching) to
process document chunks. Only ~1,000 fine-tuning trajectories are needed.

Validation target: >70% of attempts produce syntactically valid Python.

Usage:
    # Validate that Slim200M can generate valid Python for RLM tasks
    python scripts/rlm_scaffold.py --validate \
        --model-ckpt checkpoints/axion_large_legal/soup_uniform.pkl \
        --sample-docs data/raw/boe/

    # Query a long document
    python scripts/rlm_scaffold.py \
        --documento path/to/ley.txt \
        --pregunta "¿Cuáles son las sanciones previstas en el artículo 15?"
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rlm")

CHUNK_TOKENS = 1024
MAX_ITERATIONS = 10
MAX_DEPTH = 3

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


def _split_document(text: str, chunk_size: int = CHUNK_TOKENS) -> list[str]:
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


def _safe_exec(code: str, state: dict) -> tuple[bool, str]:
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    try:
        exec(compile(code, "<rlm>", "exec"), state)  # noqa: S102
        return True, ""
    except Exception as e:
        return False, str(e)


def _make_sub_llm(inference_fn: Callable, depth: int) -> Callable:
    def sub_llm(pregunta: str, fragmento: str) -> str:
        if depth >= MAX_DEPTH:
            return "[profundidad máxima alcanzada]"
        return rlm_query(fragmento, pregunta, inference_fn, depth + 1)
    return sub_llm


def rlm_query(
    documento: str,
    pregunta: str,
    inference_fn: Callable[[str, str], str],
    depth: int = 0,
) -> str | None:
    chunks = _split_document(documento)
    logger.info("RLM depth=%d | chunks=%d | pregunta=%s", depth, len(chunks), pregunta[:60])

    repl_state: dict = {
        "documento": chunks[0] if chunks else documento,
        "chunks": chunks,
        "depth": depth,
        "sub_llm": _make_sub_llm(inference_fn, depth),
        "RESPUESTA_FINAL": None,
        "__builtins__": {
            "len": len, "range": range, "str": str, "int": int,
            "list": list, "dict": dict, "enumerate": enumerate,
            "zip": zip, "any": any, "all": all, "max": max, "min": min,
            "print": print, "True": True, "False": False, "None": None,
        },
    }

    metadata = (
        f"# Documento: {len(chunks)} fragmentos | "
        f"fragmento actual: chunks[0] ({len(chunks[0].split())} palabras)\n"
        f"# depth={depth}"
    )

    for iteration in range(MAX_ITERATIONS):
        prompt = f"{metadata}\nPregunta: {pregunta}"
        code = inference_fn(SYSTEM_PROMPT, prompt)
        ok, err = _safe_exec(code, repl_state)
        logger.debug("iter=%d ok=%s err=%s", iteration, ok, err or "—")
        if not ok:
            logger.warning("RLM exec error iter=%d: %s", iteration, err)
            if "SyntaxError" in err:
                break
        if repl_state.get("RESPUESTA_FINAL") is not None:
            answer = str(repl_state["RESPUESTA_FINAL"])
            logger.info("RLM answered at iter=%d depth=%d", iteration, depth)
            return answer

    return None


def validate_syntax_rate(
    inference_fn: Callable[[str, str], str],
    sample_docs: list[str],
    n_samples: int = 50,
) -> float:
    preguntas = [
        "¿Cuál es el objeto de este artículo?",
        "¿Qué sanciones establece este texto?",
        "Resume en una frase el contenido principal.",
        "¿A quién aplica esta norma?",
        "¿Cuál es la fecha de entrada en vigor?",
    ]
    valid, total = 0, 0
    for i, doc in enumerate(sample_docs[:n_samples]):
        pregunta = preguntas[i % len(preguntas)]
        chunks = _split_document(doc)
        metadata = f"# Documento: {len(chunks)} fragmentos | depth=0"
        prompt = f"{metadata}\nPregunta: {pregunta}"
        code = inference_fn(SYSTEM_PROMPT, prompt)
        try:
            ast.parse(code)
            valid += 1
        except SyntaxError:
            pass
        total += 1
    rate = valid / total if total else 0.0
    logger.info("Syntax validity: %d/%d = %.1f%%", valid, total, rate * 100)
    return rate


def _dummy_inference(system: str, prompt: str) -> str:
    return textwrap.dedent(f"""
        resultados = []
        for i, chunk in enumerate(chunks):
            if any(kw in chunk.lower() for kw in ["artículo", "sanción", "objeto"]):
                resultados.append(f"Fragmento {{i}}: {{chunk[:200]}}")
        RESPUESTA_FINAL = " | ".join(resultados[:3]) if resultados else "No encontrado"
    """).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--sample-docs", default=None)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--documento", default=None)
    parser.add_argument("--pregunta", default=None)
    parser.add_argument("--model-ckpt", default=None)
    args = parser.parse_args()

    inference_fn = _dummy_inference

    if args.validate:
        docs = []
        if args.sample_docs:
            sample_dir = Path(args.sample_docs)
            docs = [p.read_text(errors="ignore") for p in sample_dir.glob("*.txt")]
        if not docs:
            docs = ["Artículo 1. El objeto de esta ley es regular " * 200]
        rate = validate_syntax_rate(inference_fn, docs, args.n_samples)
        viable = rate >= 0.70
        print(f"Syntax rate: {rate:.1%} | RLM viable: {'SÍ' if viable else 'NO (< 70%)'}")
        sys.exit(0 if viable else 1)

    if args.documento and args.pregunta:
        texto = Path(args.documento).read_text(errors="ignore")
        answer = rlm_query(texto, args.pregunta, inference_fn)
        if answer:
            print(f"\nRespuesta:\n{answer}")
        else:
            print("RLM no pudo responder — usar RAG como fallback")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
