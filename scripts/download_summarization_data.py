#!/usr/bin/env python3
"""Download and format Spanish summarization datasets for LoRA fine-tuning.

Sources
-------
  dacsa        — DACSA (HuggingFace): millions of Spanish article+summary pairs
                 https://huggingface.co/datasets/cimec/dacsa
  mlsum        — MLSUM Spanish (HuggingFace): El País news article+summary pairs
                 https://huggingface.co/datasets/mlsum
  arxiv        — ccdv/arxiv-summarization (HuggingFace): 200k papers with abstracts
                 https://huggingface.co/datasets/ccdv/arxiv-summarization
  ebau         — EBAU/PAU reading comprehension exams (PDF directory)
                 Download PDFs manually from your regional EBAU repository,
                 then pass --ebau-dir to the extracted PDF folder.
  pqai         — PQAI Open Data patent pairs
                 https://huggingface.co/datasets/PQAI/pqai-patent-pairs (if available)
  all          — All HuggingFace sources sequentially

Output
------
Each source writes:
  data/raw/summarization/<source>/shard_NNNNN.txt   — raw text (for corpus pre-training)
  data/finetune/summarization_<source>.jsonl         — {"prompt":…,"response":…} pairs
                                                        ready for lora_finetune.py

Usage
-----
    # All HuggingFace sources at once
    python scripts/download_summarization_data.py --source all \\
        --output data/raw/summarization \\
        --finetune-dir data/finetune

    # Only DACSA (fast — recommended first run)
    python scripts/download_summarization_data.py --source dacsa \\
        --output data/raw/summarization \\
        --finetune-dir data/finetune \\
        --max-examples 50000

    # Process local EBAU PDFs
    python scripts/download_summarization_data.py --source ebau \\
        --ebau-dir ~/Downloads/ebau_pdfs \\
        --finetune-dir data/finetune

    # Tokenise raw shards for continued pre-training
    python scripts/prepare_corpus.py \\
        --input  data/raw/summarization/ \\
        --output data/tokenized/summarization/ \\
        --extensions .txt

    # Fine-tune LoRA adapter with all summarization data
    cat data/finetune/summarization_*.jsonl > data/finetune/summarization.jsonl
    python scripts/lora_finetune.py \\
        --base-ckpt checkpoints/axion_large_legal/soup_uniform.pkl \\
        --preset    large \\
        --data      data/finetune/summarization.jsonl \\
        --specialty resumen \\
        --output    checkpoints/lora/large_resumen \\
        --steps 2000 --rank 16 --lora-alpha 32 --dtype bf16
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("summ")

# ── Instruction templates ──────────────────────────────────────────────────────

# lora_finetune.py uses "\n### Respuesta:\n" as the prompt/response separator.
PROMPT_TEMPLATES = [
    "Resume el siguiente texto en español:\n\n{text}",
    "Escribe un resumen conciso del siguiente artículo:\n\n{text}",
    "Proporciona un resumen del siguiente texto:\n\n{text}",
    "¿Cuál es el resumen de este texto?\n\n{text}",
    "Sintetiza las ideas principales del siguiente texto:\n\n{text}",
]

# Rotate templates deterministically for variety
def _make_prompt(text: str, idx: int) -> str:
    template = PROMPT_TEMPLATES[idx % len(PROMPT_TEMPLATES)]
    return template.format(text=text.strip())


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _write_jsonl(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("  JSONL → %s  (%d pairs)", path, len(pairs))


def _write_shard(texts: list[str], out_dir: Path, idx: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard_{idx:05d}.txt"
    path.write_text("\n\n".join(texts), encoding="utf-8")
    mb = path.stat().st_size / 1e6
    logger.info("  Shard %05d → %.1f MB (%d docs)", idx, mb, len(texts))


def _flush(buf: list[str], out_dir: Path, shard_idx: int) -> tuple[list[str], int]:
    if buf:
        _write_shard(buf, out_dir, shard_idx)
    return [], shard_idx + 1


# ── Source: DACSA ──────────────────────────────────────────────────────────────

def download_dacsa(output_dir: Path, finetune_dir: Path, max_examples: int) -> None:
    """
    DACSA: large-scale Spanish summarization dataset (newspaper articles).
    HuggingFace: cimec/dacsa
    Each row has fields: 'body' (full article) and 'summary'.
    """
    try:
        import datasets as hf
    except ImportError:
        logger.error("pip install datasets huggingface_hub")
        return

    raw_dir = output_dir / "dacsa"
    logger.info("Downloading DACSA …")

    ds = None
    for config in ["es", None]:
        try:
            kwargs = dict(split="train", streaming=True)
            if config:
                ds = hf.load_dataset("cimec/dacsa", config, **kwargs)
            else:
                ds = hf.load_dataset("cimec/dacsa", **kwargs)
            logger.info("  Loaded cimec/dacsa (config=%s)", config)
            break
        except Exception as exc:
            logger.debug("cimec/dacsa config=%s failed: %s", config, exc)

    if ds is None:
        logger.warning("Could not load DACSA — skipping")
        return

    pairs:    list[dict] = []
    buf:      list[str]  = []
    buf_chars = 0
    shard_idx = 0
    total     = 0

    for idx, row in enumerate(ds):
        if max_examples and total >= max_examples:
            break

        body    = (row.get("body") or row.get("text") or "").strip()
        summary = (row.get("summary") or row.get("abstract") or "").strip()

        if len(body) < 200 or len(summary) < 30:
            continue

        # Truncate very long articles to avoid OOM in tokeniser
        if len(body) > 8000:
            body = body[:8000]

        pairs.append({"prompt": _make_prompt(body, total), "response": summary})
        buf.append(body + "\n\n" + summary)
        buf_chars += len(body) + len(summary)
        total += 1

        if buf_chars >= 50_000_000:
            buf, shard_idx = _flush(buf, raw_dir, shard_idx)
            buf_chars = 0

        if total % 10_000 == 0:
            logger.info("  DACSA: %d examples …", total)

    buf, shard_idx = _flush(buf, raw_dir, shard_idx)
    _write_jsonl(pairs, finetune_dir / "summarization_dacsa.jsonl")
    logger.info("DACSA: %d examples → %d shards", total, shard_idx)


# ── Source: MLSUM ──────────────────────────────────────────────────────────────

def download_mlsum(output_dir: Path, finetune_dir: Path, max_examples: int) -> None:
    """
    MLSUM Spanish: news articles from El País with journalist-validated summaries.
    HuggingFace: mlsum (config "es")
    Fields: 'text' (article), 'summary' (lead paragraph/abstract).
    """
    try:
        import datasets as hf
    except ImportError:
        logger.error("pip install datasets huggingface_hub")
        return

    raw_dir = output_dir / "mlsum"
    logger.info("Downloading MLSUM (Spanish) …")

    splits = []
    for split in ("train", "validation", "test"):
        try:
            splits.append(hf.load_dataset("mlsum", "es", split=split, streaming=False))
            logger.info("  Loaded mlsum/es split=%s", split)
        except Exception as exc:
            logger.debug("mlsum/es split=%s failed: %s", split, exc)

    if not splits:
        logger.warning("Could not load MLSUM — skipping")
        return

    pairs:    list[dict] = []
    buf:      list[str]  = []
    buf_chars = 0
    shard_idx = 0
    total     = 0

    for ds in splits:
        for idx, row in enumerate(ds):
            if max_examples and total >= max_examples:
                break

            text    = (row.get("text") or "").strip()
            summary = (row.get("summary") or "").strip()

            if len(text) < 100 or len(summary) < 20:
                continue

            if len(text) > 8000:
                text = text[:8000]

            pairs.append({"prompt": _make_prompt(text, total), "response": summary})
            buf.append(text + "\n\n" + summary)
            buf_chars += len(text) + len(summary)
            total += 1

            if buf_chars >= 50_000_000:
                buf, shard_idx = _flush(buf, raw_dir, shard_idx)
                buf_chars = 0

    buf, shard_idx = _flush(buf, raw_dir, shard_idx)
    _write_jsonl(pairs, finetune_dir / "summarization_mlsum.jsonl")
    logger.info("MLSUM: %d examples → %d shards", total, shard_idx)


# ── Source: arXiv summarization ────────────────────────────────────────────────

def download_arxiv(output_dir: Path, finetune_dir: Path, max_examples: int) -> None:
    """
    ccdv/arxiv-summarization: ~200k papers with article body + abstract.
    Fields: 'article' (full paper), 'abstract'.
    Note: papers are in English; included for structural summarization transfer.
    """
    try:
        import datasets as hf
    except ImportError:
        logger.error("pip install datasets huggingface_hub")
        return

    raw_dir = output_dir / "arxiv"
    logger.info("Downloading ccdv/arxiv-summarization …")

    try:
        ds = hf.load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
    except Exception as exc:
        logger.warning("Could not load arxiv-summarization: %s", exc)
        return

    pairs:    list[dict] = []
    buf:      list[str]  = []
    buf_chars = 0
    shard_idx = 0
    total     = 0

    for row in ds:
        if max_examples and total >= max_examples:
            break

        article  = (row.get("article") or "").strip()
        abstract = (row.get("abstract") or "").strip()

        if len(article) < 300 or len(abstract) < 50:
            continue

        # Truncate to intro+methods section (first 6000 chars) to keep context tight
        if len(article) > 6000:
            article = article[:6000]

        # Use English template for English content
        prompt = f"Summarize the following research paper:\n\n{article}"
        pairs.append({"prompt": prompt, "response": abstract})
        buf.append(article + "\n\n" + abstract)
        buf_chars += len(article) + len(abstract)
        total += 1

        if buf_chars >= 50_000_000:
            buf, shard_idx = _flush(buf, raw_dir, shard_idx)
            buf_chars = 0

        if total % 10_000 == 0:
            logger.info("  arXiv: %d examples …", total)

    buf, shard_idx = _flush(buf, raw_dir, shard_idx)
    _write_jsonl(pairs, finetune_dir / "summarization_arxiv.jsonl")
    logger.info("arXiv: %d examples → %d shards", total, shard_idx)


# ── Source: PQAI patents ───────────────────────────────────────────────────────

def download_pqai(output_dir: Path, finetune_dir: Path, max_examples: int) -> None:
    """
    PQAI Open Data: patent pairs for summarization/retrieval.
    Tries multiple known dataset IDs.
    """
    try:
        import datasets as hf
    except ImportError:
        logger.error("pip install datasets huggingface_hub")
        return

    raw_dir = output_dir / "pqai"
    logger.info("Downloading PQAI patent data …")

    ds = None
    candidates = [
        ("PQAI/pqai-patent-pairs", None),
        ("big_patent", "es"),
        ("big_patent", "y"),   # 'y' = all fields
    ]
    for path, config in candidates:
        try:
            kwargs = dict(split="train", streaming=True)
            if config:
                ds = hf.load_dataset(path, config, **kwargs)
            else:
                ds = hf.load_dataset(path, **kwargs)
            logger.info("  Loaded %s (config=%s)", path, config)
            break
        except Exception:
            continue

    if ds is None:
        logger.warning("Could not load PQAI / big_patent — skipping")
        return

    pairs:    list[dict] = []
    buf:      list[str]  = []
    buf_chars = 0
    shard_idx = 0
    total     = 0

    # big_patent rows: {'description': ..., 'abstract': ...}
    text_fields   = ["description", "claims", "text", "body"]
    summary_fields = ["abstract", "summary", "title"]

    for row in ds:
        if max_examples and total >= max_examples:
            break

        text = ""
        for f in text_fields:
            val = row.get(f, "")
            if isinstance(val, str) and len(val) > 200:
                text = val
                break

        summary = ""
        for f in summary_fields:
            val = row.get(f, "")
            if isinstance(val, str) and len(val) > 20:
                summary = val
                break

        if not text or not summary:
            continue

        if len(text) > 6000:
            text = text[:6000]

        prompt = f"Escribe el resumen (abstract) de la siguiente patente:\n\n{text}"
        pairs.append({"prompt": prompt, "response": summary})
        buf.append(text + "\n\n" + summary)
        buf_chars += len(text) + len(summary)
        total += 1

        if buf_chars >= 50_000_000:
            buf, shard_idx = _flush(buf, raw_dir, shard_idx)
            buf_chars = 0

    buf, shard_idx = _flush(buf, raw_dir, shard_idx)
    _write_jsonl(pairs, finetune_dir / "summarization_pqai.jsonl")
    logger.info("PQAI: %d examples → %d shards", total, shard_idx)


# ── Source: EBAU/PAU PDFs ──────────────────────────────────────────────────────

_EBAU_SUMMARY_RE = re.compile(
    r"(?:realiza\s+un\s+resumen|haz\s+un\s+resumen|resume\s+el\s+texto"
    r"|síntesis\s+del\s+texto|resumen\s+del\s+texto)",
    re.IGNORECASE,
)

def _extract_pdf_text(path: Path) -> str:
    """Extract plain text from a PDF using pdfminer (preferred) or pypdf2."""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(str(path))
        if text and len(text.strip()) > 100:
            return text.strip()
    except ImportError:
        pass

    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages  = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages).strip()
    except ImportError:
        pass

    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfFileReader(f)
            pages  = [reader.getPage(i).extractText() for i in range(reader.numPages)]
        return "\n".join(pages).strip()
    except ImportError:
        pass

    logger.warning("No PDF parser found. Install: pip install pdfminer.six  OR  pip install pypdf")
    return ""


def _parse_ebau_pdf(path: Path) -> list[dict]:
    """
    Heuristic extraction of text+summary pairs from an EBAU/PAU PDF.

    Typical structure:
      [Introductory reading passage — 300–600 words]
      PREGUNTAS / QUESTIONS
      1. Realiza un resumen del texto. (5 puntos)
      ...

    We treat the passage before the questions as the article, and look for
    student model answers in adjacent files named *_respuestas.pdf / *_solutions.pdf.
    If no answer file is found we create a "task" prompt without a response
    (still useful for evaluation / zero-shot prompting).
    """
    raw = _extract_pdf_text(path)
    if not raw:
        return []

    # Split at question boundary
    split_markers = re.split(
        r"\n(?:PREGUNTAS|CUESTIONES|QUESTIONS|Bloque\s+[AB])[^\n]*\n",
        raw, maxsplit=1, flags=re.IGNORECASE,
    )
    article = split_markers[0].strip()
    rest    = split_markers[1].strip() if len(split_markers) > 1 else ""

    if len(article) < 150:
        return []

    # Look for a companion answer/solutions file
    summary = ""
    for suffix in ("_respuestas", "_solucion", "_solutions", "_answers"):
        candidate = path.with_name(path.stem + suffix + ".pdf")
        if candidate.exists():
            answer_text = _extract_pdf_text(candidate)
            # Grab the paragraph that follows a "resumen" keyword
            m = re.search(
                r"(?:resumen|síntesis)[^\n]*\n(.+?)(?:\n\n|\Z)",
                answer_text, re.IGNORECASE | re.DOTALL,
            )
            if m:
                summary = m.group(1).strip()
            break

    pairs = []
    if summary:
        pairs.append({
            "prompt":   _make_prompt(article, 0),
            "response": summary,
        })
    else:
        # No answer file — emit as open prompt (useful for evaluation)
        pairs.append({
            "prompt":   _make_prompt(article, 0),
            "response": "(Sin respuesta modelo — usa para evaluación)",
        })
    return pairs


def download_ebau(ebau_dir: Path, finetune_dir: Path) -> None:
    """Process a local directory of EBAU/PAU PDF exams."""
    if not ebau_dir or not ebau_dir.exists():
        logger.error("--ebau-dir not found: %s", ebau_dir)
        return

    pdfs = sorted(ebau_dir.rglob("*.pdf"))
    # Skip answer/solution files (they are processed as companions above)
    pdfs = [p for p in pdfs if not any(
        s in p.stem.lower() for s in ("respuesta", "solucion", "solution", "answer")
    )]

    logger.info("Processing %d EBAU PDFs in %s …", len(pdfs), ebau_dir)
    all_pairs: list[dict] = []
    for pdf in pdfs:
        pairs = _parse_ebau_pdf(pdf)
        all_pairs.extend(pairs)
        logger.debug("  %s → %d pairs", pdf.name, len(pairs))

    _write_jsonl(all_pairs, finetune_dir / "summarization_ebau.jsonl")
    logger.info("EBAU: %d PDFs → %d instruction pairs", len(pdfs), len(all_pairs))


# ── Merge helper ───────────────────────────────────────────────────────────────

def merge_all(finetune_dir: Path) -> None:
    """Concatenate all summarization_*.jsonl files into summarization.jsonl."""
    parts = sorted(finetune_dir.glob("summarization_*.jsonl"))
    if not parts:
        return
    merged = finetune_dir / "summarization.jsonl"
    total  = 0
    with open(merged, "w", encoding="utf-8") as out:
        for part in parts:
            with open(part, encoding="utf-8") as inp:
                for line in inp:
                    out.write(line)
                    total += 1
    mb = merged.stat().st_size / 1e6
    logger.info("Merged → %s  (%d pairs, %.1f MB)", merged, total, mb)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["dacsa", "mlsum", "arxiv", "pqai", "ebau", "all"],
        default="all",
        help="Dataset source to download (default: all)",
    )
    parser.add_argument(
        "--output", default="data/raw/summarization",
        help="Root directory for raw text shards (default: data/raw/summarization)",
    )
    parser.add_argument(
        "--finetune-dir", default="data/finetune",
        help="Output directory for JSONL instruction pairs (default: data/finetune)",
    )
    parser.add_argument(
        "--ebau-dir", default=None,
        help="Local directory containing EBAU/PAU PDF exams (required for --source ebau)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=0,
        help="Maximum examples per HuggingFace source, 0=unlimited (default: 0)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="After downloading, merge all summarization_*.jsonl into summarization.jsonl",
    )

    args   = parser.parse_args()
    out    = Path(args.output)
    ftdir  = Path(args.finetune_dir)
    ftdir.mkdir(parents=True, exist_ok=True)

    sources = (
        ["dacsa", "mlsum", "arxiv", "pqai"]
        if args.source == "all"
        else [args.source]
    )

    for src in sources:
        logger.info("=" * 60)
        logger.info("Source: %s", src)
        logger.info("=" * 60)

        if src == "dacsa":
            download_dacsa(out, ftdir, args.max_examples)
        elif src == "mlsum":
            download_mlsum(out, ftdir, args.max_examples)
        elif src == "arxiv":
            download_arxiv(out, ftdir, args.max_examples)
        elif src == "pqai":
            download_pqai(out, ftdir, args.max_examples)
        elif src == "ebau":
            ebau_dir = Path(args.ebau_dir) if args.ebau_dir else None
            download_ebau(ebau_dir, ftdir)

    if args.merge or args.source == "all":
        logger.info("=" * 60)
        merge_all(ftdir)

    logger.info("=" * 60)
    logger.info("Done → raw shards: %s", out)
    logger.info("       JSONL pairs: %s", ftdir)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  # Tokenise for continued pre-training:")
    logger.info("  python scripts/prepare_corpus.py \\")
    logger.info("      --input  %s \\", out)
    logger.info("      --output data/tokenized/summarization/ \\")
    logger.info("      --extensions .txt")
    logger.info("")
    logger.info("  # LoRA fine-tune summarization adapter:")
    logger.info("  python scripts/lora_finetune.py \\")
    logger.info("      --base-ckpt checkpoints/axion_large_legal/soup_uniform.pkl \\")
    logger.info("      --preset    large \\")
    logger.info("      --data      %s/summarization.jsonl \\", ftdir)
    logger.info("      --specialty resumen \\")
    logger.info("      --output    checkpoints/lora/large_resumen \\")
    logger.info("      --steps 2000 --rank 16 --lora-alpha 32 --dtype bf16")


if __name__ == "__main__":
    main()
