#!/usr/bin/env python3
"""Download text corpus from HuggingFace for Capibara Slim training.

Sources available:
  wikipedia   — Wikipedia dumps (gl, es, en, pt, …)
  culturax    — CulturaX cleaned multilingual (gl, es, en, pt, …)  [recommended for web data]
  oscar       — OSCAR web corpus (gl, es, en, pt, …)
  c4          — Cleaned English web crawl (en only)
  books       — English book collection (Abirate/english_books)
  books_multilingual — Multilingual books (en, es, pt, fr, de, …)
  code        — Clean GitHub code (codeparrot/github-code-clean, lang=Python/JavaScript/…)

Usage:
    # Galician Wikipedia (~150 MB, fast)
    python scripts/download_corpus.py \\
        --source wikipedia --lang gl \\
        --output data/raw/gl/ --max-tokens 50_000_000

    # Galician CulturaX (~2 GB, best quality web data)
    python scripts/download_corpus.py \\
        --source culturax --lang gl \\
        --output data/raw/gl/ --max-tokens 200_000_000

    # Spanish Wikipedia
    python scripts/download_corpus.py \\
        --source wikipedia --lang es \\
        --output data/raw/es/ --max-tokens 500_000_000

    # Portuguese Wikipedia
    python scripts/download_corpus.py \\
        --source wikipedia --lang pt \\
        --output data/raw/pt/ --max-tokens 200_000_000

    # English books
    python scripts/download_corpus.py \\
        --source books --lang en \\
        --output data/raw/books/ --max-tokens 200_000_000

    # Clean Python code
    python scripts/download_corpus.py \\
        --source code --lang Python \\
        --output data/raw/code/ --max-tokens 100_000_000

Install:
    pip install datasets huggingface_hub
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Dataset configs ───────────────────────────────────────────────────────────

SOURCES: dict[str, dict] = {
    "wikipedia": {
        "path": "wikimedia/wikipedia",
        "text_field": "text",
        "config_template": "20231101.{lang}",
        "split": "train",
        "notes": "Clean encyclopedia text. Ideal starting corpus.",
    },
    "culturax": {
        "path": "uonlp/CulturaX",
        "text_field": "text",
        "config_template": "{lang}",
        "split": "train",
        "notes": "CulturaX — cleaned mC4 + OSCAR blend. Best multilingual web source.",
    },
    "oscar": {
        "path": "oscar-corpus/OSCAR-2301",
        "text_field": "content",
        "config_template": "{lang}",
        "split": "train",
        "notes": "OSCAR 23.01 — deduplicated web crawl.",
    },
    "c4": {
        "path": "allenai/c4",
        "text_field": "text",
        "config_template": "en",          # c4 only has English; lang arg ignored
        "split": "train",
        "notes": "Cleaned English C4. Parquet format (no legacy scripts). English only.",
    },
    "books": {
        "path": "Abirate/english_books",
        "text_field": "text",
        "config_template": None,           # no config needed
        "split": "train",
        "notes": "English book collection — clean prose, ideal for language modelling.",
    },
    "books_multilingual": {
        "path": "storytelling-nlp/books_corpus",
        "text_field": "text",
        "config_template": None,
        "split": "train",
        "notes": "Multilingual books corpus (en/es/pt/fr/de/…).",
    },
    "code": {
        "path": "codeparrot/github-code-clean",
        "text_field": "code",
        "config_template": "{lang}",       # lang = Python, JavaScript, Java, …
        "split": "train",
        "notes": "Clean GitHub code filtered for quality. Pass --lang Python/JavaScript/etc.",
    },
}

# Approx bytes per token for byte-level tokenizer (1 byte ≈ 1 token)
BYTES_PER_TOKEN = 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_datasets() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError:
        logger.error("datasets library not found.")
        logger.error("Install with: pip install datasets huggingface_hub")
        sys.exit(1)


def _bytes_to_human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _write_shard(texts: list[str], output_dir: Path, shard_idx: int) -> int:
    """Write a list of texts to a .txt shard. Returns total bytes written."""
    path = output_dir / f"shard_{shard_idx:05d}.txt"
    content = "\n\n".join(t.strip() for t in texts if t.strip())
    path.write_text(content, encoding="utf-8")
    return len(content.encode("utf-8"))


# ── Main download ─────────────────────────────────────────────────────────────

def download(
    source: str,
    lang: str,
    output_dir: str,
    max_tokens: int,
    shard_size_mb: int,
    hf_token: str | None,
) -> None:
    _check_datasets()
    import datasets as hf_datasets

    cfg = SOURCES[source]
    hf_path = cfg["path"]
    text_field = cfg["text_field"]
    tmpl = cfg["config_template"]
    config_name = tmpl.format(lang=lang) if tmpl else None
    split = cfg["split"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Source   : %s (%s)", source, cfg["notes"])
    logger.info("Language : %s", lang)
    logger.info("HF path  : %s%s", hf_path, f" / {config_name}" if config_name else "")
    logger.info("Max tok  : %d (%.0f MB est.)", max_tokens, max_tokens / 1e6)
    logger.info("Output   : %s", out)

    # Stream to avoid downloading the full dataset at once
    load_kwargs: dict = dict(
        split=split,
        streaming=True,
        token=hf_token,
    )
    if config_name is not None:
        load_kwargs["name"] = config_name

    try:
        ds = hf_datasets.load_dataset(hf_path, **load_kwargs)
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        logger.error("Try: huggingface-cli login  (for gated datasets)")
        sys.exit(1)

    shard_bytes_limit = shard_size_mb * 1024 * 1024
    shard_idx = 0
    shard_texts: list[str] = []
    shard_bytes = 0

    total_tokens = 0
    total_docs = 0
    total_bytes = 0

    logger.info("Downloading…")

    for doc in ds:
        text = doc.get(text_field, "")
        if not text or not text.strip():
            continue

        doc_bytes = len(text.encode("utf-8"))
        doc_tokens = int(doc_bytes * BYTES_PER_TOKEN)

        shard_texts.append(text)
        shard_bytes += doc_bytes
        total_tokens += doc_tokens
        total_docs += 1
        total_bytes += doc_bytes

        # Flush shard
        if shard_bytes >= shard_bytes_limit:
            written = _write_shard(shard_texts, out, shard_idx)
            logger.info(
                "Shard %05d → %s | docs=%d | total %.1fM tok",
                shard_idx, _bytes_to_human(written), len(shard_texts),
                total_tokens / 1e6,
            )
            shard_idx += 1
            shard_texts = []
            shard_bytes = 0

        if total_tokens >= max_tokens:
            break

    # Final shard
    if shard_texts:
        written = _write_shard(shard_texts, out, shard_idx)
        logger.info("Shard %05d → %s (final)", shard_idx, _bytes_to_human(written))
        shard_idx += 1

    logger.info("=" * 60)
    logger.info("Download complete")
    logger.info("  Documents : %d", total_docs)
    logger.info("  Tokens    : %.1fM", total_tokens / 1e6)
    logger.info("  Size      : %s", _bytes_to_human(total_bytes))
    logger.info("  Shards    : %d → %s", shard_idx, out)
    logger.info("=" * 60)
    logger.info("Next step:")
    logger.info("  python scripts/prepare_corpus.py --input %s --output data/tokenized/%s/", out, lang)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", choices=list(SOURCES), default="wikipedia",
                        help="Dataset source (default: wikipedia)")
    parser.add_argument("--lang", default="gl",
                        help="Language code: gl=Galician, es=Spanish, en=English (default: gl)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: data/raw/<lang>/)")
    parser.add_argument("--max-tokens", type=int, default=50_000_000,
                        help="Stop after this many tokens (default: 50M)")
    parser.add_argument("--shard-size-mb", type=int, default=100,
                        help="Max MB per output shard (default: 100)")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token for gated datasets (or set HF_TOKEN env var)")
    parser.add_argument("--list", action="store_true",
                        help="List available sources and exit")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable sources:")
        for name, cfg in SOURCES.items():
            print(f"  {name:20s} — {cfg['notes']}")
        print("\nExamples:")
        print("  python scripts/download_corpus.py --source wikipedia --lang gl --max-tokens 50_000_000")
        print("  python scripts/download_corpus.py --source culturax  --lang pt --max-tokens 200_000_000")
        print("  python scripts/download_corpus.py --source books     --lang en --max-tokens 200_000_000")
        print("  python scripts/download_corpus.py --source code      --lang Python --max-tokens 100_000_000")
        return

    # Default output: use source name for lang-agnostic sources
    if args.output:
        output = args.output
    elif args.source in ("books", "books_multilingual", "c4"):
        output = f"data/raw/{args.source}"
    elif args.source == "code":
        output = f"data/raw/code/{args.lang.lower()}"
    else:
        output = f"data/raw/{args.lang}"

    download(
        source=args.source,
        lang=args.lang,
        output_dir=output,
        max_tokens=args.max_tokens,
        shard_size_mb=args.shard_size_mb,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
