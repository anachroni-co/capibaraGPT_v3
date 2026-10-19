#!/usr/bin/env python3
"""Extract clean text from a Trisquel website wget mirror.

wget --mirror saves pages *without* a .html extension when the server sends
content-type=text/html but the URL has no file extension (e.g. /en/forum/…).
This script handles that by detecting HTML via magic bytes rather than extension.

Language filtering:
  - By default, keeps only pages whose URL path starts with /en/ or /es/
  - Pass --all-langs to disable filtering and extract every language

Usage:
    # Extract en/es pages from a wget mirror
    python scripts/extract_trisquel.py \\
        --input  data/raw/trisquel_wget/trisquel.info \\
        --output data/raw/trisquel \\
        --langs en es

    # Extract all languages
    python scripts/extract_trisquel.py \\
        --input  data/raw/trisquel_wget/trisquel.info \\
        --output data/raw/trisquel \\
        --all-langs

    # Inspect what language directories exist
    python scripts/extract_trisquel.py \\
        --input  data/raw/trisquel_wget/trisquel.info \\
        --inspect
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trisquel")

_SHARD_SIZE = 50_000  # chars per output shard file


def _is_html(raw: bytes) -> bool:
    """Detect HTML content by magic bytes (works without .html extension)."""
    head = raw[:512].lstrip()
    return (
        head.startswith(b"<!DOCTYPE") or
        head.startswith(b"<!doctype") or
        head.startswith(b"<html") or
        head.startswith(b"<HTML") or
        b"<html" in head[:200] or
        b"<HTML" in head[:200]
    )


def _extract_text(raw: bytes) -> str:
    """Extract visible text from HTML bytes using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("beautifulsoup4 not installed: pip install beautifulsoup4 lxml")
        sys.exit(1)

    soup = BeautifulSoup(raw, "lxml")

    # Remove noise tags
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "form", "button", "meta", "link", "noscript",
                     "iframe", "aside", "advertisement"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _url_lang(filepath: Path, mirror_root: Path) -> str:
    """Return first path component of the URL (language prefix like 'en', 'es')."""
    try:
        rel = filepath.relative_to(mirror_root)
        return rel.parts[0] if rel.parts else ""
    except ValueError:
        return ""


def inspect(mirror_root: Path) -> None:
    """Print language distribution of the mirror."""
    from collections import Counter
    counts: Counter = Counter()
    for fp in mirror_root.rglob("*"):
        if not fp.is_file():
            continue
        lang = _url_lang(fp, mirror_root)
        counts[lang] += 1

    total = sum(counts.values())
    logger.info("Mirror: %s | %d files total", mirror_root, total)
    logger.info("Language distribution (top-level directory):")
    for lang, n in counts.most_common():
        logger.info("  %-12s %5d files", lang or "(root)", n)


def extract(
    mirror_root: Path,
    output_dir: Path,
    langs: list[str],
    all_langs: bool,
    min_chars: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_set = set(langs)
    total_pages = 0
    total_chars = 0
    shard_idx = 0
    shard_buf: list[str] = []
    shard_chars = 0
    skipped_lang = 0
    skipped_not_html = 0
    skipped_short = 0

    def flush_shard() -> None:
        nonlocal shard_idx, shard_buf, shard_chars
        if not shard_buf:
            return
        path = output_dir / f"shard_{shard_idx:05d}.txt"
        path.write_text("\n\n".join(shard_buf), encoding="utf-8")
        logger.info("Shard %05d → %d pages | %.1f KB", shard_idx,
                    len(shard_buf), shard_chars / 1024)
        shard_idx += 1
        shard_buf = []
        shard_chars = 0

    all_files = [fp for fp in mirror_root.rglob("*") if fp.is_file()]
    logger.info("Scanning %d files in %s …", len(all_files), mirror_root)

    for fp in sorted(all_files):
        # Language filter
        if not all_langs:
            lang = _url_lang(fp, mirror_root)
            if lang not in lang_set:
                skipped_lang += 1
                continue

        try:
            raw = fp.read_bytes()
        except OSError as e:
            logger.debug("Skip unreadable %s: %s", fp, e)
            continue

        if not _is_html(raw):
            skipped_not_html += 1
            continue

        text = _extract_text(raw)
        if len(text) < min_chars:
            skipped_short += 1
            continue

        shard_buf.append(text)
        shard_chars += len(text)
        total_pages += 1
        total_chars += len(text)

        if shard_chars >= _SHARD_SIZE * 10:  # ~500 KB per shard
            flush_shard()

    flush_shard()

    logger.info("=" * 55)
    logger.info("Done.")
    logger.info("  Pages extracted : %d", total_pages)
    logger.info("  Text size       : %.2f MB", total_chars / 1e6)
    logger.info("  Shards          : %d → %s", shard_idx, output_dir)
    logger.info("  Skipped (lang)  : %d", skipped_lang)
    logger.info("  Skipped (HTML?) : %d", skipped_not_html)
    logger.info("  Skipped (short) : %d", skipped_short)
    logger.info("=" * 55)
    if total_pages > 0:
        logger.info("Next step:")
        logger.info("  python scripts/prepare_corpus.py --input %s "
                    "--output data/tokenized/trisquel/ --extensions .txt", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Root of the wget mirror (the trisquel.info/ folder)")
    parser.add_argument("--output", default="data/raw/trisquel",
                        help="Output directory for .txt shards (default: data/raw/trisquel)")
    parser.add_argument("--langs", nargs="+", default=["en", "es"],
                        help="Language directories to include (default: en es)")
    parser.add_argument("--all-langs", action="store_true",
                        help="Extract all languages (ignores --langs)")
    parser.add_argument("--min-chars", type=int, default=200,
                        help="Discard pages with fewer visible chars (default: 200)")
    parser.add_argument("--inspect", action="store_true",
                        help="Just show language distribution and exit")
    args = parser.parse_args()

    mirror_root = Path(args.input)
    if not mirror_root.exists():
        logger.error("Mirror root not found: %s", mirror_root)
        sys.exit(1)

    if args.inspect:
        inspect(mirror_root)
        return

    extract(
        mirror_root=mirror_root,
        output_dir=Path(args.output),
        langs=args.langs,
        all_langs=args.all_langs,
        min_chars=args.min_chars,
    )


if __name__ == "__main__":
    main()
