#!/usr/bin/env python3
"""Download Spanish legal corpus for LLM specialization in derecho español.

Sources:
  git         — GitHub repos: legalize-es (12k norms), leyabierta (88k reforms),
                hpalacio/leyes (Constitución)
  boe-tc      — Tribunal Constitucional sentencias via BOE open-data API
                (official, rate-limited ~1 req/s, 1981–present, ~4k docs)
  multi-eurlex — EU multilingual legislation in Spanish (HuggingFace)
  jrc-acquis  — EU community law Spanish translations (HuggingFace)
  all         — All sources sequentially

Usage:
    # Everything (first run — takes 3–5 h for boe-tc)
    python scripts/download_legal_corpus.py --source all \\
        --output data/raw/legal/

    # Git repos only (fast, ~2 min)
    python scripts/download_legal_corpus.py --source git \\
        --output data/raw/legal/

    # TC sentencias — limit to last 20 years to save time
    python scripts/download_legal_corpus.py --source boe-tc \\
        --output data/raw/legal/ --year-from 2000

    # HuggingFace datasets
    python scripts/download_legal_corpus.py --source multi-eurlex \\
        --output data/raw/legal/
    python scripts/download_legal_corpus.py --source jrc-acquis \\
        --output data/raw/legal/

    # Tokenize everything afterwards
    python scripts/prepare_corpus.py \\
        --input  data/raw/legal/ \\
        --output data/tokenized/legal/ \\
        --extensions .md .txt .adoc
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("legal")

# ── Constants ──────────────────────────────────────────────────────────────────

GIT_REPOS = [
    {
        "name": "legalize-es",
        "url": "https://github.com/gmarko/legalize-es",
        "desc": "12,235 Spanish laws (BOE, Markdown+YAML)",
    },
    {
        "name": "leyabierta",
        "url": "https://github.com/leyabierta/leyes",
        "desc": "Consolidated Spanish legislation with 88k reform commits",
    },
    {
        "name": "constitucion",
        "url": "https://github.com/hpalacio/leyes",
        "desc": "Spanish Constitution 1978 (AsciiDoc)",
    },
]

BOE_SUMMARY_URL = "https://www.boe.es/datosabiertos/api/sumario/{date}"
BOE_XML_URL     = "https://www.boe.es/diario_boe/xml.php?id={doc_id}"
BOE_TC_KEYWORDS = {"tribunal constitucional", "sentencia", "auto del tc", "providencia"}
BOE_START_YEAR  = 1981   # TC created 1980, first BOE publications 1981

REQUEST_DELAY   = 1.2    # seconds between BOE API calls


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 4, timeout: int = 30) -> bytes | None:
    """HTTP GET with exponential backoff. Returns bytes or None on failure."""
    import urllib.request
    import urllib.error
    delay = 2
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            logger.debug("HTTP %d for %s (attempt %d)", e.code, url, attempt + 1)
        except Exception as e:
            logger.debug("Error fetching %s: %s (attempt %d)", url, e, attempt + 1)
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    return None


def _xml_text(xml_bytes: bytes) -> str:
    """Extract visible text from BOE XML document."""
    try:
        root = ET.fromstring(xml_bytes)
        # BOE XML: <documento><texto>...</texto></documento>
        texto = root.find(".//texto")
        if texto is None:
            texto = root
        parts = []
        for elem in texto.iter():
            if elem.text:
                parts.append(elem.text.strip())
            if elem.tail:
                parts.append(elem.tail.strip())
        text = "\n".join(p for p in parts if p)
        # Collapse excessive blank lines
        return re.sub(r"\n{3,}", "\n\n", text).strip()
    except ET.ParseError:
        # Fall back to regex strip if XML is malformed
        text = xml_bytes.decode("utf-8", errors="replace")
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s{3,}", "\n\n", text).strip()


def _iter_dates(year_from: int, year_to: int) -> Iterator[date]:
    """Yield weekdays (Mon–Fri) from year_from to year_to inclusive."""
    current = date(year_from, 1, 2)  # BOE published weekdays
    end = date(year_to, 12, 31)
    while current <= end:
        if current.weekday() < 5:  # Mon=0 … Fri=4
            yield current
        current += timedelta(days=1)


# ── Source: git repos ─────────────────────────────────────────────────────────

def download_git(output_dir: Path) -> None:
    for repo in GIT_REPOS:
        dest = output_dir / repo["name"]
        if dest.exists():
            logger.info("%-20s already exists — pulling latest", repo["name"])
            subprocess.run(
                ["git", "-C", str(dest), "pull", "--depth=1", "--ff-only"],
                check=False,
            )
        else:
            logger.info("Cloning %-20s — %s", repo["name"], repo["desc"])
            subprocess.run(
                ["git", "clone", "--depth=1", repo["url"], str(dest)],
                check=True,
            )
        logger.info("  → %s", dest)


# ── Source: BOE TC sentencias ─────────────────────────────────────────────────

def download_boe_tc(output_dir: Path, year_from: int, year_to: int) -> None:
    out = output_dir / "tc-sentencias"
    out.mkdir(parents=True, exist_ok=True)

    # Track already-downloaded IDs to support resumption
    seen_file = out / ".downloaded_ids.json"
    seen: set[str] = set(json.loads(seen_file.read_text()) if seen_file.exists() else [])

    total_days = sum(1 for _ in _iter_dates(year_from, year_to))
    logger.info("BOE TC: scanning %d weekdays (%d–%d) …", total_days, year_from, year_to)
    logger.info("Output: %s | Already downloaded: %d docs", out, len(seen))

    found = 0
    checked = 0

    for day in _iter_dates(year_from, year_to):
        checked += 1
        date_str = day.strftime("%Y%m%d")
        url = BOE_SUMMARY_URL.format(date=date_str)

        raw = _get(url)
        time.sleep(REQUEST_DELAY)
        if not raw:
            continue

        # Parse summary — try JSON first, then XML
        tc_ids: list[str] = []
        try:
            summary = json.loads(raw)
            # JSON structure: {"data": {"sumario": {"diario": [...]}}}
            # Items have "id" and "departamento" fields
            items = _walk_json(summary)
            for item in items:
                dept = str(item.get("departamento", "")).lower()
                titulo = str(item.get("titulo", "")).lower()
                if "tribunal constitucional" in dept or any(k in titulo for k in BOE_TC_KEYWORDS):
                    doc_id = item.get("id", "")
                    if doc_id:
                        tc_ids.append(doc_id)
        except (json.JSONDecodeError, KeyError):
            # Try XML summary
            try:
                root = ET.fromstring(raw)
                for item in root.iter("item"):
                    dept = (item.findtext("departamento") or "").lower()
                    titulo = (item.findtext("titulo") or "").lower()
                    if "tribunal constitucional" in dept or any(k in titulo for k in BOE_TC_KEYWORDS):
                        doc_id = item.findtext("id") or ""
                        if doc_id:
                            tc_ids.append(doc_id)
            except ET.ParseError:
                pass

        for doc_id in tc_ids:
            if doc_id in seen:
                continue

            xml_bytes = _get(BOE_XML_URL.format(doc_id=doc_id))
            time.sleep(REQUEST_DELAY)
            if not xml_bytes:
                continue

            text = _xml_text(xml_bytes)
            if len(text) < 200:
                continue

            out_file = out / f"{doc_id}.txt"
            out_file.write_text(text, encoding="utf-8")
            seen.add(doc_id)
            found += 1

        if checked % 500 == 0:
            pct = 100 * checked / total_days
            logger.info("  %d/%d days checked (%.0f%%) | %d TC docs downloaded",
                        checked, total_days, pct, len(seen))
            seen_file.write_text(json.dumps(list(seen)), encoding="utf-8")

    seen_file.write_text(json.dumps(list(seen)), encoding="utf-8")
    logger.info("BOE TC complete: %d new docs (total %d) → %s", found, len(seen), out)


def _walk_json(obj, depth: int = 0) -> Iterator[dict]:
    """Recursively yield dict leaves that look like BOE items (have 'id' key)."""
    if depth > 10:
        return
    if isinstance(obj, dict):
        if "id" in obj and isinstance(obj["id"], str) and obj["id"].startswith("BOE"):
            yield obj
        for v in obj.values():
            yield from _walk_json(v, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_json(item, depth + 1)


# ── Source: Multi-EURLEX ──────────────────────────────────────────────────────

def download_multi_eurlex(output_dir: Path) -> None:
    try:
        import datasets as hf
    except ImportError:
        logger.error("pip install datasets huggingface_hub")
        return

    out = output_dir / "multi-eurlex"
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Multi-EURLEX (Spanish) …")
    try:
        # level=1 = highest granularity, es = Spanish
        ds = hf.load_dataset("multi_eurlex", "es", split="train", streaming=False)
    except Exception as e:
        logger.error("Failed to load multi_eurlex: %s", e)
        return

    shard_idx = 0
    buf: list[str] = []
    buf_chars = 0
    total = 0

    for row in ds:
        text = row.get("text") or row.get("texts", {}).get("es", "")
        if not text or len(text) < 100:
            continue
        buf.append(text.strip())
        buf_chars += len(text)
        total += 1

        if buf_chars >= 50_000_000:  # ~50 MB per shard
            _write_shard(buf, out, shard_idx)
            shard_idx += 1
            buf, buf_chars = [], 0

    if buf:
        _write_shard(buf, out, shard_idx)
        shard_idx += 1

    logger.info("Multi-EURLEX: %d documents → %d shards in %s", total, shard_idx, out)


# ── Source: JRC-Acquis ────────────────────────────────────────────────────────

def download_jrc_acquis(output_dir: Path) -> None:
    try:
        import datasets as hf
    except ImportError:
        logger.error("pip install datasets huggingface_hub")
        return

    out = output_dir / "jrc-acquis"
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading JRC-Acquis (Spanish) …")

    # Try multiple known dataset IDs for JRC-Acquis Spanish
    candidates = [
        ("jrc_acquis", "es"),
        ("Helsinki-NLP/jrc_acquis", "es"),
        ("opus_books", "es-en"),
    ]

    loaded = False
    for path, config in candidates:
        try:
            ds = hf.load_dataset(path, config, split="train", streaming=True)
            logger.info("Loaded %s/%s", path, config)
            loaded = True
        except Exception:
            continue

        shard_idx = 0
        buf: list[str] = []
        buf_chars = 0
        total = 0

        text_fields = ["text", "translation", "sentence"]
        for row in ds:
            text = ""
            for field in text_fields:
                val = row.get(field)
                if isinstance(val, str) and len(val) > 50:
                    text = val
                    break
                if isinstance(val, dict):
                    text = val.get("es") or val.get("es-ES") or ""
                    if text:
                        break
            if not text:
                continue
            buf.append(text.strip())
            buf_chars += len(text)
            total += 1

            if buf_chars >= 50_000_000:
                _write_shard(buf, out, shard_idx)
                shard_idx += 1
                buf, buf_chars = [], 0

        if buf:
            _write_shard(buf, out, shard_idx)
            shard_idx += 1

        logger.info("JRC-Acquis: %d docs → %d shards in %s", total, shard_idx, out)
        break

    if not loaded:
        logger.warning("Could not load JRC-Acquis — skipping")


# ── Shared ─────────────────────────────────────────────────────────────────────

def _write_shard(texts: list[str], out_dir: Path, idx: int) -> None:
    path = out_dir / f"shard_{idx:05d}.txt"
    path.write_text("\n\n".join(texts), encoding="utf-8")
    mb = path.stat().st_size / 1e6
    logger.info("  Shard %05d → %.1f MB (%d docs)", idx, mb, len(texts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source",
                        choices=["git", "boe-tc", "multi-eurlex", "jrc-acquis", "all"],
                        default="all",
                        help="Which source(s) to download (default: all)")
    parser.add_argument("--output", default="data/raw/legal",
                        help="Root output directory (default: data/raw/legal)")
    parser.add_argument("--year-from", type=int, default=BOE_START_YEAR,
                        help=f"First year for BOE-TC scan (default: {BOE_START_YEAR})")
    parser.add_argument("--year-to", type=int, default=date.today().year,
                        help="Last year for BOE-TC scan (default: current year)")

    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    sources = (
        ["git", "boe-tc", "multi-eurlex", "jrc-acquis"]
        if args.source == "all"
        else [args.source]
    )

    for src in sources:
        logger.info("=" * 60)
        logger.info("Source: %s", src)
        logger.info("=" * 60)
        if src == "git":
            download_git(out)
        elif src == "boe-tc":
            download_boe_tc(out, args.year_from, args.year_to)
        elif src == "multi-eurlex":
            download_multi_eurlex(out)
        elif src == "jrc-acquis":
            download_jrc_acquis(out)

    logger.info("=" * 60)
    logger.info("All done → %s", out)
    logger.info("Next step:")
    logger.info("  python scripts/prepare_corpus.py \\")
    logger.info("      --input  %s \\", out)
    logger.info("      --output data/tokenized/legal/ \\")
    logger.info("      --extensions .md .txt .adoc")


if __name__ == "__main__":
    main()
