#!/usr/bin/env python3
"""Scraper for Diario Oficial de Galicia (DOG) and ATRIGA.

Downloads official Galician administrative documents for use as DAPT
corpus in cunca-gestoria training.

Sources:
  dog     — Diario Oficial de Galicia  (dog.xunta.gal)
  atriga  — Axencia Tributaria de Galicia (atriga.gal)
  xunta   — Procedementos Xunta de Galicia (sede.xunta.gal)

Output: one .txt file per document in --output directory.

Usage:
    # DOG — last 2 years of official gazette
    python scripts/download_dog.py --source dog \
        --output data/raw/dog/ --years 2

    # ATRIGA — tax guides and procedures
    python scripts/download_dog.py --source atriga \
        --output data/raw/atriga/

    # All sources
    python scripts/download_dog.py --source all \
        --output data/raw/galicia_admin/
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from html.parser import HTMLParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dog")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CunCABot/1.0; "
        "+https://github.com/anachroni-co/capibaragpt_v3)"
    ),
    "Accept-Language": "gl,es;q=0.9",
}

REQUEST_DELAY = 1.5   # seconds between requests — respectful crawling


# ── HTML text extractor ───────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, stripping tags and scripts."""

    SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside"}

    def __init__(self):
        super().__init__()
        self._skip = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            text = data.strip()
            if text:
                self.chunks.append(text)

    def get_text(self) -> str:
        return "\n".join(self.chunks)


def _fetch(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=20) as resp:
                raw = resp.read()
                encoding = resp.headers.get_content_charset() or "utf-8"
                return raw.decode(encoding, errors="replace")
        except (URLError, HTTPError) as e:
            logger.warning("Fetch error %s (attempt %d/%d): %s", url, attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = parser.get_text()
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── DOG downloader ────────────────────────────────────────────────────────────

DOG_BASE = "https://www.xunta.gal/diario-oficial-galicia"
# Portal migrado de dog.xunta.gal a xunta.gal ~2024
DOG_SUMARIO = DOG_BASE + "/portalPublico.do?data={year}{month:02d}{day:02d}&lang=gl"


def _dog_issue_urls(year: int, month: int, day: int) -> list[str]:
    """Return article URLs for a single DOG issue date."""
    url = DOG_SUMARIO.format(year=year, month=month, day=day)
    html = _fetch(url)
    if not html:
        return []
    urls = []
    # Article links in new portal format
    for match in re.finditer(
        r'href="([^"]*(?:portalPublico|publicacion)[^"]*(?:[?&](?:id|ID)=[^"&]+)[^"]*)"',
        html,
    ):
        full = match.group(1)
        if not full.startswith("http"):
            full = "https://www.xunta.gal" + full
        urls.append(full)
    # Fallback: any link inside /diario-oficial-galicia/
    if not urls:
        for match in re.finditer(r'href="(/diario-oficial-galicia/[^"#?]+)"', html):
            urls.append("https://www.xunta.gal" + match.group(1))
    return list(dict.fromkeys(urls))


def download_dog(output_dir: Path, years: int = 2) -> int:
    """Download DOG articles from the last `years` years."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    end = date.today()
    start = end - timedelta(days=365 * years)
    current = start

    while current <= end:
        if current.weekday() < 5:  # DOG published Mon-Fri
            urls = _dog_issue_urls(current.year, current.month, current.day)
            for url in urls:
                slug = re.sub(r'[^a-zA-Z0-9_-]', '_', urlparse(url).path)
                out_path = output_dir / f"dog_{slug}.txt"
                if out_path.exists():
                    current += timedelta(days=1)
                    continue
                html = _fetch(url)
                time.sleep(REQUEST_DELAY)
                if not html:
                    continue
                text = _html_to_text(html)
                if len(text.split()) < 30:
                    continue
                out_path.write_text(text, encoding="utf-8")
                saved += 1
                if saved % 50 == 0:
                    logger.info("DOG: %d articles saved...", saved)
        current += timedelta(days=1)

    logger.info("DOG download complete: %d articles -> %s", saved, output_dir)
    return saved


# ── ATRIGA downloader ─────────────────────────────────────────────────────────

ATRIGA_URLS = [
    "https://atriga.gal/tributos/irpf/",
    "https://atriga.gal/tributos/ixv/",
    "https://atriga.gal/tributos/transmisions-patrimoniais/",
    "https://atriga.gal/tributos/actos-xuridicosdocumentados/",
    "https://atriga.gal/tributos/sucesions-e-doacions/",
    "https://atriga.gal/tributos/imposto-sobre-hidrocarburos/",
    "https://atriga.gal/contribuintes/autoliquidacions/",
    "https://atriga.gal/contribuintes/devolucions/",
    "https://atriga.gal/contribuintes/aprazamentos/",
    "https://atriga.gal/contribuintes/embargos/",
    "https://atriga.gal/contribuintes/recursos/",
    "https://atriga.gal/contribuintes/catastro/",
]


def download_atriga(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    def _crawl(url: str, visited: set, depth: int = 0):
        nonlocal saved
        if depth > 2 or url in visited or not url.startswith("https://atriga.gal"):
            return
        visited.add(url)
        html = _fetch(url)
        time.sleep(REQUEST_DELAY)
        if not html:
            return
        text = _html_to_text(html)
        if len(text.split()) >= 50:
            slug = re.sub(r'[^a-zA-Z0-9_-]', '_', urlparse(url).path)[:80]
            out_path = output_dir / f"atriga_{slug}.txt"
            if not out_path.exists():
                out_path.write_text(text, encoding="utf-8")
                saved += 1
        for match in re.finditer(r'href="(https://atriga\.gal/[^"#?]+)"', html):
            _crawl(match.group(1), visited, depth + 1)

    visited: set = set()
    for seed in ATRIGA_URLS:
        _crawl(seed, visited)
        logger.info("ATRIGA: %d pages saved (after %s)...", saved, seed)

    logger.info("ATRIGA download complete: %d pages -> %s", saved, output_dir)
    return saved


# ── Xunta sede electronica downloader ─────────────────────────────────────────

XUNTA_SEDE_PROCEDEMENTOS = "https://sede.xunta.gal/detalle-procedemento"

XUNTA_PROC_IDS = [
    "PR004A", "PR004B", "PR770A",
    "FA001A", "FA002A", "FA301A", "FA302A",
    "TR301A", "TR341A", "TR820A",
    "IN201A", "IN220A",
    "SA461A",
]


def download_xunta(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for proc_id in XUNTA_PROC_IDS:
        url = f"{XUNTA_SEDE_PROCEDEMENTOS}?codigo={proc_id}"
        html = _fetch(url)
        time.sleep(REQUEST_DELAY)
        if not html:
            continue
        text = _html_to_text(html)
        if len(text.split()) < 30:
            continue
        out_path = output_dir / f"xunta_{proc_id}.txt"
        out_path.write_text(text, encoding="utf-8")
        saved += 1
    logger.info("Xunta sede: %d procedementos -> %s", saved, output_dir)
    return saved


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["dog", "atriga", "xunta", "all"],
        default="all",
        help="Source to download (default: all)",
    )
    parser.add_argument(
        "--output",
        default="data/raw/galicia_admin/",
        help="Output directory (default: data/raw/galicia_admin/)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Years of DOG history to download (default: 2)",
    )
    args = parser.parse_args()

    out = Path(args.output)
    total = 0

    if args.source in ("dog", "all"):
        logger.info("=== Downloading DOG (last %d years) ===", args.years)
        total += download_dog(out / "dog", args.years)

    if args.source in ("atriga", "all"):
        logger.info("=== Downloading ATRIGA ===")
        total += download_atriga(out / "atriga")

    if args.source in ("xunta", "all"):
        logger.info("=== Downloading Xunta sede ===")
        total += download_xunta(out / "xunta")

    logger.info("Total documents saved: %d -> %s", total, out)
    logger.info("Next: python scripts/prepare_corpus.py --input %s --output data/tokenized/galicia_admin/", out)


if __name__ == "__main__":
    main()
