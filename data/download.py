"""Capibara Slim — dataset downloader and text extractor.

Downloads any dataset registered in data/datasets_registry.yaml using the
HuggingFace `datasets` library (streaming — never loads the full dataset into
RAM) and writes plain-text shards that SlimDataset can consume directly.

Usage:
    # List available datasets
    python -m data.download list
    python -m data.download list --lang gl
    python -m data.download list --lang pt

    # Download a single dataset (streaming, writes shards)
    python -m data.download get corpusnos --output data/raw
    python -m data.download get gigaverbo --output data/raw --max-docs 500000

    # Download all datasets for one or more languages
    python -m data.download get --lang gl,pt --output data/raw --max-docs 200000

    # Show info about a specific dataset
    python -m data.download info culturax-es

Output structure:
    data/raw/
      corpusnos/
        shard_0000.txt   # ~50 000 docs per shard
        shard_0001.txt
        ...
        meta.json        # {id, docs, shards, bytes_written}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).parent / "datasets_registry.yaml"
_DOCS_PER_SHARD = 50_000
_MIN_TEXT_LEN = 50  # discard docs shorter than this (chars)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class DatasetEntry:
    id: str
    name: str
    langs: list[str]
    source: str
    hf_repo: str = ""
    hf_config: str | None = None
    hf_split: str = "train"
    text_field: str = "text"
    url: str = ""
    size_tokens: str = "?"
    license: str = "?"
    description: str = ""


def load_registry(path: Path = _REGISTRY_PATH) -> dict[str, DatasetEntry]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {
        d["id"]: DatasetEntry(**{k: v for k, v in d.items() if k != "description"}
                              | {"description": d.get("description", "").strip()})
        for d in raw["datasets"]
    }


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text(record: dict, text_field: str) -> str | None:
    """Pull the text string out of a dataset record."""
    if text_field in record:
        return record[text_field]
    # Fallback: common alternative field names
    for alt in ("content", "passage", "document", "article", "body"):
        if alt in record:
            return record[alt]
    return None


def _iter_texts(entry: DatasetEntry, max_docs: int = 0) -> Iterator[str]:
    """Stream text strings from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required.\n"
            "Install with: pip install datasets"
        ) from exc

    kwargs: dict = dict(split=entry.hf_split, streaming=True, trust_remote_code=True)
    if entry.hf_config:
        ds = load_dataset(entry.hf_repo, entry.hf_config, **kwargs)
    else:
        ds = load_dataset(entry.hf_repo, **kwargs)

    count = 0
    for record in ds:
        text = _extract_text(record, entry.text_field)
        if not text or len(text) < _MIN_TEXT_LEN:
            continue
        yield text.strip()
        count += 1
        if max_docs and count >= max_docs:
            break


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """Writes documents to numbered shard files under output_dir/<dataset_id>/."""

    def __init__(self, output_dir: Path, dataset_id: str,
                 docs_per_shard: int = _DOCS_PER_SHARD) -> None:
        self.root = output_dir / dataset_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs_per_shard = docs_per_shard
        self._shard_idx = 0
        self._doc_count = 0
        self._bytes_written = 0
        self._fh = self._open_shard()

    def _open_shard(self):
        path = self.root / f"shard_{self._shard_idx:04d}.txt"
        return open(path, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        self._fh.write(text)
        self._fh.write("\n\n")
        self._bytes_written += len(text.encode()) + 2
        self._doc_count += 1
        if self._doc_count % self.docs_per_shard == 0:
            self._fh.close()
            self._shard_idx += 1
            self._fh = self._open_shard()

    def close(self) -> dict:
        self._fh.close()
        # Remove last shard if empty
        last = self.root / f"shard_{self._shard_idx:04d}.txt"
        if last.stat().st_size == 0:
            last.unlink()
        else:
            self._shard_idx += 1
        meta = {
            "docs": self._doc_count,
            "shards": self._shard_idx,
            "bytes_written": self._bytes_written,
        }
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class DatasetDownloader:
    """Orchestrates download for one or more dataset IDs."""

    def __init__(self, output_dir: Path,
                 docs_per_shard: int = _DOCS_PER_SHARD) -> None:
        self.output_dir = output_dir
        self.docs_per_shard = docs_per_shard
        self.registry = load_registry()

    def download(self, dataset_id: str, max_docs: int = 0) -> dict:
        """Download one dataset. Returns meta dict."""
        if dataset_id not in self.registry:
            raise KeyError(
                f"Unknown dataset '{dataset_id}'. "
                f"Run 'python -m data.download list' to see available datasets."
            )
        entry = self.registry[dataset_id]
        logger.info("Downloading '%s' (%s) …", entry.name, entry.id)

        writer = ShardWriter(self.output_dir, entry.id, self.docs_per_shard)
        try:
            for i, text in enumerate(_iter_texts(entry, max_docs)):
                writer.write(text)
                if (i + 1) % 10_000 == 0:
                    logger.info("  %d docs …", i + 1)
        finally:
            meta = writer.close()

        meta["id"] = entry.id
        logger.info(
            "Done: %d docs, %d shards, %.1f MB",
            meta["docs"], meta["shards"], meta["bytes_written"] / 1e6,
        )
        return meta

    def download_by_lang(self, langs: list[str], max_docs: int = 0) -> list[dict]:
        """Download all datasets matching any of the given language codes."""
        targets = [
            e for e in self.registry.values()
            if any(l in e.langs for l in langs)
        ]
        if not targets:
            logger.warning("No datasets found for languages: %s", langs)
            return []
        results = []
        for entry in targets:
            results.append(self.download(entry.id, max_docs=max_docs))
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_list(args, registry: dict[str, DatasetEntry]) -> None:
    entries = list(registry.values())
    if args.lang:
        filter_langs = [l.strip() for l in args.lang.split(",")]
        entries = [e for e in entries
                   if any(l in e.langs for l in filter_langs)]
    if not entries:
        print("No datasets found.")
        return
    col = "{:<22} {:<14} {:<10} {}"
    print(col.format("ID", "LANGUAGES", "TOKENS", "NAME"))
    print("-" * 72)
    for e in entries:
        print(col.format(
            e.id,
            ",".join(e.langs),
            e.size_tokens,
            e.name,
        ))


def _cmd_info(args, registry: dict[str, DatasetEntry]) -> None:
    e = registry.get(args.dataset_id)
    if not e:
        print(f"Unknown dataset: {args.dataset_id}", file=sys.stderr)
        sys.exit(1)
    print(f"ID:          {e.id}")
    print(f"Name:        {e.name}")
    print(f"Languages:   {', '.join(e.langs)}")
    print(f"Source:      {e.source}")
    if e.source == "huggingface":
        cfg = e.hf_config or "(default)"
        print(f"HF repo:     {e.hf_repo}  [{cfg}]  split={e.hf_split}")
        print(f"Text field:  {e.text_field}")
    print(f"Tokens:      {e.size_tokens}")
    print(f"License:     {e.license}")
    print(f"Description: {e.description}")


def _cmd_get(args, registry: dict[str, DatasetEntry]) -> None:
    output = Path(args.output)
    downloader = DatasetDownloader(output)
    max_docs = args.max_docs or 0

    if args.lang:
        langs = [l.strip() for l in args.lang.split(",")]
        downloader.download_by_lang(langs, max_docs=max_docs)
    elif args.dataset_id:
        downloader.download(args.dataset_id, max_docs=max_docs)
    else:
        print("Specify a dataset ID or --lang.", file=sys.stderr)
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m data.download",
        description="Capibara Slim — dataset downloader",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # list
    ls = sub.add_parser("list", help="Show registered datasets")
    ls.add_argument("--lang", default="", help="Filter by language codes (e.g. gl,pt)")

    # info
    inf = sub.add_parser("info", help="Show details for one dataset")
    inf.add_argument("dataset_id")

    # get
    get = sub.add_parser("get", help="Download a dataset")
    get.add_argument("dataset_id", nargs="?", default="",
                     help="Dataset ID from registry (omit to use --lang)")
    get.add_argument("--lang", default="",
                     help="Download all datasets for these langs (e.g. gl,pt)")
    get.add_argument("--output", default="data/raw",
                     help="Output directory (default: data/raw)")
    get.add_argument("--max-docs", type=int, default=0,
                     help="Cap on documents per dataset (0 = no limit)")
    get.add_argument("--shard-size", type=int, default=_DOCS_PER_SHARD,
                     help=f"Documents per shard file (default: {_DOCS_PER_SHARD})")

    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    p = _build_parser()
    args = p.parse_args(argv)
    registry = load_registry()

    if args.command == "list":
        _cmd_list(args, registry)
    elif args.command == "info":
        _cmd_info(args, registry)
    elif args.command == "get":
        _cmd_get(args, registry)


if __name__ == "__main__":
    main()
