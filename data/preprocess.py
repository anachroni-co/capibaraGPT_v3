"""Capibara Slim — corpus preprocessor.

Merges downloaded shard files from multiple datasets into a single
train / validation / test split ready for SlimDataset.

Usage:
    # Merge all shards from one or more dataset dirs into one file
    python -m data.preprocess merge \\
        data/raw/corpusnos data/raw/culturax-gl data/raw/gigaverbo \\
        --output data/processed/gl_pt_combined.txt

    # Create train / val split (99% / 1%)
    python -m data.preprocess split data/processed/gl_pt_combined.txt \\
        --output-dir data/processed --val-ratio 0.01

    # Full pipeline shortcut: merge → split in one command
    python -m data.preprocess pipeline \\
        --datasets data/raw/corpusnos,data/raw/gigaverbo \\
        --output-dir data/processed \\
        --val-ratio 0.01

Output:
    data/processed/
      train.txt
      val.txt
      stats.json   {total_docs, total_chars, train_docs, val_docs}
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shard reader
# ---------------------------------------------------------------------------

def iter_shards(dataset_dir: Path) -> Iterator[str]:
    """Yield raw text of every shard file in a dataset directory."""
    shards = sorted(dataset_dir.glob("shard_*.txt"))
    if not shards:
        logger.warning("No shard files found in %s", dataset_dir)
        return
    for shard in shards:
        yield shard.read_text(encoding="utf-8")


def iter_docs(dataset_dir: Path) -> Iterator[str]:
    """Yield individual documents (double-newline separated) from all shards."""
    for shard_text in iter_shards(dataset_dir):
        for doc in shard_text.split("\n\n"):
            doc = doc.strip()
            if doc:
                yield doc


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_datasets(
    dataset_dirs: list[Path],
    output_file: Path,
    shuffle: bool = True,
    seed: int = 42,
) -> dict:
    """Merge shard files from multiple dataset directories into one text file.

    Documents are separated by a double newline, matching the format
    expected by SlimDataset.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Collect all documents into a list (memory-bound for large corpora;
    # for multi-TB training use an external shuffle tool like terashuf)
    docs: list[str] = []
    for d in dataset_dirs:
        before = len(docs)
        for doc in iter_docs(d):
            docs.append(doc)
        logger.info("%s: %d docs", d.name, len(docs) - before)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(docs)
        logger.info("Shuffled %d total documents (seed=%d)", len(docs), seed)

    total_chars = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc)
            f.write("\n\n")
            total_chars += len(doc) + 2

    stats = {
        "total_docs": len(docs),
        "total_chars": total_chars,
        "datasets": [str(d) for d in dataset_dirs],
        "output": str(output_file),
        "shuffled": shuffle,
    }
    logger.info(
        "Merged %d docs → %s (%.1f MB)",
        len(docs), output_file, total_chars / 1e6,
    )
    return stats


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_train_val(
    input_file: Path,
    output_dir: Path,
    val_ratio: float = 0.01,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> dict:
    """Split a merged text file into train / val (/ test) sets.

    Splits at document boundaries (double newlines).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    text = input_file.read_text(encoding="utf-8")
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]

    rng = random.Random(seed)
    rng.shuffle(docs)

    n = len(docs)
    n_val = max(1, int(n * val_ratio))
    n_test = max(0, int(n * test_ratio)) if test_ratio > 0 else 0
    n_train = n - n_val - n_test

    splits = {
        "train": docs[:n_train],
        "val": docs[n_train: n_train + n_val],
    }
    if n_test:
        splits["test"] = docs[n_train + n_val:]

    stats: dict = {"total_docs": n}
    for split_name, split_docs in splits.items():
        out = output_dir / f"{split_name}.txt"
        with open(out, "w", encoding="utf-8") as f:
            for doc in split_docs:
                f.write(doc)
                f.write("\n\n")
        stats[f"{split_name}_docs"] = len(split_docs)
        stats[f"{split_name}_chars"] = sum(len(d) for d in split_docs)
        logger.info("%-6s: %6d docs → %s", split_name, len(split_docs), out)

    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Stats saved to %s", stats_path)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_merge(args) -> None:
    dirs = [Path(d) for d in args.dirs]
    for d in dirs:
        if not d.exists():
            print(f"Directory not found: {d}", file=sys.stderr)
            sys.exit(1)
    merge_datasets(dirs, Path(args.output), shuffle=not args.no_shuffle)


def _cmd_split(args) -> None:
    inp = Path(args.input)
    if not inp.exists():
        print(f"File not found: {inp}", file=sys.stderr)
        sys.exit(1)
    split_train_val(
        inp,
        Path(args.output_dir),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


def _cmd_pipeline(args) -> None:
    dirs = [Path(d.strip()) for d in args.datasets.split(",")]
    out_dir = Path(args.output_dir)
    merged = out_dir / "merged.txt"
    merge_datasets(dirs, merged, shuffle=not args.no_shuffle)
    split_train_val(
        merged,
        out_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m data.preprocess",
        description="Capibara Slim — corpus preprocessor",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # merge
    m = sub.add_parser("merge", help="Merge shard dirs into one text file")
    m.add_argument("dirs", nargs="+", help="Dataset directories (data/raw/<id>)")
    m.add_argument("--output", required=True, help="Output .txt file")
    m.add_argument("--no-shuffle", action="store_true", help="Skip document shuffle")

    # split
    s = sub.add_parser("split", help="Split a merged file into train/val(/test)")
    s.add_argument("input", help="Merged .txt file")
    s.add_argument("--output-dir", required=True, help="Output directory")
    s.add_argument("--val-ratio", type=float, default=0.01)
    s.add_argument("--test-ratio", type=float, default=0.0)

    # pipeline (merge + split)
    pipe = sub.add_parser("pipeline", help="Merge + split in one step")
    pipe.add_argument("--datasets", required=True,
                      help="Comma-separated list of dataset dirs")
    pipe.add_argument("--output-dir", required=True)
    pipe.add_argument("--val-ratio", type=float, default=0.01)
    pipe.add_argument("--test-ratio", type=float, default=0.0)
    pipe.add_argument("--no-shuffle", action="store_true")

    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    p = _build_parser()
    args = p.parse_args(argv)

    if args.command == "merge":
        _cmd_merge(args)
    elif args.command == "split":
        _cmd_split(args)
    elif args.command == "pipeline":
        _cmd_pipeline(args)


if __name__ == "__main__":
    main()
