#!/usr/bin/env python3
"""Corpus preparation — tokenize raw text files into .npy shards.

Reads all .txt / .md / .py / .jsonl files from an input directory,
tokenizes them with the byte-level tokenizer (vocab=512), and writes
fixed-size .npy shards ready for ShardDataLoader.

Usage:
    # Local corpus → local shards
    python scripts/prepare_corpus.py \\
        --input  data/raw/ \\
        --output data/tokenized/ \\
        --shard-tokens 10_000_000

    # Then optionally upload shards to GCS:
    gsutil -m cp data/tokenized/*.npy gs://my-bucket/tokenized/

Arguments:
    --input         Directory with raw text files (recursive)
    --output        Directory for output .npy shards
    --shard-tokens  Approximate tokens per shard (default: 10M)
    --seq-len       Sequence length used during training (for stats only)
    --extensions    File extensions to include (default: .txt .md .py .jsonl)
    --max-files     Max files to process (0 = all)
    --validate      Run a quick validation pass after writing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _get_tokenizer():
    """Return a ByteLevelTokenizer instance."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.byte_level_training import ByteLevelTokenizer, ByteLevelConfig
    config = ByteLevelConfig()
    return ByteLevelTokenizer(config)


def _tokenize_text(tokenizer, text: str) -> np.ndarray:
    """Tokenize a single text string → int32 array."""
    return tokenizer.encode_with_special_tokens(
        text.encode("utf-8", errors="replace"),
        add_bos=True,
        add_eos=True,
    ).astype(np.int32)


def _tokenize_jsonl_line(tokenizer, line: str) -> np.ndarray | None:
    """Extract 'text' field from a JSONL line and tokenize it."""
    try:
        obj = json.loads(line)
        text = obj.get("text") or obj.get("content") or obj.get("body") or ""
        if not text:
            return None
        return _tokenize_text(tokenizer, text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_corpus(
    input_dir: str,
    output_dir: str,
    shard_tokens: int = 10_000_000,
    extensions: list[str] | None = None,
    max_files: int = 0,
    validate: bool = False,
    seq_len: int = 2048,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = extensions or [".txt", ".md", ".py", ".jsonl", ".rst"]

    # Gather files
    files = sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    if max_files > 0:
        files = files[:max_files]

    if not files:
        logger.error("No files found in %s with extensions %s", input_dir, extensions)
        sys.exit(1)

    logger.info("Found %d files in %s", len(files), input_dir)

    tokenizer = _get_tokenizer()
    logger.info("Tokenizer ready (vocab_size=%d)", tokenizer.vocab_size)

    # Shard state
    shard_idx = 0
    buffer: list[np.ndarray] = []
    buffer_len = 0
    total_tokens = 0
    skipped_files = 0

    def _flush_shard() -> None:
        nonlocal shard_idx, buffer, buffer_len
        if not buffer:
            return
        tokens = np.concatenate(buffer).astype(np.int32)
        shard_path = output_path / f"shard_{shard_idx:05d}.npy"
        np.save(shard_path, tokens)
        logger.info(
            "Shard %05d → %s (%d tokens, %.1f MB)",
            shard_idx, shard_path.name, len(tokens), tokens.nbytes / 1e6,
        )
        shard_idx += 1
        buffer = []
        buffer_len = 0

    for file_idx, fpath in enumerate(files):
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Cannot read %s: %s", fpath, exc)
            skipped_files += 1
            continue

        if fpath.suffix.lower() == ".jsonl":
            arrs = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                arr = _tokenize_jsonl_line(tokenizer, line)
                if arr is not None:
                    arrs.append(arr)
            if not arrs:
                skipped_files += 1
                continue
            tokens = np.concatenate(arrs)
        else:
            tokens = _tokenize_text(tokenizer, text)

        buffer.append(tokens)
        buffer_len += len(tokens)
        total_tokens += len(tokens)

        if buffer_len >= shard_tokens:
            _flush_shard()

        if (file_idx + 1) % 100 == 0:
            logger.info(
                "Progress: %d/%d files | %d tokens (%.1fM) | %d shards",
                file_idx + 1, len(files), total_tokens, total_tokens / 1e6, shard_idx,
            )

    # Final shard
    _flush_shard()

    # Summary
    logger.info("=" * 60)
    logger.info("Corpus preparation complete")
    logger.info("  Files processed : %d", len(files) - skipped_files)
    logger.info("  Files skipped   : %d", skipped_files)
    logger.info("  Total tokens    : %d (%.1fM)", total_tokens, total_tokens / 1e6)
    logger.info("  Shards written  : %d", shard_idx)
    logger.info("  Output dir      : %s", output_path)

    if shard_idx > 0:
        examples = total_tokens // (seq_len + 1)
        logger.info("  Est. examples   : %d (seq_len=%d)", examples, seq_len)

    # Optional validation
    if validate:
        logger.info("Validating shards…")
        shards = sorted(output_path.glob("shard_*.npy"))
        ok = 0
        for sp in shards:
            arr = np.load(sp)
            assert arr.dtype == np.int32, f"dtype mismatch in {sp}"
            assert arr.ndim == 1, f"expected 1-D array in {sp}"
            ok += 1
        logger.info("Validation passed: %d shards OK", ok)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  required=True, help="Raw corpus directory")
    parser.add_argument("--output", required=True, help="Output shards directory")
    parser.add_argument("--shard-tokens", type=int, default=10_000_000,
                        help="Tokens per shard (default 10M)")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Training sequence length (for stats)")
    parser.add_argument("--extensions", nargs="+",
                        default=[".txt", ".md", ".py", ".jsonl", ".rst"],
                        help="File extensions to include")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Limit number of files (0=all)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate shards after writing")
    args = parser.parse_args()

    prepare_corpus(
        input_dir=args.input,
        output_dir=args.output,
        shard_tokens=args.shard_tokens,
        extensions=args.extensions,
        max_files=args.max_files,
        validate=args.validate,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
