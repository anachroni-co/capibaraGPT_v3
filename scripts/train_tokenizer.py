#!/usr/bin/env python3
"""Train a SentencePiece BPE tokenizer on the legal corpus.

Replaces the byte-level tokenizer (vocab=512) with a BPE model (vocab=32000).
Run this ONCE before (re)tokenizing the corpus and before training any model.

Benefits vs byte-level:
  - ~4x more text fits in the same seq_len window (critical for legal docs)
  - Spanish legal terms tokenized as single/few tokens instead of char-by-char
  - Special tokens for tool calls and think tags at zero sequence cost

Special token layout:
  ID   Token        Purpose
  ---  -----------  -----------------------------------------------
   0   <pad>        padding
   1   <unk>        unknown piece
   2   <bos>        beginning of sequence
   3   <eos>        end of sequence
   4   <tool>       open tool call block   (was 0xFF + b"TOOL:")
   5   </tool>      close tool call block
   6   <result>     open tool result block
   7   </result>    close tool result block
   8   <think>      open chain-of-thought block
   9   </think>     close chain-of-thought block
  10   <sep>        segment separator (e.g. query | context)
  ...  [BPE pieces] ids 32 onward (sentencepiece reserves 10-31 as well)

Usage:
    # Step 1 — train (1-2 days on full corpus, ~30 min on 5M-line sample)
    python scripts/train_tokenizer.py \\
        --input-dir data/raw/legal/ \\
        --output    tokenizer/ \\
        --vocab-size 32000

    # Step 2 — quick smoke test
    python scripts/train_tokenizer.py --test tokenizer/capibara_legal.model

    # Step 3 — re-tokenize corpus with the new model
    python scripts/prepare_corpus.py \\
        --input  data/raw/legal/ \\
        --output data/tokenized/legal_bpe/ \\
        --tokenizer tokenizer/capibara_legal.model
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SPECIAL_TOKENS = [
    "<tool>", "</tool>",
    "<result>", "</result>",
    "<think>", "</think>",
    "<sep>",
]

# ── Tokenizer wrapper ─────────────────────────────────────────────────────────

class BPETokenizer:
    """Thin wrapper around a trained SentencePiece model.

    Exposes the same encode/decode interface used across the codebase so
    swapping byte-level → BPE requires minimal changes elsewhere.
    """

    PAD_ID   = 0
    UNK_ID   = 1
    BOS_ID   = 2
    EOS_ID   = 3
    TOOL_OPEN_ID    = 4   # <tool>
    TOOL_CLOSE_ID   = 5   # </tool>
    RESULT_OPEN_ID  = 6   # <result>
    RESULT_CLOSE_ID = 7   # </result>
    THINK_OPEN_ID   = 8   # <think>
    THINK_CLOSE_ID  = 9   # </think>
    SEP_ID          = 10  # <sep>

    def __init__(self, model_path: str | Path):
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("pip install sentencepiece")
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(model_path))
        self._vocab_size = self._sp.GetPieceSize()
        logger.info("BPETokenizer loaded: %s (vocab=%d)", model_path, self._vocab_size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = self._sp.Encode(text, out_type=int)
        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids: list[int]) -> str:
        filtered = [i for i in ids if i not in (
            self.PAD_ID, self.BOS_ID, self.EOS_ID,
            self.TOOL_OPEN_ID, self.TOOL_CLOSE_ID,
            self.RESULT_OPEN_ID, self.RESULT_CLOSE_ID,
            self.THINK_OPEN_ID, self.THINK_CLOSE_ID,
            self.SEP_ID,
        )]
        return self._sp.Decode(filtered)

    def encode_tool_call(self, json_str: str) -> list[int]:
        """Encode a tool call: <tool> {json} </tool>"""
        return [self.TOOL_OPEN_ID] + self.encode(json_str, add_bos=False, add_eos=False) + [self.TOOL_CLOSE_ID]

    def encode_tool_result(self, json_str: str) -> list[int]:
        """Encode a tool result: <result> {json} </result>"""
        return [self.RESULT_OPEN_ID] + self.encode(json_str, add_bos=False, add_eos=False) + [self.RESULT_CLOSE_ID]

    def id_to_piece(self, token_id: int) -> str:
        return self._sp.IdToPiece(token_id)

    @classmethod
    def from_dir(cls, tokenizer_dir: str | Path) -> "BPETokenizer":
        model_path = Path(tokenizer_dir) / "capibara_legal.model"
        if not model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
        return cls(model_path)


# ── Training ──────────────────────────────────────────────────────────────────

def _collect_input_file(input_dir: Path, max_lines: int, tmp_path: Path) -> int:
    """Sample lines from corpus into a single flat file for spm_train."""
    extensions = {".txt", ".md", ".adoc", ".jsonl"}
    files = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions)

    if not files:
        logger.error("No text files found in %s", input_dir)
        sys.exit(1)

    logger.info("Sampling from %d files (max_lines=%d)…", len(files), max_lines)
    lines_written = 0
    step = max(1, len(files) // 10000)  # uniform sampling across files

    import json

    with open(tmp_path, "w", encoding="utf-8") as out:
        for i, fpath in enumerate(files[::step]):
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if fpath.suffix.lower() == ".jsonl":
                for raw in text.splitlines():
                    if not raw.strip():
                        continue
                    try:
                        obj = json.loads(raw)
                        text_line = obj.get("text") or obj.get("content") or ""
                    except Exception:
                        text_line = raw
                    if text_line:
                        out.write(text_line[:2000] + "\n")
                        lines_written += 1
                        if lines_written >= max_lines:
                            break
            else:
                for line in text.splitlines():
                    if line.strip():
                        out.write(line[:2000] + "\n")
                        lines_written += 1
            if lines_written >= max_lines:
                break

    logger.info("Collected %d lines → %s (%.1f MB)",
                lines_written, tmp_path, os.path.getsize(tmp_path) / 1e6)
    return lines_written


def train_tokenizer(
    input_dir: str,
    output_dir: str,
    vocab_size: int = 32000,
    max_lines: int = 5_000_000,
    character_coverage: float = 0.9999,
    num_threads: int = 0,
) -> Path:
    try:
        import sentencepiece as spm
    except ImportError:
        logger.error("sentencepiece not installed — run: pip install sentencepiece")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_path / "capibara_legal")

    # Collect input
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    _collect_input_file(Path(input_dir), max_lines, tmp_path)

    # Build user-defined symbols string (sentencepiece wants comma-separated)
    user_defined = ",".join(SPECIAL_TOKENS)

    # spm_train parameters
    train_args = dict(
        input=str(tmp_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Reserve IDs 4-10 for our special tokens
        user_defined_symbols=user_defined,
        # Spanish legal corpus settings
        normalization_rule_name="nmt_nfkc",    # normalize without collapsing accents
        remove_extra_whitespaces=True,
        add_dummy_prefix=True,                 # ▁ prefix for word-initial pieces
        split_by_unicode_script=True,
        split_by_number=True,
        split_digits=True,
        byte_fallback=True,                    # fallback for rare chars → <0xNN>
        # Training efficiency
        input_sentence_size=max_lines,
        shuffle_input_sentence=True,
        num_threads=num_threads if num_threads > 0 else max(1, os.cpu_count() or 4),
    )

    logger.info("Starting SentencePiece BPE training (vocab=%d)…", vocab_size)
    logger.info("This takes ~30 min on a 5M-line sample, ~2 days on the full corpus.")
    logger.info("You can Ctrl+C after the model is written (training is single-pass).")

    spm.SentencePieceTrainer.Train(**{k: str(v) if not isinstance(v, (int, float, bool)) else v
                                      for k, v in train_args.items()})

    model_path = Path(f"{model_prefix}.model")
    vocab_path = Path(f"{model_prefix}.vocab")
    tmp_path.unlink(missing_ok=True)

    logger.info("Tokenizer saved: %s (%.1f MB)", model_path, os.path.getsize(model_path) / 1e6)
    logger.info("Vocabulary:      %s", vocab_path)

    # Verify special token IDs
    tok = BPETokenizer(model_path)
    logger.info("Special token verification:")
    for name, expected_id in [
        ("<pad>", 0), ("<unk>", 1), ("<bos>", 2), ("<eos>", 3),
        ("<tool>", 4), ("</tool>", 5), ("<result>", 6), ("</result>", 7),
        ("<think>", 8), ("</think>", 9), ("<sep>", 10),
    ]:
        actual_id = tok._sp.PieceToId(name)
        status = "OK" if actual_id == expected_id else f"WARN (got {actual_id})"
        logger.info("  %-15s → %d  %s", name, expected_id, status)

    return model_path


# ── Smoke test ────────────────────────────────────────────────────────────────

def run_test(model_path: str) -> None:
    tok = BPETokenizer(model_path)
    samples = [
        "El artículo 248 del Código Penal establece que cometen estafa los que...",
        "Sentencia del Tribunal Supremo núm. 234/2021, de 15 de marzo.",
        "contencioso-administrativo interdicto desahucio apremio embargo",
        "¿Cuáles son los requisitos del artículo 1.902 del Código Civil?",
        "La parte demandante solicita la nulidad del contrato de arrendamiento.",
    ]
    logger.info("Tokenizer smoke test (vocab=%d):", tok.vocab_size)
    for text in samples:
        ids = tok.encode(text, add_bos=False, add_eos=False)
        decoded = tok.decode(ids)
        logger.info("  [%3d tok] %s", len(ids), text[:70])
        assert decoded.strip() == text.strip(), f"Round-trip failed:\n  in:  {text}\n  out: {decoded}"

    # Tool call round-trip
    call_ids = tok.encode_tool_call('{"name":"get_article","ley":"CP","articulo":248}')
    logger.info("  Tool call: %d tokens (was ~50 bytes with 0xFF markers)", len(call_ids))

    logger.info("All checks passed.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")

    # train subcommand
    tr = sub.add_parser("train", help="Train the BPE tokenizer")
    tr.add_argument("--input-dir", required=True, help="Raw corpus directory")
    tr.add_argument("--output",    required=True, help="Output directory for model/vocab")
    tr.add_argument("--vocab-size",     type=int,   default=32000)
    tr.add_argument("--max-lines",      type=int,   default=5_000_000,
                    help="Lines to sample from corpus (default 5M, ~1-2 h)")
    tr.add_argument("--character-coverage", type=float, default=0.9999)
    tr.add_argument("--num-threads",    type=int,   default=4,
                    help="CPU threads for spm_train (default 4 — safe while training runs)")

    # test subcommand
    ts = sub.add_parser("test", help="Smoke-test a trained tokenizer")
    ts.add_argument("model", help="Path to .model file or tokenizer/ directory")

    # Backwards-compat: allow positional --test flag form
    parser.add_argument("--test", metavar="MODEL", help="Smoke-test shorthand")

    args = parser.parse_args()

    if args.cmd == "train":
        train_tokenizer(
            input_dir=args.input_dir,
            output_dir=args.output,
            vocab_size=args.vocab_size,
            max_lines=args.max_lines,
            character_coverage=args.character_coverage,
            num_threads=args.num_threads,
        )
    elif args.cmd == "test" or args.test:
        model_path = (args.model if args.cmd == "test" else args.test)
        if Path(model_path).is_dir():
            model_path = str(Path(model_path) / "capibara_legal.model")
        run_test(model_path)
    else:
        # Legacy positional form: python train_tokenizer.py tokenizer/
        if len(sys.argv) == 2 and Path(sys.argv[1]).exists():
            run_test(sys.argv[1])
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
