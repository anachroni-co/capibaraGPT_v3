"""Sequence packing utilities for CapibaraGPT v3.

Packing removes the compute wasted on padding when training with documents
shorter than ``max_seq_length`` (with short texts, pad-to-max can burn
30-50% of the compute on padding tokens).

Two strategies are provided:

1. :func:`pack_token_streams` — GPT-style contiguous packing for
   *pretraining*: documents are concatenated with an EOS separator and the
   resulting stream is cut into fixed-length blocks. Zero padding, maximum
   throughput. Attention may cross document boundaries (standard practice
   in GPT-2/3, LLaMA pretraining).

2. :func:`pack_examples` — first-fit packing for *finetuning/SFT*: whole
   examples are grouped into bins of ``seq_len`` without splitting them.
   Returns ``segment_ids`` so the model can build a block-diagonal
   attention mask (no attention across examples) and ``loss_mask`` to
   ignore padding in the loss.

Both are pure NumPy (no JAX/torch required) and stream-friendly.

Enabled from config with ``data.preprocessing.use_sequence_packing: true``
(config/config.yaml) or ``use_sequence_packing = true`` in
config/configs_toml/production/training.toml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

__all__ = [
    "pack_token_streams",
    "pack_examples",
    "PackedExample",
    "PackedDataset",
    "packing_efficiency",
]


def pack_token_streams(
    documents: Iterable[Sequence[int]],
    seq_len: int,
    eos_id: int,
    drop_last: bool = True,
    dtype: Any = np.int32,
) -> Iterator[np.ndarray]:
    """Concatenate token documents with EOS separators and yield fixed blocks.

    Args:
        documents: iterable of token-id sequences (one per document).
        seq_len: block length to emit (e.g. max_seq_length).
        eos_id: separator token appended after every document.
        drop_last: if True, the final partial block is discarded; if False,
            it is emitted unpadded (shorter than ``seq_len``).
        dtype: numpy dtype of the emitted blocks.

    Yields:
        np.ndarray of shape (seq_len,) — no padding tokens at all.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    buffer: List[int] = []
    for doc in documents:
        if len(doc) == 0:
            continue
        buffer.extend(int(t) for t in doc)
        buffer.append(int(eos_id))
        while len(buffer) >= seq_len:
            yield np.asarray(buffer[:seq_len], dtype=dtype)
            del buffer[:seq_len]

    if buffer and not drop_last:
        yield np.asarray(buffer, dtype=dtype)


@dataclass
class PackedExample:
    """A packed bin of whole examples (SFT-style packing).

    Attributes:
        input_ids: (seq_len,) token ids, padded with ``pad_id`` at the tail.
        segment_ids: (seq_len,) 1-based id of the example each position
            belongs to; 0 marks padding. Use it to build a block-diagonal
            attention mask: ``mask[i, j] = segment_ids[i] == segment_ids[j]``.
        loss_mask: (seq_len,) 1.0 on real tokens, 0.0 on padding.
        num_examples: how many whole examples were packed into this bin.
    """

    input_ids: np.ndarray
    segment_ids: np.ndarray
    loss_mask: np.ndarray
    num_examples: int

    def attention_mask(self) -> np.ndarray:
        """(seq_len, seq_len) block-diagonal mask (no cross-example attention)."""
        seg = self.segment_ids
        mask = (seg[:, None] == seg[None, :]) & (seg[:, None] > 0)
        return mask.astype(np.bool_)


def pack_examples(
    examples: Iterable[Sequence[int]],
    seq_len: int,
    eos_id: int,
    pad_id: int = 0,
    dtype: Any = np.int32,
) -> Iterator[PackedExample]:
    """Greedy first-fit packing of whole examples into ``seq_len`` bins.

    Examples longer than ``seq_len`` (after the EOS) are truncated to fit.
    Examples are never split across bins; a new bin starts when the current
    one cannot hold the next example.

    Args:
        examples: iterable of token-id sequences (one per example).
        seq_len: bin size (max_seq_length).
        eos_id: appended after each example.
        pad_id: token used to fill the bin tail.
        dtype: numpy dtype of emitted arrays.

    Yields:
        PackedExample bins in input order.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    ids = np.full(seq_len, pad_id, dtype=dtype)
    segs = np.zeros(seq_len, dtype=np.int32)
    cursor = 0
    segment = 0

    def _flush() -> Optional[PackedExample]:
        nonlocal ids, segs, cursor, segment
        if cursor == 0:
            return None
        out = PackedExample(
            input_ids=ids.copy(),
            segment_ids=segs.copy(),
            loss_mask=(segs > 0).astype(np.float32),
            num_examples=segment,
        )
        ids = np.full(seq_len, pad_id, dtype=dtype)
        segs = np.zeros(seq_len, dtype=np.int32)
        cursor = 0
        segment = 0
        return out

    for example in examples:
        tokens = [int(t) for t in example][: seq_len - 1] + [int(eos_id)]
        if cursor + len(tokens) > seq_len:
            flushed = _flush()
            if flushed is not None:
                yield flushed
        segment += 1
        ids[cursor : cursor + len(tokens)] = np.asarray(tokens, dtype=dtype)
        segs[cursor : cursor + len(tokens)] = segment
        cursor += len(tokens)

    flushed = _flush()
    if flushed is not None:
        yield flushed


class PackedDataset:
    """Materialised packed dataset compatible with data.core.DataLoader.

    Wraps an iterable of token documents into fixed-length packed samples.

    Args:
        documents: iterable of token-id sequences.
        seq_len: packed sample length.
        eos_id: document separator.
        mode: "stream" (pretraining, pack_token_streams) or "example"
            (SFT, pack_examples).
        pad_id: padding id for "example" mode.
    """

    def __init__(
        self,
        documents: Iterable[Sequence[int]],
        seq_len: int,
        eos_id: int,
        mode: str = "stream",
        pad_id: int = 0,
    ):
        if mode not in ("stream", "example"):
            raise ValueError(f"mode must be 'stream' or 'example', got {mode!r}")
        self.seq_len = seq_len
        self.mode = mode
        if mode == "stream":
            self.samples: List[Any] = list(
                pack_token_streams(documents, seq_len, eos_id)
            )
        else:
            self.samples = list(
                pack_examples(documents, seq_len, eos_id, pad_id=pad_id)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        return self.samples[idx]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.samples)


def packing_efficiency(
    documents: Sequence[Sequence[int]], seq_len: int
) -> Dict[str, float]:
    """Compare real-token utilisation of pad-to-max vs packing.

    Returns a dict with ``padded_utilisation``, ``packed_utilisation`` and
    ``speedup`` (ratio of batches needed: padded / packed).
    """
    total_tokens = sum(min(len(d) + 1, seq_len) for d in documents)
    if not documents or total_tokens == 0:
        return {"padded_utilisation": 0.0, "packed_utilisation": 0.0, "speedup": 1.0}

    padded_slots = len(documents) * seq_len
    packed_bins = max(1, int(np.ceil(total_tokens / seq_len)))
    packed_slots = packed_bins * seq_len

    return {
        "padded_utilisation": total_tokens / padded_slots,
        "packed_utilisation": total_tokens / packed_slots,
        "speedup": padded_slots / packed_slots,
    }
