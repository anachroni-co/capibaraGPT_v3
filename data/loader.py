"""Capibara Slim — data loader (T5.5).

Reads plain-text files (one document per line or arbitrary text),
tokenises them with SlimTokenizer, and packs tokens into fixed-length
sequences for language-model training.

Usage:
    dataset = SlimDataset("data/train.txt", tokenizer, seq_len=2048)
    loader  = create_dataloader(dataset, batch_size=4, shuffle=True)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    _TORCH = True
except ImportError:
    _TORCH = False


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError("SlimDataset requires PyTorch. Install with: pip install torch")


class SlimDataset:
    """Token-packed text dataset for causal LM training.

    Reads all text from `path`, tokenises it, and slices the flat token
    sequence into non-overlapping chunks of length `seq_len + 1`
    (input = [:seq_len], target = [1:]).
    """

    def __init__(
        self,
        path: Union[str, Path],
        tokenizer,
        seq_len: int = 2048,
    ) -> None:
        _require_torch()
        self.seq_len = seq_len
        self._tokenizer = tokenizer

        text = Path(path).read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        logger.info("SlimDataset: %d tokens from %s", len(token_ids), path)

        # Pack into fixed-length chunks
        chunk = seq_len + 1
        n_chunks = len(token_ids) // chunk
        if n_chunks == 0:
            raise ValueError(
                f"File too small: only {len(token_ids)} tokens, need at least {chunk}."
            )
        ids = torch.tensor(token_ids[: n_chunks * chunk], dtype=torch.long)
        self._chunks = ids.view(n_chunks, chunk)

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int):
        chunk = self._chunks[idx]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


def create_dataloader(
    dataset: "SlimDataset",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> "DataLoader":
    _require_torch()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
