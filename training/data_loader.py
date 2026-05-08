"""Shard-based data loader for TPU training.

Reads pre-tokenized .npy shards produced by scripts/prepare_corpus.py
and yields {input_ids, labels} batches compatible with the TPU trainer.

Supports:
  - Local filesystem shards
  - GCS paths (gs://bucket/path) via gcsfs (optional)
  - Infinite cycling with shard shuffling
  - Padding to fixed sequence length

Usage:
    from training.data_loader import ShardDataLoader, DataLoaderConfig

    loader = ShardDataLoader(DataLoaderConfig(
        data_dir="data/tokenized/",
        batch_size=32,
        seq_len=2048,
    ))
    for batch in loader:
        # batch = {'input_ids': np.array (B,T), 'labels': np.array (B,T)}
        ...
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

PAD_ID = 256  # matches ByteLevelTokenizer.special_tokens['PAD']


@dataclass
class DataLoaderConfig:
    data_dir: str = "data/tokenized"
    batch_size: int = 32
    seq_len: int = 2048
    pad_id: int = PAD_ID
    shuffle_shards: bool = True
    seed: int = 42
    prefetch_shards: int = 2        # shards kept in memory ahead
    drop_last: bool = True          # drop incomplete final batch
    gcs_project: Optional[str] = None  # set when using gs:// paths


def _list_shards(data_dir: str) -> List[str]:
    """Return sorted list of .npy shard paths (local or GCS)."""
    if data_dir.startswith("gs://"):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            prefix = data_dir[5:]  # strip gs://
            paths = [f"gs://{p}" for p in fs.ls(prefix) if p.endswith(".npy")]
            return sorted(paths)
        except ImportError:
            raise ImportError(
                "gcsfs is required for GCS paths: pip install gcsfs"
            )
    else:
        p = Path(data_dir)
        paths = sorted(str(s) for s in p.rglob("*.npy"))
        if not paths:
            raise FileNotFoundError(f"No .npy shards found in {data_dir} (searched recursively)")
        return paths


def _load_shard(path: str) -> np.ndarray:
    """Load a shard (local or GCS) → 1-D int32 token array."""
    if path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "rb") as f:
            data = np.load(f)
    else:
        data = np.load(path)
    return data.astype(np.int32).ravel()


def _make_batches(
    tokens: np.ndarray,
    seq_len: int,
    batch_size: int,
    pad_id: int,
    drop_last: bool,
) -> Generator[dict, None, None]:
    """Slice tokens into (input, label) pairs and group into batches."""
    # Each example is seq_len tokens; label = input shifted by 1
    step = seq_len
    examples_input: List[np.ndarray] = []
    examples_label: List[np.ndarray] = []

    # Need seq_len + 1 tokens per example (input + next-token label)
    for start in range(0, len(tokens) - seq_len, step):
        chunk = tokens[start: start + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        examples_input.append(chunk[:seq_len])
        examples_label.append(chunk[1: seq_len + 1])

    n = len(examples_input)
    if n == 0:
        return

    for batch_start in range(0, n, batch_size):
        batch_end = batch_start + batch_size
        if batch_end > n:
            if drop_last:
                break
            # Pad with last example to fill batch
            pad_count = batch_end - n
            examples_input += [examples_input[-1]] * pad_count
            examples_label += [examples_label[-1]] * pad_count

        inp = np.stack(examples_input[batch_start: batch_start + batch_size])
        lbl = np.stack(examples_label[batch_start: batch_start + batch_size])
        yield {"input_ids": inp, "labels": lbl}


class ShardDataLoader:
    """Infinite iterator over pre-tokenized .npy shards."""

    def __init__(self, config: DataLoaderConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._shards = _list_shards(config.data_dir)
        logger.info(
            "ShardDataLoader: %d shards in %s | batch=%d seq_len=%d",
            len(self._shards), config.data_dir, config.batch_size, config.seq_len,
        )

    def __iter__(self) -> Iterator[dict]:
        return self._infinite_iter()

    def _infinite_iter(self) -> Iterator[dict]:
        cfg = self.config
        shards = list(self._shards)

        while True:
            if cfg.shuffle_shards:
                self._rng.shuffle(shards)

            for shard_path in shards:
                try:
                    tokens = _load_shard(shard_path)
                    logger.debug("Loaded shard %s (%d tokens)", shard_path, len(tokens))
                except Exception as exc:
                    logger.warning("Failed to load shard %s: %s", shard_path, exc)
                    continue

                yield from _make_batches(
                    tokens,
                    seq_len=cfg.seq_len,
                    batch_size=cfg.batch_size,
                    pad_id=cfg.pad_id,
                    drop_last=cfg.drop_last,
                )

    def steps_per_epoch_estimate(self) -> int:
        """Rough estimate of steps per full pass over the data."""
        total = 0
        for shard in self._shards[:5]:  # sample first 5 shards
            try:
                t = _load_shard(shard)
                total += len(t)
            except Exception:
                pass
        if not total:
            return 0
        avg_per_shard = total / min(5, len(self._shards))
        examples_per_shard = avg_per_shard // (self.config.seq_len + 1)
        batches_per_shard = examples_per_shard // self.config.batch_size
        return int(batches_per_shard * len(self._shards))
