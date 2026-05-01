"""Capibara Slim — multilingual data mixer.

Builds a mixed training dataset from multiple language sources, sampling
each according to configurable weights. Designed for two-phase training:

  Phase 1 — Base pretraining  (en 80% / es 10% / pt 10%)
  Phase 2 — Continual (gl)    (gl 50% / es 25% / pt 25%)

Usage (programmatic):
    from data.mixer import DataMixer
    loader = DataMixer.from_yaml("data/mix_configs/base_pretraining.yaml",
                                  tokenizer, seq_len=2048, batch_size=4)

Usage (pre-built configs):
    from data.mixer import base_pretraining_loader, galician_continual_loader
    loader = base_pretraining_loader(sources_map, tokenizer)

Mix config YAML format:
    name: base_pretraining
    epoch_size: 100000       # items per epoch (default: sum of all dataset lengths)
    seed: 42
    sources:
      - path: data/processed/en_train.txt
        lang: en
        weight: 0.80
        name: FineWeb-Edu English
      - path: data/processed/es_train.txt
        lang: es
        weight: 0.10
      - path: data/processed/pt_train.txt
        lang: pt
        weight: 0.10
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    _TORCH = True
except ImportError:
    _TORCH = False


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------

@dataclass
class LanguageSource:
    path: str
    lang: str
    weight: float
    name: str = ""

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"weight must be > 0, got {self.weight} for {self.lang}")


@dataclass
class MixConfig:
    name: str
    sources: list[LanguageSource]
    epoch_size: int = 0          # 0 → auto (sum of dataset lengths)
    seed: int = 42
    description: str = ""

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("MixConfig requires at least one source")
        total_w = sum(s.weight for s in self.sources)
        if total_w <= 0:
            raise ValueError("Sum of weights must be > 0")

    @property
    def weights_normalized(self) -> list[float]:
        total = sum(s.weight for s in self.sources)
        return [s.weight / total for s in self.sources]

    @classmethod
    def from_dict(cls, d: dict) -> "MixConfig":
        sources = [LanguageSource(**s) for s in d["sources"]]
        return cls(
            name=d["name"],
            sources=sources,
            epoch_size=d.get("epoch_size", 0),
            seed=d.get("seed", 42),
            description=d.get("description", ""),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MixConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


# ---------------------------------------------------------------------------
# MixedDataset
# ---------------------------------------------------------------------------

class MixedDataset:
    """Samples from multiple SlimDatasets according to language weights.

    Pre-computes a deterministic epoch-length index mapping so DataLoader
    can use standard integer indexing.
    """

    def __init__(
        self,
        datasets: list[Any],       # list[SlimDataset]
        weights: list[float],
        epoch_size: int = 0,
        seed: int = 42,
        lang_names: list[str] | None = None,
    ) -> None:
        if not _TORCH:
            raise ImportError("MixedDataset requires PyTorch")
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have the same length")

        self._datasets = datasets
        self._lang_names = lang_names or [f"src{i}" for i in range(len(datasets))]

        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        if epoch_size <= 0:
            epoch_size = sum(len(ds) for ds in datasets)

        rng = np.random.default_rng(seed)
        src_indices = rng.choice(len(datasets), size=epoch_size, p=probs)

        # For each chosen source pick a random item from that source
        self._items: list[tuple[int, int]] = []
        for si in src_indices:
            li = int(rng.integers(0, len(datasets[si])))
            self._items.append((si, li))

        # Log mixing stats
        from collections import Counter
        dist = Counter(si for si, _ in self._items)
        for si, cnt in sorted(dist.items()):
            logger.info(
                "  %-20s  %6d items  (%.1f%%)",
                self._lang_names[si], cnt, 100 * cnt / epoch_size,
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        si, li = self._items[idx]
        return self._datasets[si][li]

    @property
    def lang_names(self) -> list[str]:
        return self._lang_names


# ---------------------------------------------------------------------------
# DataMixer factory
# ---------------------------------------------------------------------------

class DataMixer:
    """Builds a MixedDataset + DataLoader from a MixConfig."""

    def __init__(self, config: MixConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataMixer":
        return cls(MixConfig.from_yaml(path))

    def build_dataset(
        self,
        tokenizer: Any,
        seq_len: int = 2048,
    ) -> "MixedDataset":
        """Load each source file as a SlimDataset and combine."""
        from data.loader import SlimDataset

        datasets = []
        for src in self.config.sources:
            p = Path(src.path)
            if not p.exists():
                raise FileNotFoundError(
                    f"Source file not found: {p}\n"
                    f"Run 'python -m data.download get {src.lang}' first."
                )
            ds = SlimDataset(p, tokenizer, seq_len=seq_len)
            datasets.append(ds)
            logger.info("Loaded %-20s %7d seqs  %s", src.name or src.lang, len(ds), p)

        return MixedDataset(
            datasets=datasets,
            weights=[s.weight for s in self.config.sources],
            epoch_size=self.config.epoch_size,
            seed=self.config.seed,
            lang_names=[s.name or s.lang for s in self.config.sources],
        )

    def build_dataloader(
        self,
        tokenizer: Any,
        seq_len: int = 2048,
        batch_size: int = 4,
        num_workers: int = 0,
    ) -> "DataLoader":
        from data.loader import create_dataloader
        ds = self.build_dataset(tokenizer, seq_len)
        return create_dataloader(ds, batch_size=batch_size,
                                  num_workers=num_workers)


# ---------------------------------------------------------------------------
# Sampling stats utility
# ---------------------------------------------------------------------------

def mixing_stats(config: MixConfig) -> dict:
    """Return a dict of {lang: effective_ratio} without loading any files."""
    nw = config.weights_normalized
    return {
        src.lang: round(nw[i], 4)
        for i, src in enumerate(config.sources)
    }
