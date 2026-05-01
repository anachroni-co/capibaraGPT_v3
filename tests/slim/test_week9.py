"""Week 9 tests — DataMixer, MixConfig, two-phase training configs, CLI."""
from __future__ import annotations

import pytest
from pathlib import Path

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

needs_torch = pytest.mark.skipif(not _TORCH, reason="torch not installed")

from config.slim_loader import load_config

_MIX_CONFIGS = Path(__file__).parents[2] / "data" / "mix_configs"


@pytest.fixture(autouse=True)
def _clear_config():
    load_config.cache_clear()
    yield
    load_config.cache_clear()


# ---------------------------------------------------------------------------
# T9.1 — MixConfig
# ---------------------------------------------------------------------------

def test_mix_config_from_dict():
    from data.mixer import MixConfig
    cfg = MixConfig.from_dict({
        "name": "test",
        "sources": [
            {"path": "a.txt", "lang": "en", "weight": 0.8},
            {"path": "b.txt", "lang": "es", "weight": 0.2},
        ],
    })
    assert cfg.name == "test"
    assert len(cfg.sources) == 2


def test_mix_config_weights_normalized():
    from data.mixer import MixConfig
    cfg = MixConfig.from_dict({
        "name": "t",
        "sources": [
            {"path": "a.txt", "lang": "en", "weight": 4.0},
            {"path": "b.txt", "lang": "gl", "weight": 1.0},
        ],
    })
    nw = cfg.weights_normalized
    assert abs(nw[0] - 0.8) < 1e-6
    assert abs(nw[1] - 0.2) < 1e-6
    assert abs(sum(nw) - 1.0) < 1e-6


def test_mix_config_rejects_zero_weight():
    from data.mixer import LanguageSource
    with pytest.raises(ValueError, match="weight"):
        LanguageSource(path="x.txt", lang="gl", weight=0.0)


def test_mix_config_rejects_empty_sources():
    from data.mixer import MixConfig
    with pytest.raises(ValueError, match="source"):
        MixConfig(name="t", sources=[])


def test_mix_config_from_yaml_base():
    from data.mixer import MixConfig
    cfg = MixConfig.from_yaml(_MIX_CONFIGS / "base_pretraining.yaml")
    assert cfg.name == "base_pretraining"
    langs = [s.lang for s in cfg.sources]
    assert "en" in langs
    assert "es" in langs
    assert "pt" in langs


def test_mix_config_from_yaml_galician():
    from data.mixer import MixConfig
    cfg = MixConfig.from_yaml(_MIX_CONFIGS / "galician_continual.yaml")
    assert cfg.name == "galician_continual"
    langs = [s.lang for s in cfg.sources]
    assert "gl" in langs


def test_base_config_en_weight_dominant():
    from data.mixer import MixConfig
    cfg = MixConfig.from_yaml(_MIX_CONFIGS / "base_pretraining.yaml")
    nw = cfg.weights_normalized
    en_idx = next(i for i, s in enumerate(cfg.sources) if s.lang == "en")
    assert nw[en_idx] >= 0.7, "English should be ≥70% in base pretraining"


def test_galician_config_gl_weight_dominant():
    from data.mixer import MixConfig
    cfg = MixConfig.from_yaml(_MIX_CONFIGS / "galician_continual.yaml")
    nw = cfg.weights_normalized
    gl_idx = next(i for i, s in enumerate(cfg.sources) if s.lang == "gl")
    assert nw[gl_idx] >= 0.40, "Galician should be ≥40% in continual phase"


def test_mixing_stats_utility():
    from data.mixer import MixConfig, mixing_stats
    cfg = MixConfig.from_dict({
        "name": "t",
        "sources": [
            {"path": "a.txt", "lang": "en", "weight": 8},
            {"path": "b.txt", "lang": "gl", "weight": 2},
        ],
    })
    stats = mixing_stats(cfg)
    assert abs(stats["en"] - 0.8) < 1e-4
    assert abs(stats["gl"] - 0.2) < 1e-4


# ---------------------------------------------------------------------------
# T9.2 — MixedDataset (no torch for config tests; torch for dataset tests)
# ---------------------------------------------------------------------------

@needs_torch
def test_mixed_dataset_respects_weights(tmp_path):
    from data.mixer import MixedDataset
    from utils.tokenizer import SlimTokenizer
    from data.loader import SlimDataset

    tok = SlimTokenizer()
    seq = 8

    def _make_ds(text_file, text):
        text_file.write_text(text * 500, encoding="utf-8")
        return SlimDataset(text_file, tok, seq_len=seq)

    ds_en = _make_ds(tmp_path / "en.txt", "the quick brown fox jumps over the lazy dog ")
    ds_gl = _make_ds(tmp_path / "gl.txt", "o raposo pardo e veloz salta sobre o can preguiceiro ")

    epoch = 1000
    mixed = MixedDataset(
        datasets=[ds_en, ds_gl],
        weights=[0.8, 0.2],
        epoch_size=epoch,
        seed=0,
    )
    assert len(mixed) == epoch

    # Check that sampling proportions are approximately correct (±5%)
    from collections import Counter
    counts = Counter(mixed._items[i][0] for i in range(epoch))
    en_ratio = counts[0] / epoch
    gl_ratio = counts[1] / epoch
    assert abs(en_ratio - 0.8) < 0.05, f"en ratio {en_ratio:.3f} not close to 0.8"
    assert abs(gl_ratio - 0.2) < 0.05, f"gl ratio {gl_ratio:.3f} not close to 0.2"


@needs_torch
def test_mixed_dataset_getitem_returns_tensors(tmp_path):
    from data.mixer import MixedDataset
    from utils.tokenizer import SlimTokenizer
    from data.loader import SlimDataset

    tok = SlimTokenizer()
    text = "hello world " * 200
    ds = SlimDataset(tmp_path / "t.txt", tok, seq_len=8)
    (tmp_path / "t.txt").write_text(text)
    ds = SlimDataset(tmp_path / "t.txt", tok, seq_len=8)

    mixed = MixedDataset([ds, ds], weights=[0.5, 0.5], epoch_size=10, seed=1)
    item = mixed[0]
    assert "input_ids" in item
    assert "labels" in item
    assert item["input_ids"].shape == (8,)


@needs_torch
def test_mixed_dataset_deterministic(tmp_path):
    from data.mixer import MixedDataset
    from utils.tokenizer import SlimTokenizer
    from data.loader import SlimDataset

    tok = SlimTokenizer()
    (tmp_path / "t.txt").write_text("word " * 300)
    ds = SlimDataset(tmp_path / "t.txt", tok, seq_len=8)

    m1 = MixedDataset([ds], weights=[1.0], epoch_size=20, seed=42)
    m2 = MixedDataset([ds], weights=[1.0], epoch_size=20, seed=42)
    assert m1._items == m2._items


# ---------------------------------------------------------------------------
# T9.3 — Phase config files exist and are valid YAML
# ---------------------------------------------------------------------------

def test_base_pretraining_yaml_exists():
    assert (_MIX_CONFIGS / "base_pretraining.yaml").exists()


def test_galician_continual_yaml_exists():
    assert (_MIX_CONFIGS / "galician_continual.yaml").exists()


def test_phase_configs_parse_without_error():
    from data.mixer import MixConfig
    for name in ("base_pretraining.yaml", "galician_continual.yaml"):
        cfg = MixConfig.from_yaml(_MIX_CONFIGS / name)
        assert cfg.name
        assert len(cfg.sources) >= 2


# ---------------------------------------------------------------------------
# T9.4 — CLI --mix flag
# ---------------------------------------------------------------------------

def test_train_cli_accepts_mix_flag():
    """--mix and --data are mutually exclusive; --mix must be accepted."""
    import argparse
    from training.train import _parse_args
    args = _parse_args([
        "--mix", "data/mix_configs/base_pretraining.yaml",
        "--preset", "1.5b",
    ])
    assert args.mix == "data/mix_configs/base_pretraining.yaml"
    assert args.data is None


def test_train_cli_data_and_mix_mutually_exclusive():
    from training.train import _parse_args
    with pytest.raises(SystemExit):
        _parse_args(["--data", "x.txt", "--mix", "y.yaml", "--preset", "1.5b"])


def test_train_cli_data_still_works():
    from training.train import _parse_args
    args = _parse_args(["--data", "train.txt", "--preset", "1.5b"])
    assert args.data == "train.txt"
    assert args.mix is None
