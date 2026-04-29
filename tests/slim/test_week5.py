"""Week 5 tests — architecture, tokenizer, data loader, trainer, checkpoint (T5.1–T5.5).

All architecture/training tests are skipped when torch is absent.
Config and tokenizer tests run without torch.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

needs_torch = pytest.mark.skipif(not _TORCH, reason="torch not installed")


# ---------------------------------------------------------------------------
# T5.1 — SlimConfig
# ---------------------------------------------------------------------------

from models.architecture import SlimConfig


def test_config_defaults():
    cfg = SlimConfig()
    assert cfg.hidden_size == 2048
    assert cfg.num_layers == 24
    assert cfg.num_heads == 16


def test_config_preset_1_5b():
    cfg = SlimConfig.preset("1.5b")
    assert cfg.hidden_size == 2048
    assert cfg.num_layers == 24


def test_config_preset_3b():
    cfg = SlimConfig.preset("3b")
    assert cfg.hidden_size == 2560
    assert cfg.num_layers == 32


def test_config_preset_7b():
    cfg = SlimConfig.preset("7b")
    assert cfg.hidden_size == 4096


def test_config_preset_unknown_raises():
    with pytest.raises(ValueError):
        SlimConfig.preset("42b")


def test_config_estimate_params():
    cfg = SlimConfig.preset("1.5b")
    params = cfg.estimate_params()
    assert params > 1_000_000_000   # at least 1B
    assert params < 3_000_000_000   # at most 3B


# ---------------------------------------------------------------------------
# T5.1 — Model construction (torch required)
# ---------------------------------------------------------------------------

@needs_torch
def test_model_builds_from_tiny_config():
    from models.architecture import SlimModel
    cfg = SlimConfig(hidden_size=64, num_layers=2, num_heads=4,
                     intermediate_size=128, vocab_size=100, max_seq_len=32)
    model = SlimModel(cfg)
    assert model.num_params() > 0


@needs_torch
def test_model_forward_shape():
    from models.architecture import SlimModel
    cfg = SlimConfig(hidden_size=64, num_layers=2, num_heads=4,
                     intermediate_size=128, vocab_size=100, max_seq_len=32)
    model = SlimModel(cfg)
    ids = torch.randint(0, 100, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, 100)


@needs_torch
def test_model_hybrid_mamba_blocks():
    from models.architecture import SlimModel, MambaBlock, TransformerBlock
    cfg = SlimConfig(hidden_size=64, num_layers=4, num_heads=4,
                     intermediate_size=128, vocab_size=100, max_seq_len=32,
                     mamba_every_n=2)
    model = SlimModel(cfg)
    has_mamba = any(isinstance(b, MambaBlock) for b in model.blocks)
    has_transformer = any(isinstance(b, TransformerBlock) for b in model.blocks)
    assert has_mamba
    assert has_transformer


@needs_torch
def test_transformer_block_preserves_shape():
    from models.architecture import TransformerBlock
    cfg = SlimConfig(hidden_size=64, num_layers=1, num_heads=4,
                     intermediate_size=128, vocab_size=100, max_seq_len=32)
    block = TransformerBlock(cfg)
    x = torch.randn(2, 8, 64)
    out = block(x)
    assert out.shape == x.shape


@needs_torch
def test_mamba_block_preserves_shape():
    from models.architecture import MambaBlock
    cfg = SlimConfig(hidden_size=64, num_layers=1, num_heads=4,
                     intermediate_size=128, vocab_size=100, max_seq_len=32)
    block = MambaBlock(cfg)
    x = torch.randn(2, 8, 64)
    out = block(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# T5.2 — SlimTokenizer
# ---------------------------------------------------------------------------

from utils.tokenizer import SlimTokenizer


def test_tok_whitespace_fallback():
    tok = SlimTokenizer(model_path=None)
    ids = tok.encode("hello world")
    assert isinstance(ids, list)
    assert len(ids) == 2


def test_tok_decode_roundtrip():
    tok = SlimTokenizer(model_path=None)
    text = "the quick brown fox"
    ids = tok.encode(text)
    recovered = tok.decode(ids)
    assert recovered == text


def test_tok_add_special_tokens():
    tok = SlimTokenizer(model_path=None)
    ids = tok.encode("hello", add_special_tokens=True)
    # BOS + tokens + EOS
    assert len(ids) == 3


def test_tok_vocab_size_grows():
    tok = SlimTokenizer(model_path=None)
    tok.encode("unique_word_xyz")
    assert tok.vocab_size > 4


def test_tok_hf_path(tmp_path):
    # Non-existent path should silently fall back
    tok = SlimTokenizer(model_path=str(tmp_path / "nonexistent"))
    ids = tok.encode("fallback")
    assert isinstance(ids, list)


# ---------------------------------------------------------------------------
# T5.5 — Data loader (torch required)
# ---------------------------------------------------------------------------

@needs_torch
def test_dataset_creates_chunks(tmp_path):
    from data.loader import SlimDataset
    tok = SlimTokenizer()
    text = " ".join([f"word{i}" for i in range(300)])
    f = tmp_path / "train.txt"
    f.write_text(text)
    ds = SlimDataset(f, tok, seq_len=16)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["input_ids"].shape == (16,)
    assert sample["labels"].shape == (16,)


@needs_torch
def test_dataset_too_small_raises(tmp_path):
    from data.loader import SlimDataset
    tok = SlimTokenizer()
    f = tmp_path / "tiny.txt"
    f.write_text("too short")
    with pytest.raises(ValueError):
        SlimDataset(f, tok, seq_len=128)


@needs_torch
def test_dataloader_batches(tmp_path):
    from data.loader import SlimDataset, create_dataloader
    tok = SlimTokenizer()
    text = " ".join([f"w{i}" for i in range(1000)])
    f = tmp_path / "data.txt"
    f.write_text(text)
    ds = SlimDataset(f, tok, seq_len=16)
    loader = create_dataloader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 16)


# ---------------------------------------------------------------------------
# T5.3 — TrainConfig
# ---------------------------------------------------------------------------

from training.trainer import TrainConfig


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.max_steps == 100_000
    assert cfg.effective_batch_size == 32  # 4 * 8


def test_train_config_custom():
    cfg = TrainConfig(batch_size=8, grad_accum=4, lr=1e-4)
    assert cfg.effective_batch_size == 32
    assert cfg.lr == 1e-4


# ---------------------------------------------------------------------------
# T5.3 — SlimTrainer (torch required)
# ---------------------------------------------------------------------------

@needs_torch
def test_trainer_single_step(tmp_path):
    from models.architecture import SlimModel
    from data.loader import SlimDataset, create_dataloader
    from training.trainer import SlimTrainer

    tok = SlimTokenizer()
    cfg = SlimConfig(hidden_size=32, num_layers=1, num_heads=2,
                     intermediate_size=64, vocab_size=tok.vocab_size or 50,
                     max_seq_len=16)
    model = SlimModel(cfg)

    text = " ".join([f"w{i}" for i in range(500)])
    f = tmp_path / "d.txt"
    f.write_text(text)
    ds = SlimDataset(f, tok, seq_len=8)
    loader = create_dataloader(ds, batch_size=2, shuffle=False)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "ckpts"),
        max_steps=1, batch_size=2, grad_accum=1, dtype="fp32", device="cpu",
    )
    trainer = SlimTrainer(model, train_cfg, loader)
    batch = next(iter(loader))
    loss = trainer.train_step(batch)
    assert isinstance(loss, float)
    assert loss > 0


# ---------------------------------------------------------------------------
# T5.4 — Checkpointing (torch required)
# ---------------------------------------------------------------------------

@needs_torch
def test_checkpoint_save_and_load(tmp_path):
    from models.architecture import SlimModel
    from training.checkpoint import save_checkpoint, load_checkpoint, list_checkpoints
    import torch.optim as optim

    cfg = SlimConfig(hidden_size=32, num_layers=1, num_heads=2,
                     intermediate_size=64, vocab_size=50, max_seq_len=16)
    model = SlimModel(cfg)
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_dir = save_checkpoint(model, opt, step=100, output_dir=str(tmp_path))
    assert (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "meta.json").exists()

    meta = json.loads((ckpt_dir / "meta.json").read_text())
    assert meta["step"] == 100

    # Load into a fresh model
    model2 = SlimModel(cfg)
    step = load_checkpoint(model2, output_dir=str(tmp_path), device="cpu")
    assert step == 100

    # Weights match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


@needs_torch
def test_list_checkpoints(tmp_path):
    from models.architecture import SlimModel
    from training.checkpoint import save_checkpoint, list_checkpoints
    import torch.optim as optim

    cfg = SlimConfig(hidden_size=32, num_layers=1, num_heads=2,
                     intermediate_size=64, vocab_size=50, max_seq_len=16)
    model = SlimModel(cfg)
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for step in [100, 200, 300]:
        save_checkpoint(model, opt, step=step, output_dir=str(tmp_path))

    ckpts = list_checkpoints(str(tmp_path))
    assert len(ckpts) == 3
    steps = [int(p.name.split("_")[1]) for p in ckpts]
    assert steps == [100, 200, 300]
