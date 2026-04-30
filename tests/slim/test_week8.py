"""Week 8 tests — dataset registry, downloader, preprocessor."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from config.slim_loader import load_config


@pytest.fixture(autouse=True)
def _clear_config():
    load_config.cache_clear()
    yield
    load_config.cache_clear()


# ---------------------------------------------------------------------------
# T8.1 — Dataset registry
# ---------------------------------------------------------------------------

def test_registry_loads():
    from data.download import load_registry
    reg = load_registry()
    assert len(reg) >= 10


def test_registry_has_galician():
    from data.download import load_registry
    reg = load_registry()
    gl_ids = [e.id for e in reg.values() if "gl" in e.langs]
    assert len(gl_ids) >= 2, f"Expected ≥2 Galician datasets, got: {gl_ids}"


def test_registry_has_portuguese():
    from data.download import load_registry
    reg = load_registry()
    pt_ids = [e.id for e in reg.values()
               if any(l.startswith("pt") for l in e.langs)]
    assert len(pt_ids) >= 3, f"Expected ≥3 Portuguese datasets, got: {pt_ids}"


def test_registry_has_spanish():
    from data.download import load_registry
    reg = load_registry()
    es_ids = [e.id for e in reg.values() if "es" in e.langs]
    assert len(es_ids) >= 2


def test_registry_all_entries_have_required_fields():
    from data.download import load_registry
    reg = load_registry()
    for entry in reg.values():
        assert entry.id, "missing id"
        assert entry.name, f"{entry.id}: missing name"
        assert entry.langs, f"{entry.id}: missing langs"
        assert entry.source in ("huggingface", "direct"), \
            f"{entry.id}: invalid source '{entry.source}'"
        assert entry.license, f"{entry.id}: missing license"


def test_registry_bilingual_gl_pt():
    from data.download import load_registry
    reg = load_registry()
    bilingual = [
        e for e in reg.values()
        if "gl" in e.langs and any(l.startswith("pt") for l in e.langs)
    ]
    assert len(bilingual) >= 1, "Expected at least one gl+pt bilingual dataset"


# ---------------------------------------------------------------------------
# T8.2 — Text extraction
# ---------------------------------------------------------------------------

def test_extract_text_primary_field():
    from data.download import _extract_text
    record = {"text": "hello world", "meta": "x"}
    assert _extract_text(record, "text") == "hello world"


def test_extract_text_fallback_field():
    from data.download import _extract_text
    record = {"content": "fallback content"}
    assert _extract_text(record, "text") == "fallback content"


def test_extract_text_returns_none_when_missing():
    from data.download import _extract_text
    assert _extract_text({}, "text") is None


def test_extract_text_custom_field():
    from data.download import _extract_text
    record = {"passage": "passage text", "text": "other"}
    assert _extract_text(record, "passage") == "passage text"


# ---------------------------------------------------------------------------
# T8.3 — ShardWriter
# ---------------------------------------------------------------------------

def test_shard_writer_creates_files(tmp_path):
    from data.download import ShardWriter
    writer = ShardWriter(tmp_path, "test_ds", docs_per_shard=3)
    for i in range(7):
        writer.write(f"Document {i} with enough content to pass the filter.")
    meta = writer.close()
    assert meta["docs"] == 7
    assert meta["shards"] == 3   # 3+3+1
    shards = sorted((tmp_path / "test_ds").glob("shard_*.txt"))
    assert len(shards) == 3


def test_shard_writer_meta_json(tmp_path):
    from data.download import ShardWriter
    writer = ShardWriter(tmp_path, "ds2", docs_per_shard=10)
    writer.write("some document text " * 5)
    meta = writer.close()
    meta_file = tmp_path / "ds2" / "meta.json"
    assert meta_file.exists()
    loaded = json.loads(meta_file.read_text())
    assert loaded["docs"] == 1
    assert loaded["shards"] == 1


def test_shard_writer_content_readable(tmp_path):
    from data.download import ShardWriter
    writer = ShardWriter(tmp_path, "ds3", docs_per_shard=100)
    writer.write("first document")
    writer.write("second document")
    writer.close()
    text = (tmp_path / "ds3" / "shard_0000.txt").read_text()
    assert "first document" in text
    assert "second document" in text


# ---------------------------------------------------------------------------
# T8.4 — CLI: list command
# ---------------------------------------------------------------------------

def test_cli_list_all(capsys):
    from data.download import main
    main(["list"])
    out = capsys.readouterr().out
    assert "ID" in out
    assert "corpusnos" in out or "gigaverbo" in out


def test_cli_list_filter_lang(capsys):
    from data.download import main
    main(["list", "--lang", "gl"])
    out = capsys.readouterr().out
    assert "gl" in out
    # Every non-header data line must contain "gl" in the LANGUAGES column
    lines = [
        l for l in out.splitlines()
        if l.strip() and "ID" not in l and "-" * 4 not in l
    ]
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            assert "gl" in parts[1], f"Non-Galician row in gl filter: {line}"


def test_cli_info(capsys):
    from data.download import main
    main(["info", "corpusnos"])
    out = capsys.readouterr().out
    assert "CorpusNÓS" in out or "corpusnos" in out.lower()
    assert "gl" in out


def test_cli_info_unknown_exits(capsys):
    from data.download import main
    with pytest.raises(SystemExit):
        main(["info", "does-not-exist"])


# ---------------------------------------------------------------------------
# T8.5 — Preprocessor: merge + split
# ---------------------------------------------------------------------------

def test_iter_docs_from_shards(tmp_path):
    from data.preprocess import iter_docs
    ds_dir = tmp_path / "myds"
    ds_dir.mkdir()
    (ds_dir / "shard_0000.txt").write_text(
        "doc one\n\ndoc two\n\ndoc three\n\n", encoding="utf-8"
    )
    docs = list(iter_docs(ds_dir))
    assert docs == ["doc one", "doc two", "doc three"]


def test_iter_docs_empty_dir(tmp_path):
    from data.preprocess import iter_docs
    docs = list(iter_docs(tmp_path / "empty"))
    assert docs == []


def test_merge_datasets(tmp_path):
    from data.preprocess import merge_datasets
    ds1 = tmp_path / "ds1"
    ds1.mkdir()
    (ds1 / "shard_0000.txt").write_text("alpha\n\nbeta\n\n")
    ds2 = tmp_path / "ds2"
    ds2.mkdir()
    (ds2 / "shard_0000.txt").write_text("gamma\n\ndelta\n\n")

    out = tmp_path / "merged.txt"
    stats = merge_datasets([ds1, ds2], out, shuffle=False)

    assert stats["total_docs"] == 4
    text = out.read_text()
    for word in ["alpha", "beta", "gamma", "delta"]:
        assert word in text


def test_split_train_val(tmp_path):
    from data.preprocess import split_train_val
    # 100 docs
    inp = tmp_path / "input.txt"
    inp.write_text("\n\n".join(f"doc_{i}" for i in range(100)) + "\n\n")

    stats = split_train_val(inp, tmp_path / "out", val_ratio=0.1)

    assert stats["val_docs"] == 10
    assert stats["train_docs"] == 90
    assert (tmp_path / "out" / "train.txt").exists()
    assert (tmp_path / "out" / "val.txt").exists()


def test_split_creates_stats_json(tmp_path):
    from data.preprocess import split_train_val
    inp = tmp_path / "data.txt"
    inp.write_text("\n\n".join(f"document {i}" for i in range(20)) + "\n\n")
    split_train_val(inp, tmp_path / "out", val_ratio=0.1)
    stats = json.loads((tmp_path / "out" / "stats.json").read_text())
    assert "total_docs" in stats
    assert stats["total_docs"] == 20


def test_pipeline_cli(tmp_path):
    from data.preprocess import main
    ds = tmp_path / "raw" / "myds"
    ds.mkdir(parents=True)
    (ds / "shard_0000.txt").write_text(
        "\n\n".join(f"document number {i}" for i in range(50)) + "\n\n"
    )
    main([
        "pipeline",
        "--datasets", str(ds),
        "--output-dir", str(tmp_path / "processed"),
        "--val-ratio", "0.1",
        "--no-shuffle",
    ])
    assert (tmp_path / "processed" / "train.txt").exists()
    assert (tmp_path / "processed" / "val.txt").exists()
