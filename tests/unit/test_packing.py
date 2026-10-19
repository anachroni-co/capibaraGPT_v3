"""Unit tests for data/packing.py (sequence packing)."""

from __future__ import annotations

import numpy as np
import pytest

from data.packing import (
    PackedDataset,
    pack_examples,
    pack_token_streams,
    packing_efficiency,
)

EOS = 2
PAD = 0


# ---------------------------------------------------------------------------
# pack_token_streams (pretraining, contiguous)
# ---------------------------------------------------------------------------


def test_stream_blocks_have_exact_length_and_no_padding():
    docs = [[5] * 10, [7] * 25, [9] * 3]
    blocks = list(pack_token_streams(docs, seq_len=16, eos_id=EOS))
    total = sum(len(d) + 1 for d in docs)  # +1 EOS per doc
    assert all(len(b) == 16 for b in blocks)
    assert len(blocks) == total // 16
    # No pad tokens: every position is a real token or EOS
    for b in blocks:
        assert set(np.unique(b)).issubset({5, 7, 9, EOS})


def test_stream_preserves_order_and_separators():
    docs = [[1, 1], [3, 3, 3]]
    blocks = list(pack_token_streams(docs, seq_len=7, eos_id=EOS))
    assert len(blocks) == 1
    np.testing.assert_array_equal(blocks[0], [1, 1, EOS, 3, 3, 3, EOS])


def test_stream_drop_last_false_emits_tail():
    docs = [[1] * 5]
    blocks = list(pack_token_streams(docs, seq_len=4, eos_id=EOS, drop_last=False))
    assert len(blocks) == 2
    assert len(blocks[0]) == 4
    assert len(blocks[1]) == 2  # tail: 1 token + EOS


def test_stream_skips_empty_documents():
    blocks = list(pack_token_streams([[], [1, 1, 1]], seq_len=4, eos_id=EOS))
    assert len(blocks) == 1
    np.testing.assert_array_equal(blocks[0], [1, 1, 1, EOS])


def test_stream_invalid_seq_len_raises():
    with pytest.raises(ValueError):
        list(pack_token_streams([[1]], seq_len=0, eos_id=EOS))


# ---------------------------------------------------------------------------
# pack_examples (SFT, whole examples, block-diagonal attention)
# ---------------------------------------------------------------------------


def test_examples_never_split_across_bins():
    examples = [[1] * 6, [3] * 6, [5] * 6]  # 7 tokens each with EOS
    bins = list(pack_examples(examples, seq_len=16, eos_id=EOS, pad_id=PAD))
    assert len(bins) == 2
    assert bins[0].num_examples == 2
    assert bins[1].num_examples == 1
    # second bin starts with the third example, not a fragment
    assert bins[1].input_ids[0] == 5


def test_examples_segment_ids_and_loss_mask():
    bins = list(pack_examples([[1, 1], [3]], seq_len=8, eos_id=EOS, pad_id=PAD))
    b = bins[0]
    np.testing.assert_array_equal(b.segment_ids, [1, 1, 1, 2, 2, 0, 0, 0])
    np.testing.assert_array_equal(b.loss_mask, [1, 1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(b.input_ids, [1, 1, EOS, 3, EOS, PAD, PAD, PAD])


def test_examples_attention_mask_is_block_diagonal():
    bins = list(pack_examples([[1, 1], [3]], seq_len=8, eos_id=EOS, pad_id=PAD))
    mask = bins[0].attention_mask()
    # within example 1
    assert mask[0, 2] and mask[2, 0]
    # across examples: blocked
    assert not mask[0, 3] and not mask[3, 0]
    # padding attends to nothing
    assert not mask[5].any()


def test_examples_longer_than_seq_len_are_truncated():
    bins = list(pack_examples([[9] * 50], seq_len=8, eos_id=EOS, pad_id=PAD))
    assert len(bins) == 1
    b = bins[0]
    assert b.input_ids[-1] == EOS  # truncated to seq_len-1 tokens + EOS
    assert b.loss_mask.sum() == 8


# ---------------------------------------------------------------------------
# PackedDataset + efficiency
# ---------------------------------------------------------------------------


def test_packed_dataset_stream_mode_len_and_getitem():
    ds = PackedDataset([[1] * 10] * 5, seq_len=16, eos_id=EOS, mode="stream")
    assert len(ds) == (5 * 11) // 16
    assert len(ds[0]) == 16


def test_packed_dataset_example_mode():
    ds = PackedDataset([[1] * 6] * 3, seq_len=16, eos_id=EOS, mode="example")
    assert len(ds) == 2
    assert ds[0].num_examples == 2


def test_packed_dataset_invalid_mode():
    with pytest.raises(ValueError):
        PackedDataset([[1]], seq_len=8, eos_id=EOS, mode="bogus")


def test_packing_efficiency_reports_speedup():
    # 100 docs of 100 tokens with seq_len 2048: padding wastes ~95%
    docs = [[1] * 100] * 100
    stats = packing_efficiency(docs, seq_len=2048)
    assert stats["padded_utilisation"] < 0.06
    assert stats["packed_utilisation"] > 0.95
    assert stats["speedup"] > 15


def test_packing_efficiency_empty():
    stats = packing_efficiency([], seq_len=128)
    assert stats["speedup"] == 1.0
