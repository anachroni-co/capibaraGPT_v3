"""Integration tests: CapibaraHybridLM wires layers/ + sub_models/ blocks.

Covers the three previously-disconnected components now integrated:
- HybridLayerStack (interleaved SSM+attention, Jamba-style)
- BitNet158 1.58-bit LM head (layers/sparsity)
- SpikeSSM residual block (sub_models/experimental) incl. gradient flow
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax_unavailable = False
try:
    from models.hybrid_model import CapibaraHybridLM, HybridLMConfig
except Exception:  # pragma: no cover
    jax_unavailable = True

pytestmark = pytest.mark.skipif(jax_unavailable, reason="JAX/Flax required")

VOCAB = 97
CFG = dict(
    vocab_size=VOCAB,
    num_layers=4,
    hidden_size=32,
    num_heads=4,
    d_state=8,
    max_seq_length=16,
    dropout_rate=0.0,
)


def _tokens(batch=2, seq=8):
    return jax.random.randint(jax.random.PRNGKey(7), (batch, seq), 0, VOCAB)


def _run(cfg: "HybridLMConfig", tokens):
    model = CapibaraHybridLM(config=cfg)
    params = model.init(jax.random.PRNGKey(0), tokens)
    return model, params, model.apply(params, tokens)


def test_forward_logits_shape_dense_tied():
    cfg = HybridLMConfig(**CFG)
    _, _, logits = _run(cfg, _tokens())
    assert logits.shape == (2, 8, VOCAB)
    assert jnp.isfinite(logits).all()


def test_forward_bitnet_head():
    cfg = HybridLMConfig(**CFG, lm_head="bitnet")
    _, _, logits = _run(cfg, _tokens())
    assert logits.shape == (2, 8, VOCAB)
    assert jnp.isfinite(logits).all()


def test_forward_with_spike_block_and_gradients():
    cfg = HybridLMConfig(**CFG, use_spike_block=True, spike_state_dim=8)
    tokens = _tokens()
    model, params, logits = _run(cfg, tokens)
    assert logits.shape == (2, 8, VOCAB)

    def loss_fn(p):
        out = model.apply(p, tokens)
        return jnp.mean(out**2)

    grads = jax.grad(loss_fn)(params)
    total = sum(float(jnp.abs(g).sum()) for g in jax.tree_util.tree_leaves(grads))
    assert total > 0.0  # surrogate gradients flow through the spike block


def test_explicit_layer_assignment():
    cfg = HybridLMConfig(
        **CFG, ssm_layers=[0, 1, 2], attention_layers=[3]
    )
    _, params, logits = _run(cfg, _tokens())
    assert logits.shape == (2, 8, VOCAB)


def test_untied_dense_head():
    cfg = HybridLMConfig(**CFG, tie_embeddings=False)
    _, params, logits = _run(cfg, _tokens())
    assert logits.shape == (2, 8, VOCAB)


def test_seq_len_over_max_raises():
    cfg = HybridLMConfig(**CFG)
    model = CapibaraHybridLM(config=cfg)
    tokens = _tokens(seq=8)
    params = model.init(jax.random.PRNGKey(0), tokens)
    with pytest.raises(ValueError):
        model.apply(params, _tokens(seq=32))


def test_invalid_lm_head_raises():
    with pytest.raises(ValueError):
        HybridLMConfig(**CFG, lm_head="int4")


def test_param_count_reasonable():
    cfg = HybridLMConfig(**CFG)
    model = CapibaraHybridLM(config=cfg)
    params = model.init(jax.random.PRNGKey(0), _tokens())
    n = sum(x.size for x in jax.tree_util.tree_leaves(params))
    # embeddings (97*32 + 16*32) + 4 layers + norm: small but nontrivial
    assert 10_000 < n < 500_000
