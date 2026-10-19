"""
Layers Smoke Tests - Basic functionality tests for layer modules.

This module provides smoke tests for layer implementations, validating
that layers can be initialized and perform forward passes correctly.

Author: Skydesk International Dev Team.
"""

import pytest

from layers import SelfAttention, SelfAttentionConfig
from layers.jax_compat import jax, jnp, JAX_AVAILABLE


def test_self_attention_smoke():
    if not JAX_AVAILABLE:
        pytest.skip("JAX/Flax not available")
    x = jnp.ones((1, 4, 8))
    layer = SelfAttention(SelfAttentionConfig(num_heads=2, hidden_size=8))
    params = layer.init(jax.random.PRNGKey(0), x)
    y = layer.apply(params, x)
    assert y.shape == x.shape
