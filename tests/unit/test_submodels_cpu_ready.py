"""
Submodels CPU Ready Tests - Unit tests for CPU-ready submodel implementations.

This module provides tests for verifying that submodels work correctly
on CPU without requiring TPU/GPU hardware. Tests Mamba, Hybrid, and
DeepDialog modules.

Author: Skydesk International Dev Team.
"""

import asyncio
import numpy as np


def test_mamba_cpu_ready():
    from sub_models.mamba.mamba_module import MambaModule

    x = np.ones((1, 4, 8), dtype=np.float32)
    module = MambaModule({"hidden_size": 8, "d_state": 4, "d_conv": 2, "expand_factor": 2})
    result = module(x)
    assert result["output"].shape == x.shape


def test_hybrid_cpu_ready():
    from sub_models.hybrid.hybrid_attention_module import HybridAttentionModule

    x = np.ones((1, 4, 8), dtype=np.float32)
    module = HybridAttentionModule({"hidden_size": 8, "num_heads": 2, "mamba_threshold": 2})
    result = module(x)
    assert result["output"].shape == x.shape


def test_deep_dialog_cpu_ready():
    from sub_models.deep_dialog import DeepDialog, DeepDialogConfig, JAX_AVAILABLE

    x = np.ones((1, 4, 8), dtype=np.float32)
    config = DeepDialogConfig(hidden_size=8, num_layers=2, num_heads=2)

    if JAX_AVAILABLE:
        import jax
        import jax.numpy as jnp
        model = DeepDialog(config)
        params = model.init(jax.random.PRNGKey(0), jnp.asarray(x))
        y = model.apply(params, jnp.asarray(x))
        assert y.shape == x.shape
    else:
        model = DeepDialog(config)
        y = model(x)
        assert y.shape == x.shape


def test_reasoning_expert_cpu_ready():
    from sub_models.reasoning_enhancement import ReasoningEnhancementExpert
    from interfaces.isub_models import ExpertContext

    expert = ReasoningEnhancementExpert()
    assert expert.initialize({})

    context = ExpertContext(
        text="Solve a simple logic problem",
        task_hint="reasoning",
        constraints=None,
        flags=None,
    )

    result = asyncio.run(expert.process(context))
    assert result is not None


def test_csa_expert_cpu_ready():
    from sub_models.csa_expert import CSAExpert

    expert = CSAExpert()
    x = np.ones((1, 2, 3), dtype=np.float32)
    y = expert(x)
    assert hasattr(y, "shape")
    assert y.shape == x.shape
