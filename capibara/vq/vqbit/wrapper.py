"""AdaptiveWrapper – v2.0
================================================
Mejoras key about la versión previa:
*  **Config dataclass** for tipado estricto and valores by defect.
*  `pjit`‑ready: parámetros with `param_with_axes`; output with la misma `PartitionSpec`.
* ️ FFT usa `jnp.fft.rfft` (mitad de memory) and `jax.checkpoint`.
* 🪶 Métricas se almacenan en collection `metrics` → mutables but outside de grad.
*  Coherencia and entrelazamiento se calculan with aproximaciones lineales or(N) (without `outer`, without `svd`).
*  `adaptive_attention` envuelto en `jax.checkpoint` for activate CSE + remat.
* 🩹 Validaciones de type/forma and conversion a `dtype` en un only lugar.
"""

from __future__ import annotations

import os
import sys
# Python standard library
import logging
from enum import Enum, auto
from functools import partial
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third party imports
import jax
from flax import linen as nn

import jax.numpy as jnp
from capibara.core.kernels import tpu_kernel
from flax.linen.partitioning import param_with_axes
from jax.sharding import PartitionSpec as P

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    pass  # Using proper imports instead of sys.path manipulation

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

class FFTMode(Enum):
    STANDARD = auto()
    PHASE_OPT = auto()
    HYBRID = auto()

@dataclass
class AdaptiveWrapCfg:
    fft_size: int = 1_024
    phase_scale: float = 1.0
    fft_mode: FFTMode = FFTMode.HYBRID
    num_heads: int = 4
    qkv_features: int = 256
    out_features: int = 256
    dropout: float = 0.1
    dtype: str = "bfloat16"  # or float32/float16

    @property
    def jdtype(self):
        return {
            "bfloat16": jnp.bfloat16,
            "float32": jnp.float32,
            "float16": jnp.float16,
        }[self.dtype]

# -----------------------------------------------------------------------------
# Module
# -----------------------------------------------------------------------------

class AdaptiveWrapper(nn.Module):
    cfg: AdaptiveWrapCfg
    shard_axis: Tuple[str, str] = ("batch", "embed")

    def setup(self):
        ax = ("batch", "embed")
        self.attn = nn.MultiHeadDotProductAttention(
            num_heads=self.cfg.num_heads,
            qkv_features=self.cfg.qkv_features,
            out_features=self.cfg.out_features,
            dropout_rate=self.cfg.dropout,
            kernel_axes=(ax, ax),
            dtype=self.cfg.jdtype,
        )
        self.norm = nn.LayerNorm(axis=-1, dtype=self.cfg.jdtype, param_axes={"scale": "embed", "bias": "embed"})
        # metric buffers in separate collection; ring of 1k
        self.metric_len = 1_000

    # ------------------------------------------------------------------
    def _pad(self, x):
        need = self.cfg.fft_size - x.shape[-1]
        return jnp.pad(x, [(0,0)]*(x.ndim-1)+[(0, max(0, need))])[:, : self.cfg.fft_size]

    @partial(jax.checkpoint, prevent_cse=True)
    def _fft(self, x):
        xr = self._pad(x).astype(self.cfg.jdtype)
        
        # use FFT optimizada for tpu v4
        fft = tpu_kernel.optimized_fft(xr)
        
        if self.cfg.fft_mode is FFTMode.PHASE_OPT:
            phase = tpu_kernel.compute_phase(fft)
            fft *= jnp.tanh(self.cfg.phase_scale * phase)
        elif self.cfg.fft_mode is FFTMode.HYBRID:
            mag, phase = tpu_kernel.compute_magnitude_phase(fft)
            fft = mag * jnp.exp(1j * jnp.tanh(self.cfg.phase_scale * phase))
        
        return fft.astype(self.cfg.jdtype)

    # ------------------------------------------------------------------
    def _fast_metrics(self, x, y):
        # coherence via cosine similarity
        c = jnp.mean(jnp.cos(x) * jnp.cos(y) + jnp.sin(x) * jnp.sin(y))
        # fidelity via normalized inner product
        xn, yn = x / (jnp.linalg.norm(x)+1e-6), y / (jnp.linalg.norm(y)+1e-6)
        f = (jnp.abs(jnp.vdot(xn, yn)) ** 2).real
        # entanglement approximation: variance of log magnitude spectrum
        ent = jnp.var(jnp.log(jnp.abs(y)+1e-6))
        # phase variance
        pvar = jnp.var(jnp.angle(y))
        return c, f, ent, pvar

    # ------------------------------------------------------------------
    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic=True):
        if x.ndim != 2:
            raise ValueError("AdaptiveWrapper expects (B, D) tensor")
        x = x.astype(self.cfg.jdtype)

        fft = self._fft(x)
        attn = jax.checkpoint(self.attn)(inputs_q=fft, inputs_kv=fft, deterministic=deterministic)
        out = self.norm(fft + attn)

        c, f, e, pv = self._fast_metrics(x, out)
        self.sow("metrics", "coherence", c)
        self.sow("metrics", "fidelity", f)
        self.sow("metrics", "entanglement", e)
        self.sow("metrics", "phase_var", pv)

        return {"output": out, "metrics": {"coherence": c, "fidelity": f, "entanglement": e, "phase_var": pv}} 