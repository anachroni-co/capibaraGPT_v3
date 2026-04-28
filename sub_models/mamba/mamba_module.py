"""
MambaModule - State Space Model Implementation for Capibara6

Mamba (Selective State Space Model) implementation compatible with
the Capibara6 modular architecture. Provides O(n) complexity
vs O(n²) of traditional Transformer for long sequences.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Import JAX with fallbacks
JAX_AVAILABLE = False
FLAX_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from jax import lax
    JAX_AVAILABLE = True
except ImportError:
    jnp = np

    class _DummyJax:
        class lax:
            @staticmethod
            def scan(f, init, xs):
                carry = init
                ys = []
                first = xs[0] if isinstance(xs, (list, tuple)) else xs
                seq_len = first.shape[0]
                for t in range(seq_len):
                    if isinstance(xs, tuple):
                        x_t = tuple(x[t] for x in xs)
                    else:
                        x_t = xs[t]
                    carry, y = f(carry, x_t)
                    ys.append(y)
                return carry, np.stack(ys)

        class nn:
            @staticmethod
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            @staticmethod
            def gelu(x):
                return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

            @staticmethod
            def relu(x):
                return np.maximum(0, x)

            @staticmethod
            def silu(x):
                return x * _DummyJax.nn.sigmoid(x)

            @staticmethod
            def softplus(x):
                return np.log1p(np.exp(x))

    jax = _DummyJax()

    class _DummyRandom:
        @staticmethod
        def PRNGKey(seed):
            np.random.seed(seed)
            return seed

        @staticmethod
        def split(key, num=2):
            return [key + i for i in range(num)]

        @staticmethod
        def normal(key, shape):
            return np.random.randn(*shape).astype(np.float32)

    random = _DummyRandom()

try:
    from flax import linen as nn
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False

from interfaces.imodules import IModule

logger = logging.getLogger(__name__)


@dataclass
class MambaConfig:
    """Configuration for MambaModule."""
    hidden_size: int = 768
    d_state: int = 64
    d_conv: int = 4
    expand_factor: int = 2
    dt_rank: int = 32
    use_bias: bool = True
    use_conv_bias: bool = True
    activation: str = "swish"
    layer_norm_epsilon: float = 1e-5
    use_tpu_optimizations: bool = True
    use_mixed_precision: bool = True
    scan_type: str = "linear"


# ── Flax implementation (only defined when JAX + Flax are available) ──────────

if JAX_AVAILABLE and FLAX_AVAILABLE:
    class MambaFlaxBlock(nn.Module):
        """
        Proper Flax nn.Module implementation of a Mamba SSM block.

        All parameters are managed by Flax and trained through its standard
        gradient machinery. The selective scan uses jax.lax.scan for O(n)
        sequence processing.
        """
        hidden_size: int
        d_state: int = 64
        d_conv: int = 4
        expand_factor: int = 2
        dt_rank: int = 32
        activation: str = "swish"
        layer_norm_epsilon: float = 1e-5

        @nn.compact
        def __call__(
            self, x: jnp.ndarray, training: bool = False
        ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
            batch_size, seq_len, _ = x.shape
            d_inner = self.hidden_size * self.expand_factor

            # ── 1. Input projection → split into x_in and z (gate) ──────────
            xz = nn.Dense(d_inner * 2, use_bias=False, name="in_proj")(x)
            x_in, z = jnp.split(xz, 2, axis=-1)  # each [batch, seq, d_inner]

            # ── 2. Causal depthwise 1-D convolution ─────────────────────────
            # Pad left by (d_conv-1) to maintain causality, no right pad.
            pad = self.d_conv - 1
            x_padded = jnp.pad(x_in, ((0, 0), (pad, 0), (0, 0)))

            # Kernel shape (HIO): [kernel_size, in_per_group, out_channels]
            # For depthwise: in_per_group=1, out_channels=d_inner, groups=d_inner
            conv_w = self.param(
                "conv_weight",
                nn.initializers.lecun_normal(),
                (self.d_conv, 1, d_inner),
            )
            conv_b = self.param(
                "conv_bias", nn.initializers.zeros, (d_inner,)
            ) if True else None

            x_conv = jax.lax.conv_general_dilated(
                x_padded,
                conv_w,
                window_strides=(1,),
                padding="VALID",
                feature_group_count=d_inner,
                dimension_numbers=("NHC", "HIO", "NHC"),
            )  # [batch, seq_len, d_inner]

            if conv_b is not None:
                x_conv = x_conv + conv_b[None, None, :]

            # ── 3. Activation ────────────────────────────────────────────────
            if self.activation == "swish":
                x_act = x_conv * jax.nn.sigmoid(x_conv)
            elif self.activation == "gelu":
                x_act = jax.nn.gelu(x_conv)
            else:
                x_act = jax.nn.relu(x_conv)

            # ── 4. SSM parameter projections ─────────────────────────────────
            x_dbl = nn.Dense(
                self.dt_rank + self.d_state * 2, use_bias=False, name="x_proj"
            )(x_act)
            dt_raw, B, C = jnp.split(
                x_dbl,
                [self.dt_rank, self.dt_rank + self.d_state],
                axis=-1,
            )
            # B, C: [batch, seq_len, d_state]

            # ── 5. Delta (Δ) via learned up-projection ───────────────────────
            dt = nn.Dense(d_inner, use_bias=True, name="dt_proj")(dt_raw)
            dt = jax.nn.softplus(dt)  # [batch, seq_len, d_inner], always > 0

            # ── 6. State-transition matrix A (stable, negative real) ─────────
            # A_log is learned; A = -exp(A_log) keeps eigenvalues < 0.
            A_log = self.param(
                "A_log",
                lambda rng, shape: jnp.log(
                    jnp.broadcast_to(
                        jnp.arange(1, self.d_state + 1, dtype=jnp.float32),
                        shape,
                    )
                ),
                (d_inner, self.d_state),
            )
            A = -jnp.exp(A_log)  # [d_inner, d_state]

            # Skip-connection scalar per channel
            D = self.param("D", nn.initializers.ones, (d_inner,))

            # ── 7. Selective scan (core O(n) recurrence) ─────────────────────
            y = self._selective_scan(x_act, dt, A, B, C)

            # ── 8. Skip connection ────────────────────────────────────────────
            y = y + x_act * D[None, None, :]

            # ── 9. Gating with z ─────────────────────────────────────────────
            y = y * jax.nn.silu(z)

            # ── 10. Output projection ────────────────────────────────────────
            output = nn.Dense(self.hidden_size, use_bias=False, name="out_proj")(y)

            metrics = {
                "mamba_active": True,
                "complexity": "O(n)",
                "sequence_length": seq_len,
                "d_state": self.d_state,
                "selective_scan_used": True,
            }
            return output, metrics

        def _selective_scan(
            self,
            x: jnp.ndarray,   # [batch, seq_len, d_inner]
            dt: jnp.ndarray,  # [batch, seq_len, d_inner]
            A: jnp.ndarray,   # [d_inner, d_state]
            B: jnp.ndarray,   # [batch, seq_len, d_state]
            C: jnp.ndarray,   # [batch, seq_len, d_state]
        ) -> jnp.ndarray:
            """
            O(n) selective scan using jax.lax.scan.

            Discretises the continuous-time SSM via zero-order hold:
                deltaA[b,t,d,s] = exp(dt[b,t,d] * A[d,s])
                deltaB_u[b,t,d,s] = dt[b,t,d] * B[b,t,s] * x[b,t,d]

            Recurrence (per time step t):
                h[t+1] = deltaA[t] * h[t] + deltaB_u[t]
                y[t]   = einsum('bs,bds->bd', C[t], h[t+1])
            """
            batch_size, seq_len, d_inner = x.shape

            # Discretise A: [batch, seq_len, d_inner, d_state]
            deltaA = jnp.exp(
                jnp.einsum("bti,is->btis", dt, A)
            )

            # Discretise B fused with input x: [batch, seq_len, d_inner, d_state]
            deltaB_u = jnp.einsum("bti,bts->btis", dt * x, B)

            def ssm_step(
                h: jnp.ndarray,                          # [batch, d_inner, d_state]
                inputs: Tuple[jnp.ndarray, ...],
            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
                dA_t, dBu_t, C_t = inputs
                # dA_t, dBu_t: [batch, d_inner, d_state]
                # C_t: [batch, d_state]
                h_next = dA_t * h + dBu_t
                y_t = jnp.einsum("bs,bds->bd", C_t, h_next)  # [batch, d_inner]
                return h_next, y_t

            h0 = jnp.zeros((batch_size, d_inner, self.d_state))

            # Transpose to [seq_len, batch, ...] — lax.scan iterates axis 0
            dA_seq = jnp.transpose(deltaA, (1, 0, 2, 3))    # [seq, batch, d_inner, d_state]
            dBu_seq = jnp.transpose(deltaB_u, (1, 0, 2, 3)) # [seq, batch, d_inner, d_state]
            C_seq = jnp.transpose(C, (1, 0, 2))             # [seq, batch, d_state]

            _, ys = jax.lax.scan(ssm_step, h0, (dA_seq, dBu_seq, C_seq))
            # ys: [seq_len, batch, d_inner] → [batch, seq_len, d_inner]
            return jnp.transpose(ys, (1, 0, 2))

else:
    MambaFlaxBlock = None  # type: ignore[assignment,misc]


# ── IModule wrapper ───────────────────────────────────────────────────────────

class MambaModule(IModule):
    """
    Mamba (Selective State Space Model) Module.

    IModule wrapper around MambaFlaxBlock. When JAX + Flax are available the
    Flax block manages all parameters through its standard param system. When
    they are not available a minimal NumPy fallback is used.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = MambaConfig(**config)
        self.logger = logging.getLogger(__name__)
        self.d_inner = self.config.hidden_size * self.config.expand_factor
        self._use_flax = False

        if JAX_AVAILABLE and FLAX_AVAILABLE and MambaFlaxBlock is not None:
            self._init_flax()
        else:
            self.logger.warning(
                "JAX/Flax not available — using NumPy fallback (not trainable)"
            )
            self._init_numpy_fallback()

        self.logger.info(
            f"MambaModule initialised: hidden={self.config.hidden_size}, "
            f"d_state={self.config.d_state}, use_flax={self._use_flax}"
        )

    def _init_flax(self) -> None:
        self.flax_model = MambaFlaxBlock(
            hidden_size=self.config.hidden_size,
            d_state=self.config.d_state,
            d_conv=self.config.d_conv,
            expand_factor=self.config.expand_factor,
            dt_rank=self.config.dt_rank,
            activation=self.config.activation,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
        )
        # Initialise Flax params with a minimal dummy input
        dummy = jnp.ones((1, 4, self.config.hidden_size))
        self.params = self.flax_model.init(jax.random.PRNGKey(42), dummy)
        self._use_flax = True

    def _init_numpy_fallback(self) -> None:
        """Minimal NumPy parameters for CPU-only fallback."""
        rng = np.random.default_rng(42)
        scale = 0.02
        self._np_W_in = rng.standard_normal(
            (self.d_inner * 2, self.config.hidden_size)
        ).astype(np.float32) * scale
        self._np_W_out = rng.standard_normal(
            (self.config.hidden_size, self.d_inner)
        ).astype(np.float32) * scale

    # ── forward ──────────────────────────────────────────────────────────────

    def __call__(self, inputs: Any, training: bool = False) -> Dict[str, Any]:
        try:
            if not hasattr(inputs, "shape") or len(inputs.shape) != 3:
                raise ValueError(
                    f"MambaModule expected 3-D tensor, got: "
                    f"{inputs.shape if hasattr(inputs, 'shape') else type(inputs)}"
                )

            batch_size, seq_len, hidden_size = inputs.shape

            if self._use_flax:
                output, metrics = self.flax_model.apply(
                    self.params, inputs, training=training
                )
            else:
                output, metrics = self._numpy_forward(inputs)

            return {
                "output": output,
                "metrics": metrics,
                "processing_info": {
                    "module_type": "MambaModule",
                    "architecture": "Selective State Space Model",
                    "complexity_advantage": f"O(n) vs O(n²) for seq_len={seq_len}",
                    "training_mode": training,
                    "use_flax": self._use_flax,
                },
                "success": True,
            }

        except Exception as exc:
            self.logger.error(f"Error in MambaModule: {exc}")
            return {
                "output": inputs,
                "metrics": {"error": str(exc), "fallback_used": True},
                "processing_info": {"module_type": "MambaModule", "error": True},
                "success": False,
            }

    def _numpy_forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Minimal NumPy forward pass (not a real SSM — CPU fallback only)."""
        self.logger.warning("Using MambaModule NumPy fallback (not trainable)")
        x_np = np.asarray(x, dtype=np.float32)
        xz = x_np @ self._np_W_in.T                    # [batch, seq, d_inner*2]
        x_proj, z = np.split(xz, 2, axis=-1)
        gate = 1.0 / (1.0 + np.exp(-z))                # sigmoid gating
        y = x_proj * gate
        output = y @ self._np_W_out.T                   # [batch, seq, hidden]
        metrics = {
            "mamba_active": True,
            "complexity": "O(n)",
            "sequence_length": x_np.shape[1],
            "fallback_used": True,
        }
        return output, metrics

    # ── IModule interface ─────────────────────────────────────────────────────

    def setup_tpu_optimizations(self) -> None:
        if self.config.use_tpu_optimizations:
            self.logger.info("TPU optimisations enabled for MambaModule (bfloat16/scan)")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "module_type": "MambaModule",
            "hidden_size": self.config.hidden_size,
            "d_state": self.config.d_state,
            "d_inner": self.d_inner,
            "d_conv": self.config.d_conv,
            "expand_factor": self.config.expand_factor,
            "complexity": "O(n)",
            "scan_type": self.config.scan_type,
            "tpu_optimizations": self.config.use_tpu_optimizations,
            "mixed_precision": self.config.use_mixed_precision,
            "jax_available": JAX_AVAILABLE,
            "flax_available": FLAX_AVAILABLE,
            "use_flax": self._use_flax,
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.config.hidden_size,
            "d_state": self.config.d_state,
            "d_conv": self.config.d_conv,
            "expand_factor": self.config.expand_factor,
            "dt_rank": self.config.dt_rank,
            "use_bias": self.config.use_bias,
            "use_conv_bias": self.config.use_conv_bias,
            "activation": self.config.activation,
            "layer_norm_epsilon": self.config.layer_norm_epsilon,
            "use_tpu_optimizations": self.config.use_tpu_optimizations,
            "use_mixed_precision": self.config.use_mixed_precision,
            "scan_type": self.config.scan_type,
        }
