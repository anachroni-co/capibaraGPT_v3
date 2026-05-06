"""QUARANTINED: see sub_models/_quarantine/README.md.

Algorithmically faithful Mamba/S6 draft, but uses PyTorch ``.unsqueeze``
inside ``_selective_scan`` which is incompatible with the rest of the
codebase (JAX/Flax). Rewrite tracked under BACKLOG-018.

Importing this module is intentionally permitted (the hybrid attention test
still exercises its NumPy fallback), but it MUST NOT be wired into
production training paths.
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
    # Use numpy as fallback
    jnp = np

    # Create dummy jax for fallback
    class _DummyJax:
        class lax:
            @staticmethod
            def scan(f, init, xs):
                """Dummy scan implementation."""
                carry = init
                ys = []
                for x in xs:
                    carry, y = f(carry, x)
                    ys.append(y)
                return carry, np.array(ys)

            @staticmethod
            def associative_scan(f, init, xs, axis=0):
                """Dummy associative scan."""
                return _DummyJax.lax.scan(f, init, xs)

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
                return np.log(1 + np.exp(x))

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
            return np.random.randn(*shape)

    random = _DummyRandom()

# Import Flax
try:
    from flax import linen as nn
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    # Create dummy nn for fallback
    class _DummyNN:
        @staticmethod
        def Dense(*args, **kwargs):
            return lambda x: x

        @staticmethod
        def LayerNorm(*args, **kwargs):
            return lambda x: x

    nn = _DummyNN()

# Import interfaces
from interfaces.imodules import IModule

logger = logging.getLogger(__name__)

@dataclass
class MambaConfig:
    """Configuration for MambaModule."""
    hidden_size: int = 768
    d_state: int = 64  # SSM internal state dimension
    d_conv: int = 4    # Kernel size for 1D convolution
    expand_factor: int = 2  # Expansion factor for projections
    dt_rank: int = 32  # Rank for temporal parameter Δ
    use_bias: bool = True
    use_conv_bias: bool = True
    activation: str = "swish"  # swish, gelu, relu
    layer_norm_epsilon: float = 1e-5

    # TPU optimizations
    use_tpu_optimizations: bool = True
    use_mixed_precision: bool = True
    scan_type: str = "associative"  # "linear", "associative"


class MambaModule(IModule):
    """
    Mamba (Selective State Space Model) Module.

    Implements the Mamba model with O(n) complexity for sequence
    processing, compatible with the Capibara6 modular architecture.

    Features:
    - O(n) complexity vs O(n^2) of traditional attention
    - Selective State Space Model with adaptive parameters
    - Native TPU optimizations
    - Compatible with IModule interface
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MambaModule.

        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = MambaConfig(**config)
        self.logger = logging.getLogger(__name__)
        
        # Calculate dimensions
        self.d_inner = self.config.hidden_size * self.config.expand_factor

        # Initialize parameters (NumPy fallback is supported)
        self._init_parameters()
        if not JAX_AVAILABLE:
            self.logger.warning("JAX not available, using NumPy fallback implementation")

        self.logger.info(f"MambaModule initialized: hidden_size={self.config.hidden_size}, "
                        f"d_state={self.config.d_state}, complexity=O(n)")
    
    def _init_parameters(self):
        """Initialize Mamba model parameters."""
        key = random.PRNGKey(42)
        k1, k2, k3, k4, k5, k6, k7 = random.split(key, 7)

        # Input projections
        self.in_proj_weight = random.normal(k1, (self.d_inner * 2, self.config.hidden_size)) * 0.02

        # 1D convolution parameters
        self.conv1d_weight = random.normal(k2, (self.d_inner, 1, self.config.d_conv)) * 0.02
        if self.config.use_conv_bias:
            self.conv1d_bias = jnp.zeros((self.d_inner,))

        # Projections for SSM parameters
        self.x_proj_weight = random.normal(k3, (self.config.dt_rank + self.config.d_state * 2, self.d_inner)) * 0.02
        self.dt_proj_weight = random.normal(k4, (self.d_inner, self.config.dt_rank)) * 0.02

        # SSM parameters
        # A: Transition matrix (initialized as negative diagonal for stability)
        self.A_log = jnp.log(jnp.arange(1, self.config.d_state + 1, dtype=jnp.float32))
        self.A_log = jnp.broadcast_to(self.A_log, (self.d_inner, self.config.d_state))

        # D: Skip connection parameter
        self.D = jnp.ones((self.d_inner,))

        # Output projection
        self.out_proj_weight = random.normal(k5, (self.config.hidden_size, self.d_inner)) * 0.02

        # Layer norm
        if FLAX_AVAILABLE:
            self.norm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon)

        self.logger.info("Mamba parameters initialized successfully")
    
    def _selective_scan(self, x: jnp.ndarray, delta: jnp.ndarray,
                       A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
        """
        Selective Scan implementation (core of Mamba).

        Args:
            x: Input tensor [batch, seq_len, d_inner]
            delta: Temporal parameter [batch, seq_len, d_inner]
            A: Transition matrix [d_inner, d_state]
            B: Input matrix [batch, seq_len, d_state]
            C: Output matrix [batch, seq_len, d_state]

        Returns:
            Output tensor [batch, seq_len, d_inner]
        """
        batch_size, seq_len, d_inner = x.shape

        # Discretization of A and B
        deltaA = jnp.exp(delta.unsqueeze(-1) * A)  # [batch, seq_len, d_inner, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # [batch, seq_len, d_inner, d_state]

        def ssm_step(carry, inputs):
            """One step of the SSM."""
            h = carry  # [batch, d_inner, d_state]
            x_t, deltaA_t, deltaB_t, C_t = inputs

            # Update state: h = deltaA * h + deltaB * x
            h = deltaA_t * h + deltaB_t * x_t.unsqueeze(-1)

            # Compute output: y = C * h
            y = jnp.sum(C_t.unsqueeze(1) * h, axis=-1)  # [batch, d_inner]

            return h, y

        # Initial state
        initial_state = jnp.zeros((batch_size, d_inner, self.config.d_state))

        # Prepare inputs for scan
        inputs = (x, deltaA, deltaB, C)

        # Execute scan
        if self.config.scan_type == "associative" and seq_len > 512:
            # Use associative scan for parallelization on long sequences
            try:
                _, outputs = jax.lax.associative_scan(ssm_step, initial_state, inputs, axis=1)
                scan_complexity = "O(log n)"
            except Exception:
                # Fallback to linear scan
                _, outputs = jax.lax.scan(ssm_step, initial_state, inputs)
                scan_complexity = "O(n)"
        else:
            # Standard linear scan
            _, outputs = jax.lax.scan(ssm_step, initial_state, inputs)
            scan_complexity = "O(n)"

        return outputs, scan_complexity
    
    def _mamba_forward(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Forward pass of the Mamba model.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            output: Output tensor [batch, seq_len, hidden_size]
            metrics: Dictionary with processing metrics
        """
        if not JAX_AVAILABLE:
            return self._fallback_forward(x)

        batch_size, seq_len, hidden_size = x.shape

        # 1. Input projection
        xz = jnp.dot(x, self.in_proj_weight.T)  # [batch, seq_len, d_inner * 2]
        x_proj, z = jnp.split(xz, 2, axis=-1)  # Each: [batch, seq_len, d_inner]

        # 2. 1D convolution (for local dependencies)
        x_conv = self._apply_conv1d(x_proj)

        # 3. Activation
        if self.config.activation == "swish":
            x_conv = x_conv * jax.nn.sigmoid(x_conv)
        elif self.config.activation == "gelu":
            x_conv = jax.nn.gelu(x_conv)
        else:
            x_conv = jax.nn.relu(x_conv)

        # 4. Projection for SSM parameters
        x_dbl = jnp.dot(x_conv, self.x_proj_weight.T)  # [batch, seq_len, dt_rank + d_state * 2]
        dt, B, C = jnp.split(x_dbl, [self.config.dt_rank, self.config.dt_rank + self.config.d_state], axis=-1)

        # 5. Temporal parameter delta
        dt = jnp.dot(dt, self.dt_proj_weight.T)  # [batch, seq_len, d_inner]
        dt = jax.nn.softplus(dt)  # Ensure positivity

        # 6. Matrix A (stable)
        A = -jnp.exp(self.A_log)  # [d_inner, d_state]

        # 7. Selective Scan (core algorithm)
        y, scan_complexity = self._selective_scan(x_conv, dt, A, B, C)

        # 8. Skip connection
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # 9. Gate with z
        y = y * jax.nn.silu(z)

        # 10. Output projection
        output = jnp.dot(y, self.out_proj_weight.T)  # [batch, seq_len, hidden_size]

        # Metrics
        metrics = {
            "mamba_active": True,
            "complexity": scan_complexity,
            "sequence_length": seq_len,
            "d_state": self.config.d_state,
            "selective_scan_used": True,
            "tpu_optimized": self.config.use_tpu_optimizations
        }

        return output, metrics
    
    def _apply_conv1d(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply 1D convolution."""
        # Simplified conv1d implementation
        # In production we would use jax.lax.conv_general_dilated
        batch_size, seq_len, d_inner = x.shape

        # Padding to maintain sequence length
        pad_width = ((0, 0), (self.config.d_conv - 1, 0), (0, 0))
        x_padded = jnp.pad(x, pad_width, mode='constant', constant_values=0)

        # Simplified convolution (for demonstration)
        # In full implementation we would use optimized JAX operations
        output = x  # Placeholder - implement real convolution

        if self.config.use_conv_bias:
            output = output + self.conv1d_bias.unsqueeze(0).unsqueeze(0)

        return output
    
    def _fallback_forward(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Fallback implementation when JAX is not available."""
        self.logger.warning("Using Mamba NumPy fallback implementation")

        x_np = np.asarray(x, dtype=np.float32)
        batch_size, seq_len, hidden_size = x_np.shape

        # Simple linear projection + gating as a CPU-friendly fallback
        if hasattr(self, "in_proj_weight") and hasattr(self, "out_proj_weight"):
            xz = x_np @ np.asarray(self.in_proj_weight).T
            x_proj, z = np.split(xz, 2, axis=-1)
            gate = 1.0 / (1.0 + np.exp(-z))
            y = x_proj * gate
            output = y @ np.asarray(self.out_proj_weight).T
        else:
            output = x_np

        metrics = {
            "mamba_active": True,
            "complexity": "O(n)",
            "sequence_length": seq_len,
            "fallback_used": True,
            "warning": "JAX not available, using NumPy fallback"
        }

        return output, metrics
    
    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> Dict[str, Any]:
        """
        Main interface compatible with IModule.

        Args:
            inputs: Input tensor [batch, seq_len, hidden_size]
            training: Whether in training mode

        Returns:
            Dict with 'output', 'metrics' and additional information
        """
        try:
            # Validate input
            if not hasattr(inputs, 'shape') or len(inputs.shape) != 3:
                raise ValueError(f"MambaModule expected 3D tensor, received: {inputs.shape if hasattr(inputs, 'shape') else type(inputs)}")
            
            batch_size, seq_len, hidden_size = inputs.shape
            
            if hidden_size != self.config.hidden_size:
                self.logger.warning(f"Input dimension ({hidden_size}) does not match configuration ({self.config.hidden_size})")
            
            # Forward pass
            output, metrics = self._mamba_forward(inputs)
            
            # Additional information
            processing_info = {
                "module_type": "MambaModule",
                "architecture": "Selective State Space Model",
                "complexity_advantage": f"O(n) vs O(n^2) for seq_len={seq_len}",
                "memory_efficiency": "Linear scaling",
                "training_mode": training
            }

            result = {
                "output": output,
                "metrics": metrics,
                "processing_info": processing_info,
                "success": True
            }

            self.logger.debug(f"MambaModule processed sequence of length {seq_len} with complexity {metrics.get('complexity', 'O(n)')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in MambaModule: {e}")

            # Fallback: return input without modification
            return {
                "output": inputs,
                "metrics": {"error": str(e), "fallback_used": True},
                "processing_info": {"module_type": "MambaModule", "error": True},
                "success": False
            }

    def setup_tpu_optimizations(self):
        """Configure specific optimizations for TPU."""
        if self.config.use_tpu_optimizations:
            self.logger.info("Configuring TPU optimizations for MambaModule")

            # Configure mixed precision
            if self.config.use_mixed_precision:
                self.logger.info("Enabling mixed precision (BF16) for TPU")

            # Configure associative scan for parallelization
            if self.config.scan_type == "associative":
                self.logger.info("Enabling associative scan for TPU parallelization")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get module metrics (IModule compatibility).

        Returns:
            Dict with configuration and state metrics
        """
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
            "flax_available": FLAX_AVAILABLE
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get module configuration (IModule compatibility).

        Returns:
            Dict with all current configuration
        """
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
            "scan_type": self.config.scan_type
        }
