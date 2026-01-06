"""Optimization Module for CapibaraGPT Training.

This module provides advanced optimization functionality for training CapibaraGPT models,
including Gradient Centralization, memory optimizations, profiling, and training state
management. It integrates with JAX/Flax for efficient TPU/GPU training.

The optimization system features:
- Gradient Centralization (GC) for improved convergence
- Training state management with metrics tracking
- Early stopping with best model checkpointing
- Memory-efficient batch creation
- TensorBoard and Weights & Biases logging integration
- Distributed training support via JAX

Key Components:
    - TrainingMetrics: Extended training metrics dataclass
    - TrainingState: Enhanced Flax TrainState with metrics and profiling
    - EarlyStopping: Early stopping with model checkpointing
    - apply_gc: Gradient Centralization application
    - train_step: Training step with GC and profiling
    - val_step: Validation step function
    - create_dummy_batch: Dummy batch creation for initialization

Example:
    Basic training setup:

    >>> from capibara.core.optimization import TrainingState, train_step
    >>> from capibara.config import Config
    >>>
    >>> # Create training state
    >>> config = Config()
    >>> state = TrainingState.create(
    ...     apply_fn=model.apply,
    ...     params=params,
    ...     tx=optimizer,
    ...     config=config
    ... )
    >>>
    >>> # Training step
    >>> state, metrics = train_step(state, batch, dropout_rng)
    >>> print(f"Loss: {metrics['loss']:.4f}")

    Early stopping:

    >>> from capibara.core.optimization import EarlyStopping
    >>>
    >>> early_stop = EarlyStopping(config)
    >>> should_stop, best_state = early_stop(val_loss, state)
    >>> if should_stop:
    ...     print("Early stopping triggered")
    ...     state = best_state  # Restore best model

Note:
    This module requires JAX, Flax, and Optax for core functionality. Optional
    dependencies include TensorBoard (tensorboardX) and Weights & Biases (wandb)
    for logging.

    The module automatically handles distributed training coordination, ensuring
    logging only occurs on the main process (process index 0).

See Also:
    - capibara.config: Configuration management
    - capibara.training: High-level training loops
    - Flax documentation: https://flax.readthedocs.io/
    - Optax documentation: https://optax.readthedocs.io/
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, Union, Callable
# Get current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import logging
# from flax import struct  # TODO: Use capibara's struct implementation
try:
    from flax import struct
except ImportError:
    # Fallback implementation
    import dataclasses
    class struct:
        @staticmethod
        def dataclass(cls):
            return dataclasses.dataclass(cls)
try:
    import ndb #type: ignore
except ImportError:
    ndb = None
from pathlib import Path
try:
    import optax #type: ignore
except ImportError:
    optax = None
try:
    from capibara.jax import ng
except ImportError:
    ng = None
from capibara.jax import jax
from capibara.jax import numpy as jnp
try:
    from flax.training import train_state
except ImportError:
    train_state = None
try:
    from capibara.config import Config, GCConfig
except ImportError:
    Config = None
    GCConfig = None 
try:
    from tensorboardX import SummaryWriter #type: ignore
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    class _WandbFallback:
        run = None
    wandb = _WandbFallback()
from capibara.jax.sharding import Mesh, PartitionSpec

# Define TrainingState fallback if not available
if train_state is None:
    class TrainingState:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
else:
    # Flax changed TrainingState to TrainState
    if hasattr(train_state, 'TrainState'):
        TrainingState = train_state.TrainState
    elif hasattr(train_state, 'TrainingState'):
        TrainingState = train_state.TrainingState
    else:
        # Fallback implementation
        class TrainingState:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@struct.dataclass
class TrainingMetrics:
    """Extended training metrics for monitoring model performance.

    This dataclass tracks comprehensive training metrics including gradients,
    learning rate, validation performance, and throughput statistics.

    Attributes:
        grad_norm (float): Gradient norm for monitoring gradient flow.
        learning_rate (float): Current learning rate value.
        val_loss (float): Validation loss.
        val_perplexity (float): Validation perplexity.
        throughput (float): Training throughput (tokens/second).
        step_time (float): Time per training step in seconds.
        tokens_per_second (float): Tokens processed per second.

    Example:
        >>> metrics = TrainingMetrics(
        ...     grad_norm=1.5,
        ...     learning_rate=0.001,
        ...     throughput=1000.0
        ... )
    """
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    val_loss: float = 0.0
    val_perplexity: float = 0.0
    throughput: float = 0.0
    step_time: float = 0.0
    tokens_per_second: float = 0.0

def apply_gc(g: jnp.ndarray, layer_name: Optional[str] = None, gc_config = None) -> jnp.ndarray:
    """Apply Gradient Centralization to a gradient tensor.

    Gradient Centralization normalizes gradients by subtracting their mean,
    which can improve training stability and convergence.

    Args:
        g (jnp.ndarray): Gradient tensor to centralize.
        layer_name (Optional[str], optional): Layer name for layer-specific
            configuration. Defaults to None.
        gc_config: Gradient Centralization configuration object with 'enabled'
            and 'layer_specific' attributes.

    Returns:
        jnp.ndarray: Centralized gradient tensor, or original if GC disabled.

    Example:
        >>> import jax.numpy as jnp
        >>> from capibara.config import GCConfig
        >>>
        >>> gc_config = GCConfig(enabled=True)
        >>> grad = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> centered_grad = apply_gc(grad, gc_config=gc_config)

    Note:
        GC is only applied to tensors with ndim > 1. Scalar and 1D tensors
        are returned unchanged. Layer-specific disabling takes precedence
        over global enable flag.
    """
    if not gc_config.enabled:
        return g
    
    # Safe validation of layer_specific configuration
    if layer_name is not None and layer_name in gc_config.layer_specific:
        if not gc_config.layer_specific[layer_name]:
            return g

    return g - jnp.mean(g, axis=tuple(range(1, g.ndim)), keepdims=True) if g.ndim > 1 else g

def create_dummy_batch(config: Config) -> Dict[str, jnp.ndarray]:
    """Create a dummy batch with correct precision for initialization.

    Creates a synthetic batch of data matching the configured batch size,
    sequence length, and data type. Useful for model initialization and
    shape inference.

    Args:
        config (Config): Model configuration containing training parameters
            (batch_size, seq_length) and precision settings.

    Returns:
        Dict[str, jnp.ndarray]: Dictionary containing:
            - 'input_ids': Token IDs tensor of shape (batch_size, seq_length)
            - 'attention_mask': Attention mask of shape (batch_size, seq_length)
            - 'labels': Label tensor of shape (batch_size, seq_length)
            All tensors filled with ones and using the configured dtype.

    Example:
        >>> from capibara.config import Config
        >>>
        >>> config = Config()
        >>> batch = create_dummy_batch(config)
        >>> print(batch['input_ids'].shape)
        >>> print(batch['input_ids'].dtype)

    Note:
        All tensors are filled with ones (not zeros) to avoid numerical
        issues during initialization. The dtype is determined by
        config.get_precision_dtype().
    """
    dtype = config.get_precision_dtype()
    return {
        'input_ids': jnp.ones((config.training.batch_size, config.training.seq_length), dtype=dtype),
        'attention_mask': jnp.ones((config.training.batch_size, config.training.seq_length), dtype=dtype),
        'labels': jnp.ones((config.training.batch_size, config.training.seq_length), dtype=dtype)
    }

class EarlyStopping:
    """Early stopping with best model checkpointing.

    Monitors validation loss and stops training when no improvement is observed
    for a specified number of epochs (patience). Automatically saves the best
    model state to disk.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in loss to qualify as improvement.
        best_model_path (Path): Path to save best model checkpoint.
        best_loss (float): Best validation loss observed so far.
        counter (int): Number of epochs without improvement.
        best_state (Optional[TrainingState]): Training state with best loss.

    Example:
        >>> from capibara.core.optimization import EarlyStopping
        >>> from capibara.config import Config
        >>>
        >>> config = Config()
        >>> early_stop = EarlyStopping(config)
        >>>
        >>> for epoch in range(100):
        ...     # Train and validate
        ...     val_loss = validate(model, val_data)
        ...     should_stop, best_state = early_stop(val_loss, state)
        ...     if should_stop:
        ...         print(f"Early stopping at epoch {epoch}")
        ...         state = best_state  # Restore best model
        ...         break

    Note:
        The best model is automatically saved to disk when a new best loss
        is achieved. The saved state can be restored later for inference
        or continued training.
    """

    def __init__(self, config: Config):
        """Initialize early stopping with configuration.

        Args:
            config (Config): Configuration containing early stopping parameters:
                - training.early_stopping_patience: Epochs to wait
                - training.early_stopping_min_delta: Minimum improvement
                - training.best_model_dir: Directory to save best model
        """
        self.patience = config.training.early_stopping_patience
        self.min_delta = config.training.early_stopping_min_delta
        self.best_model_path = Path(config.training.best_model_dir)
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state: Optional[TrainingState] = None

    def __call__(
        self,
        current_loss: float,
        state: TrainingState
    ) -> Tuple[bool, Optional[TrainingState]]:
        """Check if training should stop based on current loss.

        Args:
            current_loss (float): Current validation loss.
            state (TrainingState): Current training state to potentially save.

        Returns:
            Tuple[bool, Optional[TrainingState]]: Tuple containing:
                - should_stop (bool): True if training should stop
                - best_state (Optional[TrainingState]): Best model state if stopping,
                  None otherwise

        Example:
            >>> early_stop = EarlyStopping(config)
            >>> should_stop, best_state = early_stop(0.5, state)
            >>> if should_stop:
            ...     print("Stopping training")
            ...     state = best_state

        Note:
            Loss improvement is checked with min_delta tolerance. The counter
            is reset when improvement is detected.
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.best_state = state
            # Save best model to disk
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.best_model_path, 'wb') as f:
                f.write(jax.serialization.to_bytes(state))
            return False, None

        self.counter += 1
        if self.counter >= self.patience:
            return True, self.best_state

        return False, None

if train_state is not None:
    class TrainingState(train_state.TrainState):
        """Extended training state with metrics and profiling support.

        Extends Flax's TrainState to include training metrics, configuration,
        validation step function, and TensorBoard writer for comprehensive
        training monitoring.

        Attributes:
            metrics (TrainingMetrics): Training metrics tracker.
            config (Optional[Config]): Model and training configuration.
            val_step_fn (Optional[Callable]): Validation step function.
            last_val_step (int): Last step where validation was performed.
            writer (Optional[SummaryWriter]): TensorBoard writer for logging.

        Example:
            >>> state = TrainingState.create(
            ...     apply_fn=model.apply,
            ...     params=params,
            ...     tx=optimizer,
            ...     config=config
            ... )

        Note:
            This class is only available when Flax's train_state module is
            installed. Otherwise, a fallback implementation is used.
        """
        metrics: TrainingMetrics
        config: Optional[Config] = struct.field(pytree_node=False, default=None)
        val_step_fn: Optional[Callable[[Any, Dict[str, jnp.ndarray]], Dict[str, float]]] = struct.field(pytree_node=False, default=None)
        last_val_step: int = 0
        writer: Optional[SummaryWriter] = struct.field(pytree_node=False, default=None)
    
    @classmethod
    def create(
        cls,
        *,
        apply_fn,
        params,
        tx,
        config: Config,
        metrics: Optional[TrainingMetrics] = None,
        writer: Optional[SummaryWriter] = None,
        **kwargs
    ):
        if metrics is None:
            metrics = TrainingMetrics()
        return super().create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            config=config,
            metrics=metrics,
            writer=writer,
            **kwargs
        )

def train_step(
    state: TrainingState,
    batch: Dict[str, jnp.ndarray],
    dropout_rng: jax.random.PRNGKey
) -> Tuple[TrainingState, Dict[str, float]]:
    """Training step with Gradient Centralization and profiling.

    Performs a single training step including forward pass, loss computation,
    backpropagation, gradient centralization, clipping, and optimizer update.
    Tracks comprehensive metrics including throughput and tokens per second.

    Args:
        state (TrainingState): Current training state containing model params
            and optimizer state.
        batch (Dict[str, jnp.ndarray]): Training batch with keys:
            - 'input_ids': Input token IDs
            - 'attention_mask': Attention mask
            - 'labels': Target labels
        dropout_rng (jax.random.PRNGKey): Random key for dropout.

    Returns:
        Tuple[TrainingState, Dict[str, float]]: Updated training state and
            metrics dictionary containing:
            - 'loss': Training loss
            - 'grad_norm': Gradient norm
            - 'learning_rate': Current learning rate
            - 'throughput': Tokens per second
            - 'tokens_per_second': Processing speed
            - 'step_time': Step duration in seconds

    Example:
        >>> import jax
        >>> rng = jax.random.PRNGKey(0)
        >>> state, metrics = train_step(state, batch, rng)
        >>> print(f"Loss: {metrics['loss']:.4f}")
        >>> print(f"Throughput: {metrics['throughput']:.1f} tok/s")

    Note:
        - Gradient Centralization is applied based on config.gc.apply_every
        - Gradients are clipped to config.training.gradient_clip_norm
        - Logging to W&B and TensorBoard only occurs on process 0
        - Step timing includes full forward/backward pass and updates
    """
    start_time = time.time()
    
    def loss_fn(params):
        logits = state.apply_fn(
            params,
            batch['input_ids'],
            batch['attention_mask'],
            rngs={'dropout': dropout_rng},
            training=True
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['labels']
        ).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    # Apply Gradient Centralization according to configuration
    if state.step % state.config.gc.apply_every == 0:
        grads = jax.tree_util.tree_map_with_path(
            lambda path, g: apply_gc(g, '/'.join(str(p) for p in path), state.config.gc),
            grads
        )

    # Apply gradient clipping using optax.global_norm
    grad_norm = optax.global_norm(grads)
    grads = jax.tree_util.tree_map(
        lambda g: jnp.clip(g, -state.config.training.gradient_clip_norm, state.config.training.gradient_clip_norm),
        grads
    )

    # Update training state
    state = state.apply_gradients(grads=grads)

    # Calculate throughput and tokens per second
    step_time = time.time() - start_time
    batch_size = batch['input_ids'].shape[0]
    seq_length = batch['input_ids'].shape[1]
    throughput = (batch_size * seq_length) / step_time
    tokens_per_second = throughput / step_time

    # Update metrics
    metrics = {
        'loss': loss,
        'grad_norm': grad_norm,
        'learning_rate': state.opt_state.hyperparams['learning_rate'],
        'throughput': throughput,
        'tokens_per_second': tokens_per_second,
        'step_time': step_time
    }

    # Log to wandb and tensorboard only on main process
    if jax.process_index() == 0:
        if state.config.logging.use_wandb:
            wandb.log(metrics, step=state.step)
        if state.writer is not None:
            for k, v in metrics.items():
                state.writer.add_scalar(f'train/{k}', v, state.step)
    
    return state, metrics

@jax.jit
def val_step(
    state: TrainingState,
    batch: Dict[str, jnp.ndarray]
) -> Dict[str, float]:
    """
    Paso de validación con caching de JIT y float32.
    
    Args:
        state: Estado de entrenamiento
        batch: Lote de datos de validación
        
    Returns:
        Métricas de validación
    """
    # Usar float32 para validación
    with jax.default_matmul_precision('float32'):
        logits = state.apply_fn(
            state.params,
            batch['input_ids'],
            batch['attention_mask'],
            training=False
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['labels']
        ).mean()
        
        # Calcular perplexity
        perplexity = jnp.exp(loss)
    
    return {
        'val_loss': loss,
        'val_perplexity': perplexity
    }

def setup_profiling(config: Config) -> None:
    """
    Configura el profiling de JAX.
    
    Args:
        config: Configuración del modelo
    """
    if config.training.profiling:
        jax.profiler.start_trace(config.training.profile_dir)
    
def stop_profiling(config: Config) -> None:
    """
    Detiene el profiling de JAX.
    
    Args:
        config: Configuración del modelo
    """
    if config.training.profiling:
        jax.profiler.stop_trace()

def save_checkpoint(state: TrainingState, step: int) -> None:
    """
    Guarda un checkpoint del modelo.
    
    Args:
        state: Estado del modelo
        step: Paso actual
    """
    if jax.process_index() == 0:
        checkpoint_path = Path(state.config.training.checkpoint_dir) / f'checkpoint_{step}'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            f.write(jax.serialization.to_bytes(state))

def load_checkpoint(config: Config, step: Optional[int] = None) -> Optional[TrainingState]:
    """
    Carga un checkpoint del modelo.
    
    Args:
        config: Configuración del modelo
        step: Paso del checkpoint a cargar (opcional)
        
    Returns:
        Estado del modelo o None si no se encuentra
    """
    checkpoint_dir = Path(config.training.checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    if step is None:
        # Cargar el último checkpoint
        checkpoints = list(checkpoint_dir.glob('checkpoint_*'))
        if not checkpoints:
            return None
        checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
    else:
        checkpoint_path = checkpoint_dir / f'checkpoint_{step}'
    
    if not checkpoint_path.exists():
        return None
    
    with open(checkpoint_path, 'rb') as f:
        return jax.serialization.from_bytes(TrainingState, f.read())

def setup_logging(config: Config) -> Tuple[Optional[SummaryWriter], Optional[wandb.run]]:
    """
    Configura el logging para el entrenamiento.
    
    Args:
        config: Configuración del modelo
        
    Returns:
        Tuple[Optional[SummaryWriter], Optional[wandb.run]]: Writer de TensorBoard y run de WandB
    """
    writer = None
    wandb_run = None
    
    if jax.process_index() == 0:
        if config.logging.use_wandb:
            wandb_run = wandb.init(
                project=config.logging.wandb_project,
                entity=config.logging.wandb_entity,
                config=config.__dict__
            )
        
        writer = SummaryWriter(logdir=config.logging.log_dir)
    
    return writer, wandb_run

def teardown_logging(writer: Optional[SummaryWriter], wandb_run: Optional[wandb.run]) -> None:
    """
    Cierra los recursos de logging.
    
    Args:
        writer: Writer de TensorBoard
        wandb_run: Run de WandB
    """
    if jax.process_index() == 0:
        if writer is not None:
            writer.close()
        if wandb_run is not None:
            wandb_run.finish() 