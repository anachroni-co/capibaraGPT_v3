"""
Convexity-Aware Training Strategy for JAX/Flax TPU Training.

This module provides enhanced training strategies that integrate the ConvexityController
with existing TPU v6e training infrastructure, offering adaptive learning rate control
based on loss curve convexity analysis.

Key Features:
- Integration with TPU v6e-64 training pipeline
- JAX/Flax optimized implementations
- Support for pmap distributed training
- Real-time convexity monitoring and metrics
- Checkpointing support for controller state
"""

from __future__ import annotations
import logging
import time
import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from flax.training import train_state
from flax import struct

from .convexity_controller import (
    ConvexityController, 
    ConvexityControllerConfig,
    create_default_controller,
    apply_lr_to_updates,
    prepare_lr_for_pmap
)

logger = logging.getLogger(__name__)


class TrainStateWithConvexity(train_state.TrainState):
    """Extended TrainState with convexity-aware learning rate."""
    lr_now: float = struct.field(pytree_node=False, default=0.0)
    lr_gain: float = struct.field(pytree_node=False, default=1.0)
    base_lr: float = struct.field(pytree_node=False, default=0.0)


@dataclass
class ConvexityTrainingConfig:
    """Configurestion for convexity-aware training."""
    
    # Base training parameters
    total_steps: int = 10000
    base_lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    
    # Optimizer configuration
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    
    # Convexity controller configuration
    convexity_config: Optional[ConvexityControllerConfig] = None
    
    # Logging and monitoring
    log_every: int = 50
    checkpoint_every: int = 1000
    enable_wandb: bool = False
    
    # Advanced options
    min_lr_factor: float = 0.1  # Minimum LR as fraction of base_lr
    
    def __post_init__(self):
        """Initialize default convexity config if not provided."""
        if self.convexity_config is None:
            self.convexity_config = ConvexityControllerConfig()


class ConvexityTrainingStrategy:
    """
    Enhanced training strategy with convexity-aware learning rate control.
    
    Integrates ConvexityController into JAX/Flax training loops with support
    for TPU v6e-64 distributed training and comprehensive monitoring.
    """
    
    def __init__(self, config: ConvexityTrainingConfig):
        self.config = config
        self.controller = ConvexityController(config=config.convexity_config)
        self.step_count = 0
        self.start_time = time.time()
        
        logger.info(f"🎯 ConvexityTrainingStrategy initialized")
        logger.info(f"📊 Config: {self.config}")
    
    def create_optimizer(self) -> optax.GradientTransformation:
        """Creates optimizer without fixed learning rate scaling."""
        return optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip_norm),
            optax.adamw(
                b1=self.config.adam_b1,
                b2=self.config.adam_b2,
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay
            )
            # Note: No optax.scale(-lr) here - we handle LR dynamically
        )
    
    def create_base_lr_schedule(self) -> Callable[[int], float]:
        """Creates base learning rate schedule (warmup + cosine decay)."""
        def schedule_fn(step: int) -> float:
            step = jnp.minimum(step, self.config.total_steps)
            
            # Warmup phase
            warmup_factor = jnp.minimum(step / jnp.maximum(1, self.config.warmup_steps), 1.0)
            
            # Cosine decay phase
            decay_steps = self.config.total_steps - self.config.warmup_steps
            progress = (step - self.config.warmup_steps) / jnp.maximum(1, decay_steps)
            progress = jnp.clip(progress, 0.0, 1.0)
            
            # Cosine decay to min_lr_factor * base_lr
            cosine_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
            decay_factor = self.config.min_lr_factor + (1.0 - self.config.min_lr_factor) * cosine_factor
            
            return self.config.base_lr * warmup_factor * decay_factor
        
        return schedule_fn
    
    def create_train_step(self, apply_fn: Callable) -> Callable:
        """Creates convexity-aware training step function."""
        
        def train_step_fn(
            state: TrainStateWithConvexity, 
            batch: Dict[str, jnp.ndarray], 
            lr_now: jnp.ndarray
        ) -> Tuple[TrainStateWithConvexity, Dict[str, jnp.ndarray]]:
            """Single training step with dynamic learning rate."""
            
            def loss_fn(params):
                """Compute loss for the batch."""
                logits = apply_fn({'params': params}, batch['x'])
                loss = jnp.mean(
                    optax.softmax_cross_entropy_with_integer_labels(logits, batch['y'])
                )
                return loss
            
            # Compute loss and gradients
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            
            # Average across devices (for pmap)
            loss = jax.lax.pmean(loss, axis_name='data')
            grads = jax.lax.pmean(grads, axis_name='data')
            
            # Get optimizer updates
            updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
            
            # Apply dynamic learning rate
            updates = apply_lr_to_updates(updates, lr_now)
            
            # Update parameters
            new_params = optax.apply_updates(state.params, updates)
            
            # Create new state
            new_state = state.replace(
                params=new_params,
                opt_state=new_opt_state,
                step=state.step + 1,
                lr_now=lr_now
            )
            
            # Prepare metrics
            metrics = {
                "loss": loss,
                "lr_now": lr_now,
                "grad_norm": optax.global_norm(grads),
                "param_norm": optax.global_norm(new_params)
            }
            
            return new_state, metrics
        
        return train_step_fn
    
    def create_pmap_train_step(self, apply_fn: Callable) -> Callable:
        """Creates pmap-distributed training step."""
        train_step = self.create_train_step(apply_fn)
        return jax.pmap(train_step, axis_name='data', donate_argnums=(0,))
    
    def train_loop(
        self,
        state: TrainStateWithConvexity,
        apply_fn: Callable,
        dataset_iterator,
        log_fn: Optional[Callable] = None
    ) -> TrainStateWithConvexity:
        """
        Main training loop with convexity-aware learning rate control.
        
        Args:
            state: Initial training state
            apply_fn: Model apply function
            dataset_iterator: Iterator over training batches
            log_fn: Optional logging function
            
        Returns:
            Final training state
        """
        
        # Create training functions
        base_schedule = self.create_base_lr_schedule()
        p_train_step = self.create_pmap_train_step(apply_fn)
        
        logger.info(f"🚀 Starting convexity-aware training for {self.config.total_steps} steps")
        
        try:
            for step in range(1, self.config.total_steps + 1):
                self.step_count = step
                
                # Get current batch
                try:
                    batch = next(dataset_iterator)
                except StopIteration:
                    logger.warning("Dataset iterator exhausted")
                    break
                
                # Compute base learning rate
                base_lr_host = float(base_schedule(step))
                
                # Get current convexity gain
                lr_gain = self.controller.lr_gain
                lr_actual_host = base_lr_host * lr_gain
                
                # Prepare LR for devices
                lr_now_devices = jax.device_put_replicated(
                    prepare_lr_for_pmap(lr_actual_host),
                    jax.local_devices()
                )
                
                # Execute training step
                state, metrics = p_train_step(state, batch, lr_now_devices)
                
                # Extract host-side loss for controller
                loss_host = float(metrics["loss"][0])  # Take from first device
                
                # Update convexity controller
                lr_gain, conv_metrics = self.controller.update(loss_host)
                
                # Update state with new LR info
                state = state.replace(
                    lr_now=lr_actual_host,
                    lr_gain=lr_gain,
                    base_lr=base_lr_host
                )
                
                # Combine metrics
                combined_metrics = {
                    "step": step,
                    "loss": loss_host,
                    "base_lr": base_lr_host,
                    "lr_gain": lr_gain,
                    "lr_actual": lr_actual_host,
                    "grad_norm": float(metrics["grad_norm"][0]),
                    "param_norm": float(metrics["param_norm"][0]),
                    **conv_metrics
                }
                
                # Logging
                if step % self.config.log_every == 0 or step == 1:
                    self._log_metrics(step, combined_metrics)
                    if log_fn:
                        log_fn(combined_metrics)
                
                # Checkpointing
                if step % self.config.checkpoint_every == 0:
                    self._checkpoint_controller(step)
                
                if step >= self.config.total_steps:
                    break
                    
        except KeyboardInterrupt:
            logger.info("🛑 Training interrupted by user")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        
        elapsed = time.time() - self.start_time
        logger.info(f"✅ Training completed in {elapsed:.2f}s ({self.step_count} steps)")
        
        return state
    
    def _log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log training metrics."""
        elapsed = time.time() - self.start_time
        steps_per_sec = step / elapsed if elapsed > 0 else 0
        
        log_msg = (
            f"Step {step:>6d} | "
            f"Loss: {metrics['loss']:.6f} | "
            f"BaseLR: {metrics['base_lr']:.2e} | "
            f"Gain: {metrics['lr_gain']:.4f} | "
            f"ActualLR: {metrics['lr_actual']:.2e} | "
            f"Convex: {'✓' if metrics.get('convex_ok', 0) > 0.5 else '✗'} | "
            f"SecDiff: {metrics.get('second_diff', 0):.2e} | "
            f"{steps_per_sec:.1f} steps/s"
        )
        
        logger.info(log_msg)
        
        # WandB logging if enabled
        if self.config.enable_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except ImportError:
                pass
    
    def _checkpoint_controller(self, step: int) -> None:
        """Save controller state for recovery."""
        try:
            controller_state = self.controller.get_state()
            # You can extend this to save to disk if needed
            logger.debug(f"💾 Controller state checkpointed at step {step}")
        except Exception as e:
            logger.warning(f"Failed to checkpoint controller: {e}")
    
    def get_controller_state(self) -> Dict[str, Any]:
        """Get current controller state."""
        return self.controller.get_state()
    
    def load_controller_state(self, state: Dict[str, Any]) -> None:
        """Load controller state from checkpoint."""
        self.controller.load_state(state)
    
    def reset_controller(self) -> None:
        """Reset controller to initial state."""
        self.controller.reset()


def create_convexity_train_state(
    params: Any,
    tx: optax.GradientTransformation,
    apply_fn: Callable
) -> TrainStateWithConvexity:
    """Creates training state with convexity support."""
    return TrainStateWithConvexity.create(
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        lr_now=0.0,
        lr_gain=1.0,
        base_lr=0.0
    )


def create_cosine_schedule_with_warmup(
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 1000,
    min_lr_factor: float = 0.1
) -> Callable[[int], float]:
    """Creates cosine decay schedule with warmup."""
    def schedule_fn(step: int) -> float:
        step = jnp.minimum(step, total_steps)
        
        # Warmup
        warmup_factor = jnp.minimum(step / jnp.maximum(1, warmup_steps), 1.0)
        
        # Cosine decay
        decay_steps = total_steps - warmup_steps
        progress = (step - warmup_steps) / jnp.maximum(1, decay_steps)
        progress = jnp.clip(progress, 0.0, 1.0)
        
        cosine_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        decay_factor = min_lr_factor + (1.0 - min_lr_factor) * cosine_factor
        
        return base_lr * warmup_factor * decay_factor
    
    return schedule_fn


# Factory functions for different training scenarios

def create_tpu_v6e_convexity_strategy(
    total_steps: int = 50000,
    base_lr: float = 3e-4,
    model_size: str = "300M"
) -> ConvexityTrainingStrategy:
    """Creates strategy optimized for TPU v6e-64 training."""
    
    # Scale learning rate based on actual model scaling: 300M → 15B → 30B
    lr_scale = {
        "300M": 1.0,   # Base model: 3e-4
        "15B": 0.5,    # Large model: 1.5e-4 
        "30B": 0.3     # Massive model: 9e-5
    }.get(model_size, 1.0)
    
    # Adjust convexity parameters based on model size
    convexity_params = {
        "300M": {
            "tolerance": 1e-5,
            "cooldown_steps": 50,
            "up_multiplier": 1.01,
            "down_multiplier": 0.90,
            "ema_alpha": 0.2
        },
        "15B": {
            "tolerance": 3e-5,       # More conservative for large model
            "cooldown_steps": 75,     # Longer cooldown
            "up_multiplier": 1.005,   # Slower increases
            "down_multiplier": 0.95,  # Gentler decreases
            "ema_alpha": 0.15         # More smoothing
        },
        "30B": {
            "tolerance": 5e-5,       # Most conservative
            "cooldown_steps": 100,    # Longest cooldown
            "up_multiplier": 1.002,   # Minimal increases
            "down_multiplier": 0.97,  # Very gentle decreases
            "ema_alpha": 0.1          # Maximum smoothing
        }
    }
    
    params = convexity_params.get(model_size, convexity_params["300M"])
    
    config = ConvexityTrainingConfig(
        total_steps=total_steps,
        base_lr=base_lr * lr_scale,
        warmup_steps=max(1000, total_steps // 50),
        weight_decay=0.1,
        convexity_config=ConvexityControllerConfig(
            window_size=5,
            ema_alpha=params["ema_alpha"],
            tolerance=params["tolerance"],
            down_multiplier=params["down_multiplier"],
            up_multiplier=params["up_multiplier"],
            cooldown_steps=params["cooldown_steps"],
            warmup_enable_steps=max(500, total_steps // 100)
        ),
        log_every=50,
        checkpoint_every=1000
    )
    
    return ConvexityTrainingStrategy(config)


def create_experimental_convexity_strategy(
    total_steps: int = 10000,
    base_lr: float = 5e-4
) -> ConvexityTrainingStrategy:
    """Creates strategy for experimental/research training."""
    
    config = ConvexityTrainingConfig(
        total_steps=total_steps,
        base_lr=base_lr,
        warmup_steps=500,
        convexity_config=ConvexityControllerConfig(
            window_size=3,
            ema_alpha=0.3,
            tolerance=5e-6,
            down_multiplier=0.85,
            up_multiplier=1.02,
            cooldown_steps=25,
            warmup_enable_steps=200,
            max_gain=5.0
        ),
        log_every=10,
        checkpoint_every=500
    )
    
    return ConvexityTrainingStrategy(config)


# Module exports
__all__ = [
    'ConvexityTrainingStrategy',
    'ConvexityTrainingConfig',
    'TrainStateWithConvexity',
    'create_convexity_train_state',
    'create_cosine_schedule_with_warmup',
    'create_tpu_v6e_convexity_strategy',
    'create_experimental_convexity_strategy'
]