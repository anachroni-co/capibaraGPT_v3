"""
Convexity-Aware Learning Rate Controller for JAX/Flax Training.

This module implements an adaptive learning rate controller that maintains
local convexity of the loss curve through second-order difference analysis.
Designed for integration with TPU v6e-64 training infrastructure.

Key Features:
- EMA smoothing for noise reduction
- Local convexity analysis via second differences
- Multiplicative gain adjustment over base LR schedule
- Cooldown mechanism to prevent oscillations
- JAX/Flax optimized implementation
"""

from __future__ import annotations
import logging
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Tuple, Dict, Optional, Any, Union
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ConvexityControllerConfig:
    """Configurestion for convexity-aware learning rate control."""
    
    # Core parameters
    window_size: int = 5                    # Window size for second difference (>=3)
    ema_alpha: float = 0.2                  # EMA smoothing factor [0,1]
    tolerance: float = 1e-5                 # Convexity tolerance (>=0)
    
    # Adjustment factors
    down_multiplier: float = 0.90           # Factor when losing convexity
    up_multiplier: float = 1.01             # Factor when maintaining convexity
    
    # Safety bounds
    min_gain: float = 0.1                   # Minimum LR gain
    max_gain: float = 3.0                   # Maximum LR gain
    
    # Control parameters
    cooldown_steps: int = 50                # Steps between reductions
    warmup_enable_steps: int = 500          # Steps before enabling increases
    
    # Advanced options
    enable_logging: bool = True             # Enable detailed logging
    stability_threshold: float = 1e-8       # Stability threshold for metrics
    
    def __post_init__(self):
        """Validates configuration parameters."""
        if self.window_size < 3:
            raise ValueError("window_size must be >= 3")
        if not 0 <= self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in [0,1]")
        if self.tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        if self.min_gain <= 0 or self.max_gain <= self.min_gain:
            raise ValueError("Invalid gain bounds")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0")


@dataclass
class ConvexityController:
    """
    Convexity-aware learning rate controller for JAX/Flax training.
    
    Monitors loss curve convexity through EMA-smoothed second differences
    and adjusts learning rate multiplicatively over base schedule.
    
    Usage:
        controller = ConvexityController()
        for step, loss in training_loop:
            lr_gain, metrics = controller.update(loss)
            lr_actual = base_lr_schedule(step) * lr_gain
    """
    
    config: ConvexityControllerConfig = field(default_factory=ConvexityControllerConfig)
    
    # Internal state (not serialized by default)
    _loss_buffer: Deque[float] = field(default_factory=lambda: deque(maxlen=64), init=False)
    _ema_loss: Optional[float] = field(default=None, init=False)
    _current_gain: float = field(default=1.0, init=False)
    _last_downstep: int = field(default=-10**9, init=False)
    _step_count: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Initialize controller state."""
        if not hasattr(self, '_loss_buffer'):
            self._loss_buffer = deque(maxlen=64)
        if not hasattr(self, '_ema_loss'):
            self._ema_loss = None
        if not hasattr(self, '_current_gain'):
            self._current_gain = 1.0
        if not hasattr(self, '_last_downstep'):
            self._last_downstep = -10**9
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        if not hasattr(self, '_initialized'):
            self._initialized = False
        
        self._initialized = True
        
        if self.config.enable_logging:
            logger.info(f"🎯 ConvexityController initialized with config: {self.config}")
    
    def update(self, loss_value: Union[float, jnp.ndarray]) -> Tuple[float, Dict[str, float]]:
        """
        Update controller with new loss value and return LR gain and metrics.
        
        Args:
            loss_value: Current loss value (host-side scalar)
            
        Returns:
            Tuple of (lr_gain, metrics_dict)
        """
        if not self._initialized:
            self.__post_init__()
        
        # Convert to Python float for consistency
        if hasattr(loss_value, 'item'):
            loss_val = float(loss_value.item())
        else:
            loss_val = float(loss_value)
        
        self._step_count += 1
        
        # Update EMA loss (noise filtering)
        if self._ema_loss is None:
            self._ema_loss = loss_val
        else:
            alpha = self.config.ema_alpha
            self._ema_loss = (1.0 - alpha) * self._ema_loss + alpha * loss_val
        
        # Add to buffer
        self._loss_buffer.append(self._ema_loss)
        
        # Compute convexity metrics
        convex_ok, second_diff = self._compute_convexity()
        
        # Update learning rate gain
        self._update_gain(convex_ok)
        
        # Prepare metrics
        metrics = self._prepare_metrics(loss_val, second_diff, convex_ok)
        
        return float(self._current_gain), metrics
    
    def _compute_convexity(self) -> Tuple[bool, float]:
        """Compute convexity status and second difference."""
        if len(self._loss_buffer) < 3:
            return True, 0.0
        
        # Get last three EMA values
        L_minus1 = self._loss_buffer[-3]
        L_0 = self._loss_buffer[-2] 
        L_plus1 = self._loss_buffer[-1]
        
        # Compute discrete second difference
        second_diff = L_plus1 - 2.0 * L_0 + L_minus1
        
        # Check convexity condition: second_diff >= -tolerance
        convex_ok = second_diff >= -self.config.tolerance
        
        return convex_ok, float(second_diff)
    
    def _update_gain(self, convex_ok: bool) -> None:
        """Update learning rate gain based on convexity status."""
        if convex_ok:
            # Increase gain only after warmup period
            if self._step_count > self.config.warmup_enable_steps:
                self._current_gain = min(
                    self._current_gain * self.config.up_multiplier,
                    self.config.max_gain
                )
        else:
            # Decrease gain with cooldown mechanism
            steps_since_reduction = self._step_count - self._last_downstep
            if steps_since_reduction >= self.config.cooldown_steps:
                self._current_gain = max(
                    self._current_gain * self.config.down_multiplier,
                    self.config.min_gain
                )
                self._last_downstep = self._step_count
    
    def _prepare_metrics(self, raw_loss: float, second_diff: float, convex_ok: bool) -> Dict[str, float]:
        """Prepare metrics dictionary for logging."""
        return {
            "step": float(self._step_count),
            "raw_loss": float(raw_loss),
            "ema_loss": float(self._ema_loss or 0.0),
            "second_diff": float(second_diff),
            "convex_ok": 1.0 if convex_ok else 0.0,
            "lr_gain": float(self._current_gain),
            "buffer_size": float(len(self._loss_buffer)),
            "steps_since_reduction": float(self._step_count - self._last_downstep),
            "warmup_active": 1.0 if self._step_count <= self.config.warmup_enable_steps else 0.0,
        }
    
    @property
    def lr_gain(self) -> float:
        """Current learning rate gain factor."""
        return float(self._current_gain)
    
    @property
    def step_count(self) -> int:
        """Current step count."""
        return self._step_count
    
    @property
    def ema_loss(self) -> Optional[float]:
        """Current EMA loss value."""
        return self._ema_loss
    
    def get_state(self) -> Dict[str, Any]:
        """Get controller state for checkpointing."""
        return {
            'config': self.config,
            'loss_buffer': list(self._loss_buffer),
            'ema_loss': self._ema_loss,
            'current_gain': self._current_gain,
            'last_downstep': self._last_downstep,
            'step_count': self._step_count
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load controller state from checkpoint."""
        self.config = state.get('config', self.config)
        
        buffer_data = state.get('loss_buffer', [])
        self._loss_buffer = deque(buffer_data, maxlen=64)
        
        self._ema_loss = state.get('ema_loss', None)
        self._current_gain = state.get('current_gain', 1.0)
        self._last_downstep = state.get('last_downstep', -10**9)
        self._step_count = state.get('step_count', 0)
        self._initialized = True
        
        if self.config.enable_logging:
            logger.info(f"🔄 ConvexityController state loaded at step {self._step_count}")
    
    def reset(self) -> None:
        """Reset controller to initial state."""
        self._loss_buffer.clear()
        self._ema_loss = None
        self._current_gain = 1.0
        self._last_downstep = -10**9
        self._step_count = 0
        
        if self.config.enable_logging:
            logger.info("🔄 ConvexityController reset to initial state")


def create_convexity_aware_lr_schedule(
    base_schedule: Callable[[int], float],
    controller: ConvexityController
) -> Callable[[int, float], float]:
    """
    Create a convexity-aware learning rate schedule.
    
    Args:
        base_schedule: Base LR schedule function (step -> lr)
        controller: ConvexityController instance
        
    Returns:
        Function that takes (step, loss) and returns adjusted LR
    """
    def convexity_aware_schedule(step: int, loss: float) -> float:
        """Convexity-aware LR schedule."""
        base_lr = base_schedule(step)
        lr_gain, _ = controller.update(loss)
        return base_lr * lr_gain
    
    return convexity_aware_schedule


# Integration helpers for JAX/Flax

def apply_lr_to_updates(
    updates: Any, 
    lr_now: Union[float, jnp.ndarray]
) -> Any:
    """Apply dynamic learning rate to optimizer updates."""
    return jax.tree.map(lambda u: u * lr_now, updates)


@jax.jit
def prepare_lr_for_pmap(lr_host: float) -> jnp.ndarray:
    """Prepare learning rate for pmap distribution."""
    return jnp.array(lr_host, dtype=jnp.float32)


def integrate_convexity_control(
    train_step_fn: Callable,
    controller: ConvexityController,
    base_lr_schedule: Callable[[int], float]
) -> Callable:
    """
    Integrate convexity control into existing training step function.
    
    Args:
        train_step_fn: Original training step function
        controller: ConvexityController instance
        base_lr_schedule: Base learning rate schedule
        
    Returns:
        Enhanced training step function with convexity control
    """
    def enhanced_train_step(state, batch, step: int):
        # Get base learning rate
        base_lr = base_lr_schedule(step)
        
        # Apply current gain from controller
        lr_now = base_lr * controller.lr_gain
        
        # Execute training step with dynamic LR
        new_state, metrics = train_step_fn(state, batch, lr_now)
        
        # Update controller with observed loss
        if 'loss' in metrics:
            lr_gain, conv_metrics = controller.update(metrics['loss'])
            metrics.update(conv_metrics)
            metrics['base_lr'] = base_lr
            metrics['lr_actual'] = lr_now
        
        return new_state, metrics
    
    return enhanced_train_step


# Example usage and factory functions

def create_default_controller() -> ConvexityController:
    """Creates controller with recommended default settings."""
    config = ConvexityControllerConfig(
        window_size=5,
        ema_alpha=0.2,
        tolerance=1e-5,
        down_multiplier=0.90,
        up_multiplier=1.01,
        min_gain=0.1,
        max_gain=3.0,
        cooldown_steps=50,
        warmup_enable_steps=500
    )
    return ConvexityController(config=config)


def create_conservative_controller() -> ConvexityController:
    """Creates controller with conservative settings."""
    config = ConvexityControllerConfig(
        window_size=7,
        ema_alpha=0.15,
        tolerance=3e-5,
        down_multiplier=0.95,
        up_multiplier=1.005,
        min_gain=0.2,
        max_gain=2.0,
        cooldown_steps=75,
        warmup_enable_steps=1000
    )
    return ConvexityController(config=config)


def create_aggressive_controller() -> ConvexityController:
    """Creates controller with aggressive settings for fast adaptation."""
    config = ConvexityControllerConfig(
        window_size=3,
        ema_alpha=0.3,
        tolerance=1e-6,
        down_multiplier=0.85,
        up_multiplier=1.02,
        min_gain=0.05,
        max_gain=5.0,
        cooldown_steps=25,
        warmup_enable_steps=200
    )
    return ConvexityController(config=config)


# Module exports
__all__ = [
    'ConvexityController',
    'ConvexityControllerConfig', 
    'create_convexity_aware_lr_schedule',
    'apply_lr_to_updates',
    'prepare_lr_for_pmap',
    'integrate_convexity_control',
    'create_default_controller',
    'create_conservative_controller',
    'create_aggressive_controller'
]