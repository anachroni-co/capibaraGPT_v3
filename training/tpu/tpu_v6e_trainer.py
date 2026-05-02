"""
TPU v6e Robust Trainer for CapibaraGPT v3

Specialized trainer for TPU v6e with 8x8 topology (64 chips) that handles:
- Preemptible instance interruptions
- Frequent automatic checkpoints
- Automatic recovery from last checkpoint
- Specific optimizations for 8x8 mesh topology
- Distributed parallelization across 8 hosts

Target configuration:
- Topology: 8x8 (64 TPU v6e chips)
- Hosts: 8
- VMs: 16
- Type: v6e-64 (ct6e-standard-4t)
"""

import os
import sys
import signal
import pickle
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

import numpy as np

logger = logging.getLogger(__name__)

# JAX/Flax/Optax import guards
try:
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state, checkpoints
    from flax import core, struct
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    optax = None
    train_state = None
    checkpoints = None
    core = None
    struct = None
    logger.warning("JAX/Flax/Optax not available. TPU trainer will not function.")

# W&B import guard
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Import optimized modules (with guard)
try:
    from .unified_trainer import UnifiedTrainer, TrainingMetrics
    from .tpu_optimizations import setup_tpu_environment, TPUOptimizer
except ImportError:
    UnifiedTrainer = None
    TrainingMetrics = None
    setup_tpu_environment = None
    TPUOptimizer = None

# MoE auxiliary loss (optional — only used when use_moe_aux_loss=True)
try:
    from core.moe.dynamic_moe import _compute_load_balance_loss as _moe_load_balance_loss
    MOE_AUX_AVAILABLE = True
except ImportError:
    _moe_load_balance_loss = None  # type: ignore[assignment]
    MOE_AUX_AVAILABLE = False

@dataclass
class TPUv6eConfig:
    """Configuration specific for TPU v6e 8x8."""
    # Topology
    mesh_rows: int = 8
    mesh_cols: int = 8
    total_chips: int = 64
    hosts: int = 8
    vms: int = 16

    # Specific hardware
    accelerator_type: str = "v6e-64"
    machine_type: str = "ct6e-standard-4t"

    # Optimizations
    use_bf16: bool = True
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 8

    # Parallelization
    data_parallel_size: int = 16  # Across VMs
    model_parallel_size: int = 4  # Within each VM
    pipeline_parallel_size: int = 1

    # Checkpointing for preemptibles
    checkpoint_every_steps: int = 100  # Very frequent for interruptions
    keep_last_n_checkpoints: int = 3
    emergency_checkpoint_interval: int = 50  # Emergency checkpoint

    # Fault tolerance
    max_retries: int = 3
    retry_delay_base: float = 30.0  # Seconds
    health_check_interval: int = 10  # Steps between health checks

    # MoE auxiliary loss — set use_moe_aux_loss=True when the model exposes
    # expert_loads via Flax intermediates (self.sow('intermediates', 'expert_loads', ...))
    use_moe_aux_loss: bool = False
    moe_load_balance_weight: float = 0.01
    moe_num_experts: int = 32

@dataclass
class CheckpointMetadata:
    """Checkpoint metadata for robust recovery."""
    step: int
    epoch: int
    model_scale: str
    loss: float
    timestamp: float
    host_id: int
    mesh_coordinates: Tuple[int, int]
    training_metrics: Dict[str, Any]
    optimizer_state: bool = True
    model_state: bool = True

class TPUv6eRobustTrainer:
    """
    Robust trainer for TPU v6e that handles preemptible interruptions
    and ensures training continuity from checkpoints.
    """
    
    def __init__(
        self,
        model_scale: str = "7B",
        base_output_dir: str = "./checkpoints_tpu_v6e",
        use_wandb: bool = True,
        config: Optional[TPUv6eConfig] = None
    ):
        self.model_scale = model_scale
        self.base_output_dir = Path(base_output_dir)
        self.use_wandb = use_wandb
        self.config = config or TPUv6eConfig()
        
        # Internal states
        self.training_state = None
        self.current_step = 0
        self.current_epoch = 0
        self.is_training = False
        self.interrupted = False
        self.recovery_mode = False

        # Interruption handling
        self._setup_signal_handlers()

        # Checkpoint directories
        self._setup_checkpoint_dirs()

        # Configure TPU v6e
        self._setup_tpu_v6e()

        # Metrics
        self.metrics_history = []
        
    def _setup_signal_handlers(self):
        """Configure signal handlers for preemptible interruptions."""
        def signal_handler(signum, frame):
            logger.warning(f"SIGNAL {signum} RECEIVED - Starting emergency checkpoint...")
            self.interrupted = True
            if self.is_training:
                asyncio.create_task(self._emergency_checkpoint())

        # Handle SIGTERM (Google Cloud preemption signal)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        logger.info("Signal handlers configured for handling preemption")
    
    def _setup_checkpoint_dirs(self):
        """Configure checkpoint directories."""
        self.checkpoint_dir = self.base_output_dir / f"tpu_v6e_{self.model_scale}"
        self.emergency_dir = self.checkpoint_dir / "emergency"
        self.metadata_dir = self.checkpoint_dir / "metadata"

        # Create directories
        for dir_path in [self.checkpoint_dir, self.emergency_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Checkpoint dirs configured: {self.checkpoint_dir}")
    
    def _setup_tpu_v6e(self):
        """Configure TPU v6e with 8x8 topology."""
        logger.info("Configuring TPU v6e with 8x8 topology...")

        # Configure JAX for TPU v6e
        os.environ['XLA_FLAGS'] = '--xla_tpu_enable_async_collective_fusion=true'
        os.environ['LIBTPU_INIT_ARGS'] = '--xla_tpu_enable_async_collective_fusion_multiple_steps=true'

        # Initialize JAX
        jax.config.update('jax_enable_x64', False)  # Use float32/bfloat16
        jax.config.update('jax_default_matmul_precision', 'bfloat16' if self.config.use_bf16 else 'float32')

        # Configure mesh topology 8x8
        devices = jax.devices()
        logger.info(f"Devices detected: {len(devices)}")
        
        if len(devices) != self.config.total_chips:
            logger.warning(f"️ Expected {self.config.total_chips} devices, got {len(devices)}")
        
        # Create 8x8 mesh
        devices_array = np.array(devices).reshape(self.config.mesh_rows, self.config.mesh_cols)
        self.mesh = jax.sharding.Mesh(devices_array, ('data', 'model'))

        logger.info(f"TPU v6e Mesh configured: {self.config.mesh_rows}x{self.config.mesh_cols}")
        logger.info(f"   Data parallel: {self.config.data_parallel_size}")
        logger.info(f"   Model parallel: {self.config.model_parallel_size}")
        
    async def load_or_create_training_state(self, model, dataset):
        """Load training state from checkpoint or create new."""
        logger.info("Searching for existing checkpoints...")

        # Find latest valid checkpoint
        latest_checkpoint = self._find_latest_checkpoint()

        if latest_checkpoint:
            logger.info(f"Checkpoint found: {latest_checkpoint}")
            try:
                return await self._load_checkpoint(latest_checkpoint, model)
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.info("Creating new training state...")

        # Create new state
        return self._create_new_training_state(model)
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint."""
        checkpoint_pattern = self.checkpoint_dir / "checkpoint_step_*"
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*"))
        
        if not checkpoints:
            return None
            
        # Sort by step number
        def extract_step(path):
            try:
                return int(path.name.split('_')[-1])
            except ValueError:
                return 0
                
        latest = max(checkpoints, key=extract_step)
        
        # Verify integrity
        if self._verify_checkpoint_integrity(latest):
            return latest
        else:
            logger.warning(f"️ Corrupted checkpoint: {latest}")
            return None
    
    def _verify_checkpoint_integrity(self, checkpoint_path: Path) -> bool:
        """Verify checkpoint integrity."""
        try:
            # Verify required files
            required_files = ["checkpoint", "metadata.pkl"]
            for file in required_files:
                if not (checkpoint_path / file).exists():
                    return False
            
            # Verify metadata
            with open(checkpoint_path / "metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)  # nosec B301 — trusted checkpoint
                
            # Validate metadata
            required_fields = ['step', 'epoch', 'model_scale', 'timestamp']
            for field in required_fields:
                if not hasattr(metadata, field):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error verifying checkpoint: {e}")
            return False
    
    async def _load_checkpoint(self, checkpoint_path: Path, model) -> train_state.TrainState:
        """Load checkpoint with robust error handling."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        try:
            # Load metadata
            with open(checkpoint_path / "metadata.pkl", 'rb') as f:
                metadata: CheckpointMetadata = pickle.load(f)  # nosec B301 — trusted checkpoint

            # Restore training state
            restored_state = checkpoints.restore_checkpoint(
                ckpt_dir=str(checkpoint_path),
                target=None,
                prefix="checkpoint"
            )

            # Update internal states
            self.current_step = metadata.step
            self.current_epoch = metadata.epoch
            self.recovery_mode = True

            logger.info(f"Checkpoint loaded successfully")
            logger.info(f"   Step: {self.current_step}")
            logger.info(f"   Epoch: {self.current_epoch}")
            logger.info(f"   Loss: {metadata.loss:.4f}")

            return restored_state

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def _create_new_training_state(self, model) -> train_state.TrainState:
        """Create new training state."""
        logger.info("Creating new training state...")
        
        # Configure optimizer with warm-up for TPU v6e
        learning_rate = self._create_learning_rate_schedule()
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=0.1
        )
        
        # Create initial state
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, 512), dtype=jnp.int32)  # Dummy sequence
        
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=model.init(rng, dummy_input),
            tx=optimizer
        )
        
        logger.info("New training state created")
        return state

    def _create_learning_rate_schedule(self):
        """Create learning rate schedule optimized for TPU v6e."""
        warmup_steps = 2000
        max_lr = 3e-4
        
        # Cosine decay with warm-up
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=max_lr,
            warmup_steps=warmup_steps,
            decay_steps=100000,
            end_value=1e-5
        )
        
        return schedule
    
    async def train(self, model, train_dataset, val_dataset=None, max_steps: int = 100000):
        """Main training with robust interruption handling."""
        logger.info(f"Starting TPU v6e training for {max_steps} steps")
        
        self.is_training = True
        
        try:
            # Load or create state
            state = await self.load_or_create_training_state(model, train_dataset)
            
            # Configure W&B if enabled
            if self.use_wandb:
                self._setup_wandb()
            
            # Main training loop
            while self.current_step < max_steps and not self.interrupted:
                try:
                    # Training step
                    step_metrics = await self._training_step(state, train_dataset)
                    
                    # Update metrics
                    self._update_metrics(step_metrics)
                    
                    # Log progress
                    if self.current_step % 10 == 0:
                        self._log_progress(step_metrics)
                    
                    # Automatic checkpoint
                    if self.current_step % self.config.checkpoint_every_steps == 0:
                        await self._save_checkpoint(state, step_metrics)
                    
                    # Emergency checkpoint (more frequent)
                    if self.current_step % self.config.emergency_checkpoint_interval == 0:
                        await self._quick_checkpoint(state)
                    
                    # Health check
                    if self.current_step % self.config.health_check_interval == 0:
                        await self._health_check()
                    
                    # Periodic validation
                    if val_dataset and self.current_step % 1000 == 0:
                        val_metrics = await self._validation_step(state, val_dataset)
                        self._log_validation(val_metrics)
                    
                    self.current_step += 1
                    
                except Exception as e:
                    logger.error(f" Error in step {self.current_step}: {e}")
                    await self._handle_training_error(state, e)
            
            # Final checkpoint
            if not self.interrupted:
                await self._save_checkpoint(state, self._get_current_metrics(), is_final=True)
                logger.info("Training completed successfully")
            else:
                logger.info("Training interrupted - Emergency checkpoint saved")
                
        except Exception as e:
            logger.error(f"Critical training error: {e}")
            if self.training_state:
                await self._emergency_checkpoint()
            raise
        finally:
            self.is_training = False
    
    async def _training_step(self, state, dataset) -> Dict[str, Any]:
        """Single training step optimized for TPU v6e."""
        start_time = time.time()

        # Get batch
        batch = next(dataset)

        use_moe_aux = self.config.use_moe_aux_loss and MOE_AUX_AVAILABLE
        moe_weight = self.config.moe_load_balance_weight
        moe_num_experts = self.config.moe_num_experts

        if use_moe_aux:
            # Separate JIT path that calls apply() with mutable=['intermediates']
            # so Flax models using self.sow('intermediates', 'expert_loads', ...)
            # expose their expert routing loads for the auxiliary loss.
            @jax.jit
            def train_step(state, batch):
                def loss_fn(params):
                    model_out, mutables = state.apply_fn(
                        params, batch['input_ids'], mutable=['intermediates']
                    )
                    # TPUOptimizedSSM returns (logits, recurrent_state); handle both
                    logits = model_out[0] if isinstance(model_out, tuple) else model_out
                    main_loss = optax.softmax_cross_entropy_with_integer_labels(
                        logits, batch['labels']
                    ).mean()
                    # Add load-balance auxiliary loss when expert_loads are present
                    expert_loads = mutables.get('intermediates', {}).get('expert_loads')
                    if expert_loads is not None:
                        aux = _moe_load_balance_loss(expert_loads, moe_num_experts)
                        return main_loss + jnp.asarray(moe_weight, dtype=jnp.float32) * aux, main_loss
                    return main_loss, main_loss

                (loss, main_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                grads = optax.clip_by_global_norm(1.0)(grads)
                state = state.apply_gradients(grads=grads)
                return state, loss, main_loss
        else:
            @jax.jit
            def train_step(state, batch):
                def loss_fn(params):
                    result = state.apply_fn(params, batch['input_ids'])
                    # TPUOptimizedSSM returns (logits, recurrent_state); handle both
                    logits = result[0] if isinstance(result, tuple) else result
                    return optax.softmax_cross_entropy_with_integer_labels(
                        logits, batch['labels']
                    ).mean()

                loss, grads = jax.value_and_grad(loss_fn)(state.params)
                grads = optax.clip_by_global_norm(1.0)(grads)
                state = state.apply_gradients(grads=grads)
                return state, loss, loss  # main_loss == total loss when no aux

        # Execute step
        with self.mesh:
            state, loss, main_loss = train_step(state, batch)

        # Update training state for emergency checkpoints
        self.training_state = state

        step_time = time.time() - start_time

        # Step metrics
        metrics = {
            'loss': float(loss),
            'main_loss': float(main_loss),
            'moe_aux_loss': float(loss) - float(main_loss),
            'step_time': step_time,
            'tokens_per_second': batch['input_ids'].size / step_time,
            'learning_rate': float(state.opt_state.hyperparams['learning_rate']),
            'step': self.current_step
        }
        
        return metrics
    
    async def _validation_step(self, state, val_dataset) -> Dict[str, Any]:
        """Validation step."""
        val_losses = []

        # Evaluate on some validation batches
        for i in range(min(10, len(val_dataset))):
            batch = next(val_dataset)
            
            @jax.jit
            def eval_step(state, batch):
                logits = state.apply_fn(state.params, batch['input_ids'])
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits, batch['labels']
                ).mean()
                return loss
            
            with self.mesh:
                val_loss = eval_step(state, batch)
                val_losses.append(float(val_loss))
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        perplexity = jnp.exp(avg_val_loss)
        
        return {
            'val_loss': avg_val_loss,
            'val_perplexity': float(perplexity),
            'step': self.current_step
        }
    
    def _log_validation(self, val_metrics: Dict[str, Any]):
        """Log validation metrics."""
        logger.info(
            f"VALIDATION | Step {self.current_step:>6} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val PPL: {val_metrics['val_perplexity']:.2f}"
        )
        
        if self.use_wandb and wandb.run:
            wandb.log(val_metrics, step=self.current_step)
    
    def _update_metrics(self, step_metrics: Dict[str, Any]):
        """Update internal metrics tracking."""
        self.metrics_history.append(step_metrics)
        
        # Keep only last 1000 metrics in memory
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        if not self.metrics_history:
            return {'loss': 0.0, 'step': self.current_step}
        return self.metrics_history[-1]
    
    async def _handle_training_error(self, state, error: Exception):
        """Handle training errors with retry logic."""
        logger.error(f"Training error at step {self.current_step}: {error}")
        
        # Save emergency checkpoint
        await self._emergency_checkpoint()
        
        # Check if it's a recoverable error
        if isinstance(error, (RuntimeError, OSError)):
            logger.info("Attempting to recover from error...")
            await asyncio.sleep(self.config.retry_delay_base)
            return
        
        # For non-recoverable errors, reraise
        raise error
    
    async def _save_checkpoint(self, state, metrics: Dict[str, Any], is_final: bool = False):
        """Save complete checkpoint."""
        checkpoint_name = f"checkpoint_step_{self.current_step}"
        if is_final:
            checkpoint_name += "_final"
            
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        try:
            # Save model state
            checkpoints.save_checkpoint(
                ckpt_dir=str(checkpoint_path),
                target=state,
                step=self.current_step,
                prefix="checkpoint",
                keep=self.config.keep_last_n_checkpoints
            )
            
            # Save metadata
            metadata = CheckpointMetadata(
                step=self.current_step,
                epoch=self.current_epoch,
                model_scale=self.model_scale,
                loss=metrics.get('loss', 0.0),
                timestamp=time.time(),
                host_id=jax.process_index(),
                mesh_coordinates=(self.config.mesh_rows, self.config.mesh_cols),
                training_metrics=metrics
            )
            
            with open(checkpoint_path / "metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Cleanup old checkpoints
            await self._cleanup_old_checkpoints()

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise
    
    async def _quick_checkpoint(self, state):
        """Quick emergency checkpoint."""
        emergency_path = self.emergency_dir / f"emergency_step_{self.current_step}.pkl"

        try:
            # Save only essentials
            quick_state = {
                'step': self.current_step,
                'epoch': self.current_epoch,
                'params': state.params,
                'timestamp': time.time()
            }

            with open(emergency_path, 'wb') as f:
                pickle.dump(quick_state, f)

            logger.debug(f"Quick checkpoint: {emergency_path}")

        except Exception as e:
            logger.warning(f"Error in quick checkpoint: {e}")
    
    async def _emergency_checkpoint(self):
        """Emergency checkpoint in case of interruption."""
        if not self.training_state:
            return

        logger.warning("EMERGENCY CHECKPOINT - Saving state...")

        emergency_path = self.emergency_dir / f"EMERGENCY_step_{self.current_step}_{int(time.time())}.pkl"

        try:
            emergency_data = {
                'step': self.current_step,
                'epoch': self.current_epoch,
                'state': self.training_state,
                'interrupted_at': time.time(),
                'recovery_info': {
                    'mesh_config': (self.config.mesh_rows, self.config.mesh_cols),
                    'model_scale': self.model_scale
                }
            }

            with open(emergency_path, 'wb') as f:
                pickle.dump(emergency_data, f)

            logger.warning(f"EMERGENCY CHECKPOINT SAVED: {emergency_path}")

        except Exception as e:
            logger.error(f"EMERGENCY CHECKPOINT FAILED: {e}")
    
    async def _health_check(self):
        """Health check of the TPU system."""
        try:
            # Verify that all devices are available
            devices = jax.devices()
            if len(devices) != self.config.total_chips:
                logger.warning(f"️ Device count mismatch: {len(devices)}/{self.config.total_chips}")
            
            # Simple compute test
            test_array = jnp.ones((1000, 1000))
            result = jnp.sum(test_array)
            
            if not jnp.isfinite(result):
                logger.error(" Health check failed - computation error")
                
        except Exception as e:
            logger.error(f" Health check failed: {e}")
    
    def _setup_wandb(self):
        """Configure W&B for TPU v6e."""
        if self.use_wandb:
            try:
                config = {
                    'model_scale': self.model_scale,
                    'hardware': 'tpu_v6e',
                    'topology': f"{self.config.mesh_rows}x{self.config.mesh_cols}",
                    'total_chips': self.config.total_chips,
                    'preemptible': True,
                    'checkpoint_frequency': self.config.checkpoint_every_steps,
                    'recovery_mode': self.recovery_mode
                }
                
                wandb.init(
                    project=f"capibara-tpu-v6e-{self.model_scale.lower()}",
                    config=config,
                    tags=['tpu-v6e', 'preemptible', 'robust-training'],
                    resume='allow' if self.recovery_mode else None
                )
                
                logger.info("W&B configured for TPU v6e")

            except Exception as e:
                logger.warning(f"Error configuring W&B: {e}")
    
    def _log_progress(self, metrics: Dict[str, Any]):
        """Log training progress."""
        moe_suffix = ""
        if self.config.use_moe_aux_loss and metrics.get('moe_aux_loss', 0.0) != 0.0:
            moe_suffix = f" | MoE-aux: {metrics['moe_aux_loss']:.4f}"
        logger.info(
            f"Step {self.current_step:>6} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"LR: {metrics['learning_rate']:.2e} | "
            f"TPS: {metrics['tokens_per_second']:.0f} | "
            f"Time: {metrics['step_time']:.2f}s"
            + moe_suffix
        )
        
        if self.use_wandb and wandb.run:
            wandb.log(metrics, step=self.current_step)
    
    async def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*"))
        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            # Sort by step
            def extract_step(path):
                try:
                    return int(path.name.split('_')[-1])
                except ValueError:
                    return 0

            checkpoints.sort(key=extract_step, reverse=True)

            # Delete old checkpoints
            for old_checkpoint in checkpoints[self.config.keep_last_n_checkpoints:]:
                try:
                    import shutil
                    shutil.rmtree(old_checkpoint)
                    logger.debug(f"Checkpoint deleted: {old_checkpoint}")
                except Exception as e:
                    logger.warning(f"Error deleting checkpoint: {e}")

# Convenience function for robust training
async def train_robust_tpu_v6e(
    model,
    train_dataset,
    val_dataset=None,
    model_scale: str = "7B",
    max_steps: int = 100000,
    output_dir: str = "./checkpoints_tpu_v6e",
    use_wandb: bool = True
):
    """
    Main function for robust training on TPU v6e.

    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        model_scale: Model scale (300M, 1B, 3B, 7B, 13B)
        max_steps: Maximum number of steps
        output_dir: Base directory for checkpoints
        use_wandb: Use Weights & Biases
    """

    logger.info("STARTING ROBUST TPU v6e TRAINING")
    logger.info("="*80)
    logger.info(f"Model: {model_scale}")
    logger.info(f"Hardware: TPU v6e 8x8 (64 chips)")
    logger.info(f"Preemptible: YES (with automatic recovery)")
    logger.info(f"Checkpoints: Every 100 steps + emergency every 50")
    logger.info("="*80)

    # Create trainer
    trainer = TPUv6eRobustTrainer(
        model_scale=model_scale,
        base_output_dir=output_dir,
        use_wandb=use_wandb
    )

    # Train
    await trainer.train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_steps=max_steps
    )

    logger.info("TRAINING COMPLETED!")