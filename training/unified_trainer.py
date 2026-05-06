"""
Unified Trainer for CapibaraGPT v3 (consensus-distillation oriented).

==============================================================================
  STATUS: NOT THE CANONICAL TRAINER. DO NOT IMPORT FROM PRODUCTION CODE YET.
==============================================================================

This file was preserved from the v2 generation of CapibaraGPT and is kept
on disk as the seed for a future *consensus-distillation* training path
(many-teachers + critic + progressive distillation). It is NOT what you
want for a regular pretraining run of a 1B / 3B Mixtral-style model.

For canonical pretraining of `core.model_factory.CapibaraMoEModel`, use
`core/trainer/` (introduced in BACKLOG-009 - see BACKLOG.md). That module
loads `configs/1b.toml` / `configs/3b.toml`, sums the MoE load-balance
auxiliary loss, applies warmup-cosine + grad-clip + Orbax checkpointing,
and ships a `scripts/train.py` launcher.


Why this file is currently dead code
------------------------------------

1. THREE IMPORTS POINT TO MOVED MODULES (lines below the docstring):

       from .training_config       import ...     # NOT in training/
       from .consensus_strategies  import ...     # NOT in training/
       from .tpu_optimizations     import ...     # NOT in training/

   The symbols exist, but the modules were reorganised after this file
   was last touched. Reviving the file requires rewriting the imports to:

       from config.training_config import (
           ModelScale, TrainingConfigFactory, get_config_for_scale,
       )
       from training.consensus.unified_consensus import (
           DistillationManager,
           AdvancedVotingSystem,
           should_use_consensus_for_scale,
           create_consensus_system_for_scale,
       )
       from training.optimizations.tpu_optimizations import (
           setup_tpu_environment, TPUOptimizer, verify_tpu_setup,
       )

2. FORWARD-PASS SIGNATURE INCOMPATIBLE WITH `CapibaraMoEModel`.

   - This file calls `model.init(rng, dummy_input, training=True)` and
     `state.apply_fn({'params': params}, batch['inputs'], training=True)`.
   - `core.model_factory.CapibaraMoEModel.__call__` accepts
     `(input_ids, deterministic=True)`, NOT `training=`. It also requires
     `rngs={"router": <key>, "dropout": <key>}` because the Mixtral-style
     top-k router consumes a PRNG key per call when `router_jitter > 0`.

3. LOSS IGNORES THE MoE LOAD-BALANCE TERM.

   `CapibaraMoEModel.__call__` returns
       ModelOutput(logits=(B,T,V), aux_loss=scalar)
   This trainer's `loss_fn(outputs, targets)` does
       optax.softmax_cross_entropy_with_integer_labels(outputs, targets).mean()
   directly on `outputs`, which would fail (ModelOutput is not a tensor)
   AND, even if it were patched to use `outputs.logits`, it would still
   drop `aux_loss`. Without
       total_loss = ce_loss + cfg.load_balance_weight * outputs.aux_loss
   the experts collapse to a single dispatch path within the first epoch
   and the MoE FFN stops learning. This is the most important fix.

4. HARDCODED HYPER-PARAMETERS / SHAPES.

   - `optimizer = optax.adamw(learning_rate=3e-4)` ignores the
     `[training]` section of the TOML configs (warmup, cosine schedule,
     weight_decay, beta1/beta2, grad_clip).
   - `dummy_input = jnp.ones((1, 2048), dtype=jnp.int32)` is hardcoded;
     1b.toml and 3b.toml both use seq_len=4096.
   - `jax.grad(step_fn)` then re-calls `step_fn(state.params)` to get a
     loss for logging; should be `jax.value_and_grad(step_fn)` to avoid
     a second forward pass.

5. CHECKPOINTING IS A STUB.

   The "_initialize_training_state" method has a placeholder comment
   "Restoration logic would go here". There is no Orbax integration,
   no async checkpoint, no resume-from-step. The new trainer in
   `core/trainer/` covers this end-to-end.

6. WANDB IS HARD-WIRED.

   `wandb.init(...)` runs unconditionally when `use_wandb=True`. Smoke
   tests in CI cannot afford an external service call; the new trainer
   makes wandb fully optional behind an `--enable-wandb` flag.

7. CONSENSUS-DISTILLATION PATH IS COUPLED IN BY DEFAULT.

   `_consensus_train_step` builds teacher signals from the student's own
   logits, which is a placeholder, not a real teacher ensemble. The
   realistic plan is to load N independent teacher checkpoints and run
   them in parallel - that infrastructure does not exist here yet.


When this file becomes useful
-----------------------------

The reason we keep it on disk (instead of moving it to `_deprecated/`) is
that the consensus-distillation track is on the roadmap. Once the basic
pretraining trainer is producing 1B / 3B checkpoints, the natural next
step is:

  (a) Train M independent students with `core/trainer/` to get teachers.
  (b) Reuse the scaffold here -- `UnifiedTrainer.__init__`, the metrics
      dataclass, the consensus init in `_initialize_components`, and the
      structure of `_consensus_train_step` -- to wire those teachers
      into a distillation run, replacing the placeholder
      student-as-teacher block with real teacher forward passes.
  (c) Drop the wandb hard-wiring and align the optimizer with the
      `[training]` section of the TOML, exactly as `core/trainer/` does.

By that point, items 1-4 of the "Why this file is dead code" section
will need to be addressed regardless. The integration plan is therefore:

  Phase 1 (BACKLOG-009, separate file): build `core/trainer/`. Done first.
  Phase 2 (future): port `_consensus_train_step` and the metrics scaffold
                    into `core/trainer/distillation.py`, drop this file.

Until Phase 2, this module stays here as documentation-of-intent. Do
not import it; the relative imports below will raise ModuleNotFoundError.

History (preserved for grep): merges of trainer.py + train_unified.py +
train_TPU.py + parts of train_300M_scale.py from CapibaraGPT v2.
"""

import os
import time
import logging
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optax
import wandb
from flax.training import train_state

import jax
from jax import numpy as jnp

# ----------------------------------------------------------------------------
# INTEGRATION-TODO (consensus-distillation track) - see module docstring above.
#
# These three relative imports are broken by design: the modules they point to
# live elsewhere now. Importing this file will raise ModuleNotFoundError. When
# you wire this trainer back in, replace them with the absolute paths shown
# in item 1 of the "Why this file is currently dead code" section above.
# ----------------------------------------------------------------------------

# Import optimized modules
from .training_config import ModelScale, TrainingConfigFactory, get_config_for_scale
from .consensus_strategies import (
    DistillationManager,
    AdvancedVotingSystem,
    should_use_consensus_for_scale,
    create_consensus_system_for_scale,
)
from .tpu_optimizations import setup_tpu_environment, TPUOptimizer, verify_tpu_setup

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Unified training metrics."""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    perplexity: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    
    # Consensus metrics (if applicable)
    consensus_confidence: Optional[float] = None
    consensus_votes: Optional[int] = None
    distillation_loss: Optional[float] = None
    
    # Performance metrics
    tokens_per_second: float = 0.0
    batch_time: float = 0.0
    memory_usage: float = 0.0

class UnifiedTrainer:
    """
    Unified and intelligent trainer for CapibaraGPT v3.

    Features:
    - Auto-detection of model scale
    - Automatic consensus distilling for 3B+ models
    - Automatic TPU v4-32 optimizations
    - Distributed training
    - Real-time monitoring
    """
    
    def __init__(
        self,
        model_scale: Union[str, ModelScale],
        model,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "checkpoints",
        use_wandb: bool = True
    ):
        # Normalize model scale
        if isinstance(model_scale, str):
            self.model_scale_name = model_scale
            try:
                self.model_scale = ModelScale(model_scale)
            except ValueError:
                logger.warning(f"Unknown scale: {model_scale}, using 3B by default")
                self.model_scale = ModelScale.SMALL_3B
                self.model_scale_name = "3B"
        else:
            self.model_scale = model_scale
            self.model_scale_name = model_scale.value
        
        self.model = model
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-setup based on scale
        self._setup_auto_configuration()

        # Initialize components
        self.state = None
        self.tpu_optimizer = None
        self.consensus_system = None
        self.distillation_manager = None
        self.wandb_run = None
        
        # Metrics
        self.metrics_history = []
        self.current_metrics = TrainingMetrics()

        # configure logging
        self._setup_logging()

        # Initialize components automatically
        self._initialize_components(use_wandb)
    
    def _setup_auto_configuration(self):
        """Automatically configure based on model scale."""
        # Obtain automatic configuration
        auto_config = get_config_for_scale(self.model_scale)

        # Determine whether to use consensus distilling automatically
        self.use_consensus = auto_config["use_consensus_distilling"]
        
        if self.use_consensus:
            logger.info(f" CONSENSUS DISTILLING AUTOMATICALLY ACTIVATED for model {self.model_scale_name}")
            logger.info("    Peer voting enabled")
            logger.info("    Critic arbitration enabled")
            logger.info("    Progressive distillation enabled")
        else:
            logger.info(f" Standard training for model {self.model_scale_name}")
        
        # Auto-detect hardware
        self.hardware_info = verify_tpu_setup()
        if self.hardware_info.get('tpu_available', False):
            logger.info(" TPU detected - TPU optimizations activated")
            self.use_tpu = True
        else:
            logger.info(" GPU/CPU detected - Standard training")
            self.use_tpu = False
    
    def _setup_logging(self):
        """Configure detailed logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("="*80)
        logger.info(f" CAPIBARAGPT-V2 UNIFIED TRAINER INITIALIZED")
        logger.info(f" Model scale: {self.model_scale_name}")
        logger.info(f" Consensus distilling: {' ACTIVE' if self.use_consensus else ' INACTIVE'}")
        logger.info(f" Hardware: {'TPU' if self.use_tpu else 'GPU/CPU'}")
        logger.info("="*80)
    
    def _initialize_components(self, use_wandb: bool):
        """Initialize all necessary components."""
        # Initialize TPU optimizer if available
        if self.use_tpu:
            self.tpu_optimizer = setup_tpu_environment()
            logger.info(" TPU Optimizer initialized")

        # Initialize consensus system if necessary
        if self.use_consensus:
            self.consensus_system = create_consensus_system_for_scale(self.model_scale_name)
            if self.consensus_system:
                logger.info(f" Consensus System initialized - {self.consensus_system.n_teachers} teachers, {self.consensus_system.n_critics} critics")

            # Initialize distillation manager
            self.distillation_manager = DistillationManager(
                temperature=4.0,
                alpha=0.3 if self.model_scale_name in ["3B", "7B", "13B"] else 0.5
            )
            logger.info(" Distillation Manager initialized")

        # Initialize W&B if enabled
        if use_wandb:
            self._setup_wandb()
    
    def _setup_wandb(self):
        """configure Weights & Biases."""
        try:
            project_name = f"capibara-{self.model_scale_name.lower()}"
            tags = ["capibara-gpt-v2", f"scale-{self.model_scale_name}"]
            
            if self.use_consensus:
                tags.append("consensus-distilling")
            if self.use_tpu:
                tags.append("tpu-v4-32")
            
            self.wandb_run = wandb.init(
                project=project_name,
                tags=tags,
                config={
                    "model_scale": self.model_scale_name,
                    "use_consensus": self.use_consensus,
                    "use_tpu": self.use_tpu,
                    "trainer_version": "unified-v2.0"
                }
            )
            logger.info(" W&B initialized")
        except Exception as e:
            logger.warning(f"Error initializing W&B: {e}")
            self.wandb_run = None
    
    async def train(
        self,
        train_dataset,
        val_dataset=None,
        num_epochs: int = 10,
        eval_every: int = 1000,
        save_every: int = 5000,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main unified training.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            num_epochs: Number of epochs
            eval_every: Evaluate every N steps
            save_every: Save checkpoint every N steps
            resume_from: Checkpoint to resume from (optional)

        Returns:
            Training results
        """
        logger.info(" STARTING UNIFIED TRAINING")

        # Initialize training state
        await self._initialize_training_state(resume_from)
        
        # Configure training functions
        train_step_fn = self._create_train_step_function()
        eval_step_fn = self._create_eval_step_function() if val_dataset else None
        
        # Start metrics
        start_time = time.time()
        total_steps = 0
        
        try:
            for epoch in range(num_epochs):
                logger.info(f" Starting epoch {epoch + 1}/{num_epochs}")
                epoch_start = time.time()

                # Training per epoch
                async for batch in self._get_training_batches(train_dataset):
                    step_start = time.time()
                    
                    # Training step
                    if self.use_consensus and self.consensus_system:
                        # Training with consensus distilling
                        self.state, step_metrics = await self._consensus_train_step(
                            self.state, batch, train_step_fn
                        )
                    else:
                        # Standard training
                        self.state, step_metrics = train_step_fn(self.state, batch)
                    
                    # Update metrics
                    self._update_metrics(step_metrics, time.time() - step_start)
                    
                    total_steps += 1
                    self.current_metrics.step = total_steps
                    self.current_metrics.epoch = epoch
                    
                    # Periodic evaluation
                    if val_dataset and total_steps % eval_every == 0:
                        eval_metrics = await self._evaluate(val_dataset, eval_step_fn)
                        self._log_evaluation(eval_metrics)
                    
                    # Periodic checkpoint
                    if total_steps % save_every == 0:
                        await self._save_checkpoint(total_steps)
                    
                    # Logging
                    if total_steps % 100 == 0:
                        self._log_training_progress(total_steps)
                
                # End of epoch
                epoch_time = time.time() - epoch_start
                logger.info(f" Epoch {epoch + 1} completed in {epoch_time:.2f}s")

                # Evaluation at end of epoch
                if val_dataset:
                    eval_metrics = await self._evaluate(val_dataset, eval_step_fn)
                    self._log_evaluation(eval_metrics)
                
                # Checkpoint at end of epoch
                await self._save_checkpoint(f"epoch_{epoch + 1}")
        
        except Exception as e:
            logger.error(f" Error during training: {e}")
            raise
        
        finally:
            # Cleanup and final results
            total_time = time.time() - start_time
            final_results = self._generate_final_results(total_steps, total_time)
            
            if self.wandb_run:
                wandb.finish()
            
            logger.info(" TRAINING COMPLETED")
            return final_results
    
    async def _initialize_training_state(self, resume_from: Optional[str]):
        """Initialize training state."""
        # Create optimizer
        optimizer = optax.adamw(learning_rate=3e-4)
        
        # Initialize model parameters
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, 2048), dtype=jnp.int32)
        variables = self.model.init(rng, dummy_input, training=True)
        
        # Create training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=optimizer
        )
        
        # Apply TPU optimizations if available
        if self.tpu_optimizer:
            # Optimize parameters for TPU
            logger.info(" Applying TPU optimizations to parameters")

        # Restore checkpoint if specified
        if resume_from:
            logger.info(f" Restoring checkpoint from: {resume_from}")
            # Restoration logic would go here

        logger.info(" Training state initialized")
    
    def _create_train_step_function(self):
        """Create optimized training step function."""
        def loss_fn(outputs, targets):
            return optax.softmax_cross_entropy_with_integer_labels(outputs, targets).mean()
        
        if self.tpu_optimizer:
            # Use TPU-optimized function
            return self.tpu_optimizer.create_distributed_train_step(loss_fn)
        else:
            # Standard function
            @jax.jit
            def train_step(state, batch):
                def step_fn(params):
                    outputs = state.apply_fn(
                        {'params': params},
                        batch['inputs'],
                        training=True
                    )
                    loss = loss_fn(outputs, batch['targets'])
                    return loss
                
                grad_fn = jax.grad(step_fn)
                grads = grad_fn(state.params)
                new_state = state.apply_gradients(grads=grads)
                
                return new_state, {'loss': step_fn(state.params)}
            
            return train_step
    
    def _create_eval_step_function(self):
        """Create evaluation function."""
        def loss_fn(outputs, targets):
            return optax.softmax_cross_entropy_with_integer_labels(outputs, targets).mean()
        
        @jax.jit
        def eval_step(state, batch):
            outputs = state.apply_fn(
                {'params': state.params},
                batch['inputs'],
                training=False
            )
            loss = loss_fn(outputs, batch['targets'])
            return {'eval_loss': loss, 'eval_perplexity': jnp.exp(loss)}
        
        return eval_step
    
    async def _consensus_train_step(self, state, batch, train_step_fn):
        """Training step with consensus distilling."""
        if not self.consensus_system or not self.distillation_manager:
            return train_step_fn(state, batch)
        
        # Obtain student output
        student_output = state.apply_fn(
            {'params': state.params},
            batch['inputs'],
            training=True
        )

        # Use student output as teacher signals when no separate teachers exist
        teacher_outputs = [student_output for _ in range(self.consensus_system.n_teachers)]

        # Derive prompt and outputs from logits for consensus voting
        prompt = batch.get("text", "training_prompt")
        output_text = self._logits_to_text(student_output)
        outputs = [output_text for _ in range(len(teacher_outputs))]
        model_scores = self._compute_model_scores(student_output, len(teacher_outputs))
        latencies = [0.0 for _ in range(len(teacher_outputs))]
        
        # Perform consensus voting
        consensus_result = await self.consensus_system.enhanced_peer_vote(
            prompt=str(prompt),
            outputs=outputs,
            model_scores=model_scores,
            latencies=latencies
        )
        
        # Apply progressive distillation
        distill_loss, distill_metrics = await self.distillation_manager.progressive_distillation(
            teacher_outputs=teacher_outputs,
            student_output=student_output,
            targets=batch['targets']
        )
        
        # Update parameters using distillation loss
        grads = jax.grad(lambda params: distill_loss)(state.params)
        new_state = state.apply_gradients(grads=grads)
        
        # Combine metrics
        metrics = {
            'loss': float(distill_loss),
            'consensus_confidence': consensus_result[2].get('consensus_confidence', 0.0),
            'distillation_loss': distill_metrics['distill_loss'],
            'ce_loss': distill_metrics['ce_loss']
        }
        
        return new_state, metrics

    def _logits_to_text(self, logits: jnp.ndarray) -> str:
        """Convert logits to a minimal token string for consensus."""
        try:
            if logits.ndim >= 2:
                first = logits[0]
            else:
                first = logits
            token_id = int(jnp.argmax(first))
            return f"token_{token_id}"
        except Exception:
            return "token_0"

    def _compute_model_scores(self, logits: jnp.ndarray, count: int) -> List[float]:
        """Compute simple confidence scores from logits."""
        try:
            if logits.ndim >= 2:
                probs = jax.nn.softmax(logits, axis=-1)
                max_probs = jnp.max(probs, axis=-1)
                score = float(jnp.mean(max_probs))
            else:
                score = float(jnp.max(jax.nn.softmax(logits)))
        except Exception:
            score = 0.0
        return [score for _ in range(count)]
    
    async def _get_training_batches(self, dataset):
        """Training batch generator."""
        for batch in dataset:
            yield batch
    
    def _update_metrics(self, step_metrics: Dict[str, Any], batch_time: float):
        """Update training metrics."""
        self.current_metrics.loss = step_metrics.get('loss', 0.0)
        self.current_metrics.perplexity = jnp.exp(self.current_metrics.loss)
        self.current_metrics.batch_time = batch_time
        self.current_metrics.grad_norm = step_metrics.get('grad_norm', 0.0)
        
        # Consensus metrics if available
        if 'consensus_confidence' in step_metrics:
            self.current_metrics.consensus_confidence = step_metrics['consensus_confidence']
        if 'distillation_loss' in step_metrics:
            self.current_metrics.distillation_loss = step_metrics['distillation_loss']
        
        # Add to history
        self.metrics_history.append(dict(self.current_metrics.__dict__))

        # Keep only the last 1000 metrics in memory
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
    
    async def _evaluate(self, val_dataset, eval_step_fn) -> Dict[str, float]:
        """Perform evaluation on validation dataset."""
        total_loss = 0.0
        total_batches = 0
        
        for batch in val_dataset:
            eval_metrics = eval_step_fn(self.state, batch)
            total_loss += eval_metrics['eval_loss']
            total_batches += 1
            
            # Limit evaluation to avoid taking too long
            if total_batches >= 100:
                break
        
        avg_loss = total_loss / max(total_batches, 1)
        return {
            'eval_loss': float(avg_loss),
            'eval_perplexity': float(jnp.exp(avg_loss)),
            'eval_batches': total_batches
        }
    
    def _log_evaluation(self, eval_metrics: Dict[str, float]):
        """Logging de métricas de evaluación."""
        logger.info(f" EVALUACIÓN - Loss: {eval_metrics['eval_loss']:.4f}, "
                   f"Perplexity: {eval_metrics['eval_perplexity']:.2f}")
        
        if self.wandb_run:
            wandb.log({
                f"eval/{k}": v for k, v in eval_metrics.items()
            }, step=self.current_metrics.step)
    
    def _log_training_progress(self, step: int):
        """Logging de progreso de entrenamiento."""
        metrics = self.current_metrics
        
        log_msg = f"Step {step:6d} | Loss: {metrics.loss:.4f} | Perplexity: {metrics.perplexity:.2f}"
        
        if metrics.consensus_confidence is not None:
            log_msg += f" | Consensus: {metrics.consensus_confidence:.3f}"
        
        if metrics.batch_time > 0:
            log_msg += f" | Time: {metrics.batch_time:.3f}s"
        
        logger.info(log_msg)
        
        # Log a W&B
        if self.wandb_run:
            wandb_metrics = {
                'train/loss': metrics.loss,
                'train/perplexity': metrics.perplexity,
                'train/batch_time': metrics.batch_time
            }
            
            if metrics.consensus_confidence is not None:
                wandb_metrics['consensus/confidence'] = metrics.consensus_confidence
            if metrics.distillation_loss is not None:
                wandb_metrics['distillation/loss'] = metrics.distillation_loss
            
            wandb.log(wandb_metrics, step=step)
    
    async def _save_checkpoint(self, step_or_name: Union[int, str]):
        """save checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_{step_or_name}"
        
        # En implementation real, here se guardaría el estado complete
        logger.info(f" Guardando checkpoint: {checkpoint_path}")
        
        # save métricas also
        metrics_path = self.output_dir / f"metrics_{step_or_name}.json"
        # En implementation real, save métricas en JSON
    
    def _generate_final_results(self, total_steps: int, total_time: float) -> Dict[str, Any]:
        """generate resultados finales del entrenamiento."""
        results = {
            'model_scale': self.model_scale_name,
            'total_steps': total_steps,
            'total_time_seconds': total_time,
            'use_consensus_distilling': self.use_consensus,
            'use_tpu_optimizations': self.use_tpu,
            'final_loss': self.current_metrics.loss,
            'final_perplexity': self.current_metrics.perplexity
        }
        
        if self.use_consensus and self.consensus_system:
            consensus_metrics = self.consensus_system.get_comprehensive_metrics()
            results['consensus_metrics'] = consensus_metrics
        
        logger.info(" RESULTADOS FINALES:")
        logger.info(f"    Pasos totales: {total_steps}")
        logger.info(f"    Tiempo total: {total_time:.2f}s")
        logger.info(f"    Pérdida final: {self.current_metrics.loss:.4f}")
        logger.info(f"    Perplexity final: {self.current_metrics.perplexity:.2f}")
        
        return results

# Funciones de conveniencia for uso fast
def train_model(
    model_scale: str,
    model,
    train_dataset,
    val_dataset=None,
    output_dir: str = "checkpoints",
    num_epochs: int = 10,
    use_wandb: bool = True,
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Función principal para entrenar un modelo CapibaraGPT.
    
    Esta función automáticamente:
    - Detecta si usar consensus distilling (para modelos 3B+)
    - Aplica optimizaciones TPU si están disponibles
    - Configura logging y monitoreo
    
    Args:
        model_scale: Escala del modelo (ej: "3B", "7B", "30B")
        model: Modelo a entrenar
        train_dataset: Dataset de entrenamiento
        val_dataset: Dataset de validación (opcional)
        output_dir: Directorio de salida
        num_epochs: Número de épocas
        use_wandb: Usar Weights & Biases
        custom_config: Configuración personalizada
        
    Returns:
        Resultados del entrenamiento
    """
    trainer = UnifiedTrainer(
        model_scale=model_scale,
        model=model,
        config=custom_config,
        output_dir=output_dir,
        use_wandb=use_wandb
    )
    
    return asyncio.run(trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=num_epochs
    )) 
