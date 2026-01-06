"""
TPU v6e Robust Trainer for CapibaraGPT-v2

Entrenador especializado para TPU v6e con topología 8x8 (64 chips) que maneja:
- Interrupciones de instancias preemptibles
- Checkpoints automáticos frecuentes
- Recuperación automática desde el último checkpoint
- Optimizaciones específicas para 8x8 mesh topology
- Paralelización distribuida a través de 8 hosts

Configuración target:
- Topología: 8x8 (64 chips TPU v6e)
- Hosts: 8
- VMs: 16
- Tipo: v6e-64 (ct6e-standard-4t)
"""

import os
import sys
import signal
import pickle
import asyncio
import logging
import jax
import jax.numpy as jnp
import optax
import wandb
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List, Callable
from dataclasses import dataclass, field
from flax.training import train_state, checkpoints
from flax import core, struct
import numpy as np
from contextlib import contextmanager

# Import optimized modules
from .unified_trainer import UnifiedTrainer, TrainingMetrics
from .tpu_optimizations import setup_tpu_environment, TPUOptimizer

logger = logging.getLogger(__name__)

@dataclass
class TPUv6eConfig:
    """Configuration específica para TPU v6e 8x8."""
    # Topología
    mesh_rows: int = 8
    mesh_cols: int = 8 
    total_chips: int = 64
    hosts: int = 8
    vms: int = 16
    
    # Hardware específico
    accelerator_type: str = "v6e-64"
    machine_type: str = "ct6e-standard-4t"
    
    # Optimizaciones
    use_bf16: bool = True
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 8
    
    # Paralelización
    data_parallel_size: int = 16  # Across VMs
    model_parallel_size: int = 4  # Within each VM
    pipeline_parallel_size: int = 1
    
    # Checkpointing para preemptibles
    checkpoint_every_steps: int = 100  # Muy frecuente para interrupciones
    keep_last_n_checkpoints: int = 3
    emergency_checkpoint_interval: int = 50  # Checkpoint de emergencia
    
    # Tolerancia a fallos
    max_retries: int = 3
    retry_delay_base: float = 30.0  # Segundos
    health_check_interval: int = 10  # Steps entre health checks

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
    Entrenador robusto para TPU v6e que maneja interrupciones preemptibles
    y garantiza continuidad del entrenamiento from checkpoints.
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
        
        # Estados internos
        self.training_state = None
        self.current_step = 0
        self.current_epoch = 0
        self.is_training = False
        self.interrupted = False
        self.recovery_mode = False
        
        # Manejo de interrupciones
        self._setup_signal_handlers()
        
        # Directorios de checkpoint
        self._setup_checkpoint_dirs()
        
        # Configurar TPU v6e
        self._setup_tpu_v6e()
        
        # Métricas
        self.metrics_history = []
        
    def _setup_signal_handlers(self):
        """Configures manejadores de señales para interrupciones preemptibles."""
        def signal_handler(signum, frame):
            logger.warning(f"🚨 SEÑAL {signum} RECIBIDA - Iniciando checkpoint de emergencia...")
            self.interrupted = True
            if self.is_training:
                asyncio.create_task(self._emergency_checkpoint())
        
        # Manejar SIGTERM (Google Cloud preemption signal)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        logger.info("✅ Signal handlers configurados for handling preemption")
    
    def _setup_checkpoint_dirs(self):
        """Configures directorios de checkpoint."""
        self.checkpoint_dir = self.base_output_dir / f"tpu_v6e_{self.model_scale}"
        self.emergency_dir = self.checkpoint_dir / "emergency"
        self.metadata_dir = self.checkpoint_dir / "metadata"
        
        # Crear directorios
        for dir_path in [self.checkpoint_dir, self.emergency_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"📁 Checkpoint dirs configurados: {self.checkpoint_dir}")
    
    def _setup_tpu_v6e(self):
        """Configures TPU v6e con topología 8x8."""
        logger.info("🚀 Configurando TPU v6e con topología 8x8...")
        
        # Configurar JAX para TPU v6e
        os.environ['XLA_FLAGS'] = '--xla_tpu_enable_async_collective_fusion=true'
        os.environ['LIBTPU_INIT_ARGS'] = '--xla_tpu_enable_async_collective_fusion_multiple_steps=true'
        
        # Inicializar JAX
        jax.config.update('jax_enable_x64', False)  # Usar float32/bfloat16
        jax.config.update('jax_default_matmul_precision', 'bfloat16' if self.config.use_bf16 else 'float32')
        
        # Configurar mesh topology 8x8
        devices = jax.devices()
        logger.info(f"🔍 Devices detectados: {len(devices)}")
        
        if len(devices) != self.config.total_chips:
            logger.warning(f"⚠️ Expected {self.config.total_chips} devices, got {len(devices)}")
        
        # Crear mesh 8x8
        devices_array = np.array(devices).reshape(self.config.mesh_rows, self.config.mesh_cols)
        self.mesh = jax.sharding.Mesh(devices_array, ('data', 'model'))
        
        logger.info(f"✅ TPU v6e Mesh configurado: {self.config.mesh_rows}x{self.config.mesh_cols}")
        logger.info(f"   📊 Data parallel: {self.config.data_parallel_size}")
        logger.info(f"   🔧 Model parallel: {self.config.model_parallel_size}")
        
    async def load_or_create_training_state(self, model, dataset):
        """Cargar estado de entrenamiento from checkpoint o crear nuevo."""
        logger.info("🔄 Buscando checkpoints existentes...")
        
        # Buscar último checkpoint válido
        latest_checkpoint = self._find_latest_checkpoint()
        
        if latest_checkpoint:
            logger.info(f"📂 Checkpoint encontrado: {latest_checkpoint}")
            try:
                return await self._load_checkpoint(latest_checkpoint, model)
            except Exception as e:
                logger.error(f"❌ Error cargando checkpoint: {e}")
                logger.info("🆕 Creando nuevo estado de entrenamiento...")
        
        # Crear nuevo estado
        return self._create_new_training_state(model)
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Encontrar el checkpoint más reciente."""
        checkpoint_pattern = self.checkpoint_dir / "checkpoint_step_*"
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*"))
        
        if not checkpoints:
            return None
            
        # Ordenar por step number
        def extract_step(path):
            try:
                return int(path.name.split('_')[-1])
            except ValueError:
                return 0
                
        latest = max(checkpoints, key=extract_step)
        
        # Verificar integridad
        if self._verify_checkpoint_integrity(latest):
            return latest
        else:
            logger.warning(f"⚠️ Checkpoint corrupto: {latest}")
            return None
    
    def _verify_checkpoint_integrity(self, checkpoint_path: Path) -> bool:
        """Verifiesr integridad del checkpoint."""
        try:
            # Verificar archivos necesarios
            required_files = ["checkpoint", "metadata.pkl"]
            for file in required_files:
                if not (checkpoint_path / file).exists():
                    return False
            
            # Verificar metadata
            with open(checkpoint_path / "metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
                
            # Validar metadata
            required_fields = ['step', 'epoch', 'model_scale', 'timestamp']
            for field in required_fields:
                if not hasattr(metadata, field):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error verificando checkpoint: {e}")
            return False
    
    async def _load_checkpoint(self, checkpoint_path: Path, model) -> train_state.TrainState:
        """Cargar checkpoint con manejo robusto de errores."""
        logger.info(f"📂 Cargando checkpoint: {checkpoint_path}")
        
        try:
            # Cargar metadata
            with open(checkpoint_path / "metadata.pkl", 'rb') as f:
                metadata: CheckpointMetadata = pickle.load(f)
            
            # Restaurar training state
            restored_state = checkpoints.restore_checkpoint(
                ckpt_dir=str(checkpoint_path),
                target=None,
                prefix="checkpoint"
            )
            
            # Actualizar estados internos
            self.current_step = metadata.step
            self.current_epoch = metadata.epoch
            self.recovery_mode = True
            
            logger.info(f"✅ Checkpoint cargado exitosamente")
            logger.info(f"   📊 Step: {self.current_step}")
            logger.info(f"   📅 Epoch: {self.current_epoch}")
            logger.info(f"   📈 Loss: {metadata.loss:.4f}")
            
            return restored_state
            
        except Exception as e:
            logger.error(f"❌ Error cargando checkpoint: {e}")
            raise
    
    def _create_new_training_state(self, model) -> train_state.TrainState:
        """Createsr nuevo estado de entrenamiento."""
        logger.info("🆕 Creando nuevo estado de entrenamiento...")
        
        # Configurar optimizador con warm-up para TPU v6e
        learning_rate = self._create_learning_rate_schedule()
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=0.1
        )
        
        # Crear estado inicial
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, 512), dtype=jnp.int32)  # Dummy sequence
        
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=model.init(rng, dummy_input),
            tx=optimizer
        )
        
        logger.info("✅ Nuevo estado de entrenamiento creado")
        return state
    
    def _create_learning_rate_schedule(self):
        """Createsr schedule de learning rate optimizado para TPU v6e."""
        warmup_steps = 2000
        max_lr = 3e-4
        
        # Cosine decay con warm-up
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=max_lr,
            warmup_steps=warmup_steps,
            decay_steps=100000,
            end_value=1e-5
        )
        
        return schedule
    
    async def train(self, model, train_dataset, val_dataset=None, max_steps: int = 100000):
        """Entrenamiento principal con manejo robusto de interrupciones."""
        logger.info(f"🚀 Iniciando entrenamiento TPU v6e para {max_steps} steps")
        
        self.is_training = True
        
        try:
            # Cargar o crear estado
            state = await self.load_or_create_training_state(model, train_dataset)
            
            # Configurar W&B si está habilitado
            if self.use_wandb:
                self._setup_wandb()
            
            # Loop principal de entrenamiento
            while self.current_step < max_steps and not self.interrupted:
                try:
                    # Training step
                    step_metrics = await self._training_step(state, train_dataset)
                    
                    # Actualizar métricas
                    self._update_metrics(step_metrics)
                    
                    # Log progreso
                    if self.current_step % 10 == 0:
                        self._log_progress(step_metrics)
                    
                    # Checkpoint automático
                    if self.current_step % self.config.checkpoint_every_steps == 0:
                        await self._save_checkpoint(state, step_metrics)
                    
                    # Checkpoint de emergencia (más frecuente)
                    if self.current_step % self.config.emergency_checkpoint_interval == 0:
                        await self._quick_checkpoint(state)
                    
                    # Health check
                    if self.current_step % self.config.health_check_interval == 0:
                        await self._health_check()
                    
                    # Validación periódica
                    if val_dataset and self.current_step % 1000 == 0:
                        val_metrics = await self._validation_step(state, val_dataset)
                        self._log_validation(val_metrics)
                    
                    self.current_step += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error en step {self.current_step}: {e}")
                    await self._handle_training_error(state, e)
            
            # Checkpoint final
            if not self.interrupted:
                await self._save_checkpoint(state, self._get_current_metrics(), is_final=True)
                logger.info("🎉 Entrenamiento completado exitosamente")
            else:
                logger.info("⏸️ Entrenamiento interrumpido - Checkpoint de emergencia guardado")
                
        except Exception as e:
            logger.error(f"💥 Error crítico en entrenamiento: {e}")
            if self.training_state:
                await self._emergency_checkpoint()
            raise
        finally:
            self.is_training = False
    
    async def _training_step(self, state, dataset) -> Dict[str, Any]:
        """Single training step optimizado para TPU v6e."""
        start_time = time.time()
        
        # Obtener batch
        batch = next(dataset)
        
        # Definir training step con sharding
        @jax.jit
        def train_step(state, batch):
            def loss_fn(params):
                logits = state.apply_fn(params, batch['input_ids'])
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits, batch['labels']
                ).mean()
                return loss
            
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            
            # Gradient clipping
            grads = optax.clip_by_global_norm(1.0)(grads)
            
            # Update state
            state = state.apply_gradients(grads=grads)
            
            return state, loss
        
        # Ejecutar step
        with self.mesh:
            state, loss = train_step(state, batch)
        
        # Actualizar training state para emergency checkpoints
        self.training_state = state
        
        step_time = time.time() - start_time
        
        # Métricas del step
        metrics = {
            'loss': float(loss),
            'step_time': step_time,
            'tokens_per_second': batch['input_ids'].size / step_time,
            'learning_rate': float(state.opt_state.hyperparams['learning_rate']),
            'step': self.current_step
        }
        
        return metrics
    
    async def _validation_step(self, state, val_dataset) -> Dict[str, Any]:
        """Validatestion step."""
        val_losses = []
        
        # Evaluar en algunos batches de validación
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
        """Guardar checkpoint completo."""
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
            
            # Guardar metadata
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
            
            logger.info(f"💾 Checkpoint guardado: {checkpoint_path}")
            
            # Cleanup checkpoints antiguos
            await self._cleanup_old_checkpoints()
            
        except Exception as e:
            logger.error(f"❌ Error guardando checkpoint: {e}")
            raise
    
    async def _quick_checkpoint(self, state):
        """Checkpoint rápido de emergencia."""
        emergency_path = self.emergency_dir / f"emergency_step_{self.current_step}.pkl"
        
        try:
            # Guardar solo lo esencial
            quick_state = {
                'step': self.current_step,
                'epoch': self.current_epoch,
                'params': state.params,
                'timestamp': time.time()
            }
            
            with open(emergency_path, 'wb') as f:
                pickle.dump(quick_state, f)
                
            logger.debug(f"⚡ Quick checkpoint: {emergency_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error en quick checkpoint: {e}")
    
    async def _emergency_checkpoint(self):
        """Checkpoint de emergencia en caso de interrupción."""
        if not self.training_state:
            return
            
        logger.warning("🚨 EMERGENCY CHECKPOINT - Guardando estado...")
        
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
            
            logger.warning(f"🚨 EMERGENCY CHECKPOINT GUARDADO: {emergency_path}")
            
        except Exception as e:
            logger.error(f"💥 FALLO EN EMERGENCY CHECKPOINT: {e}")
    
    async def _health_check(self):
        """Health check del system TPU."""
        try:
            # Verify that todos los devices estén disponibles
            devices = jax.devices()
            if len(devices) != self.config.total_chips:
                logger.warning(f"⚠️ Device count mismatch: {len(devices)}/{self.config.total_chips}")
            
            # Test simple de compute
            test_array = jnp.ones((1000, 1000))
            result = jnp.sum(test_array)
            
            if not jnp.isfinite(result):
                logger.error("❌ Health check failed - computation error")
                
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
    
    def _setup_wandb(self):
        """Configures W&B para TPU v6e."""
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
                
                logger.info("✅ W&B configurado para TPU v6e")
                
            except Exception as e:
                logger.warning(f"⚠️ Error configurando W&B: {e}")
    
    def _log_progress(self, metrics: Dict[str, Any]):
        """Log del progreso de entrenamiento."""
        logger.info(
            f"Step {self.current_step:>6} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"LR: {metrics['learning_rate']:.2e} | "
            f"TPS: {metrics['tokens_per_second']:.0f} | "
            f"Time: {metrics['step_time']:.2f}s"
        )
        
        if self.use_wandb and wandb.run:
            wandb.log(metrics, step=self.current_step)
    
    async def _cleanup_old_checkpoints(self):
        """Limpiar checkpoints antiguos."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*"))
        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            # Ordenar por step
            def extract_step(path):
                try:
                    return int(path.name.split('_')[-1])
                except ValueError:
                    return 0
            
            checkpoints.sort(key=extract_step, reverse=True)
            
            # Eliminar checkpoints antiguos
            for old_checkpoint in checkpoints[self.config.keep_last_n_checkpoints:]:
                try:
                    import shutil
                    shutil.rmtree(old_checkpoint)
                    logger.debug(f"🗑️ Checkpoint eliminado: {old_checkpoint}")
                except Exception as e:
                    logger.warning(f"⚠️ Error eliminando checkpoint: {e}")

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
    Función principal para entrenamiento robusto en TPU v6e.
    
    Args:
        model: Modelo a entrenar
        train_dataset: Dataset de entrenamiento
        val_dataset: Dataset de validación (opcional)
        model_scale: Escala del modelo (300M, 1B, 3B, 7B, 13B)
        max_steps: Máximo número de steps
        output_dir: Directorio base para checkpoints
        use_wandb: Usar Weights & Biases
    """
    
    logger.info("🦙 INICIANDO ENTRENAMIENTO ROBUSTO TPU v6e")
    logger.info("="*80)
    logger.info(f"🎯 Modelo: {model_scale}")
    logger.info(f"🚀 Hardware: TPU v6e 8x8 (64 chips)")
    logger.info(f"⚡ Preemptible: SÍ (con recovery automático)")
    logger.info(f"💾 Checkpoints: Cada 100 steps + emergencia cada 50")
    logger.info("="*80)
    
    # Crear trainer
    trainer = TPUv6eRobustTrainer(
        model_scale=model_scale,
        base_output_dir=output_dir,
        use_wandb=use_wandb
    )
    
    # Entrenar
    await trainer.train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_steps=max_steps
    )
    
    logger.info("🎉 ENTRENAMIENTO COMPLETADO!")