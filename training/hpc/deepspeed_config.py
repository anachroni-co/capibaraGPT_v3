"""
DeepSpeed Configuration for HPC Training

Provides ZeRO Stage 1/2/3 configurations optimized for:
- Large model training (3B+ parameters)
- Multi-node distributed training
- Memory-efficient training on limited GPU memory
- Gradient checkpointing and mixed precision
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ZeROStage(Enum):
    """DeepSpeed ZeRO optimization stages."""
    DISABLED = 0    # No ZeRO
    STAGE_1 = 1     # Optimizer state partitioning
    STAGE_2 = 2     # Optimizer + gradient partitioning
    STAGE_3 = 3     # Full parameter partitioning


class OffloadDevice(Enum):
    """Offload target devices."""
    NONE = "none"
    CPU = "cpu"
    NVME = "nvme"


@dataclass
class ZeROConfig:
    """ZeRO optimization configuration."""

    stage: ZeROStage = ZeROStage.STAGE_2

    # Stage 3 specific
    stage3_max_live_parameters: int = 1_000_000_000
    stage3_max_reuse_distance: int = 1_000_000_000
    stage3_prefetch_bucket_size: int = 50_000_000
    stage3_param_persistence_threshold: int = 100_000
    stage3_gather_16bit_weights_on_model_save: bool = True

    # Offloading
    offload_optimizer: OffloadDevice = OffloadDevice.NONE
    offload_param: OffloadDevice = OffloadDevice.NONE
    offload_optimizer_pin_memory: bool = True
    offload_param_pin_memory: bool = True

    # NVMe offloading (for stage 3)
    nvme_path: str = "/local_nvme"

    # Communication
    reduce_bucket_size: int = 500_000_000
    allgather_bucket_size: int = 500_000_000
    overlap_comm: bool = True
    contiguous_gradients: bool = True

    # Memory
    reduce_scatter: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to DeepSpeed config format."""
        config = {
            "stage": self.stage.value,
            "reduce_bucket_size": self.reduce_bucket_size,
            "allgather_bucket_size": self.allgather_bucket_size,
            "overlap_comm": self.overlap_comm,
            "contiguous_gradients": self.contiguous_gradients,
            "reduce_scatter": self.reduce_scatter,
        }

        if self.stage == ZeROStage.STAGE_3:
            config.update({
                "stage3_max_live_parameters": self.stage3_max_live_parameters,
                "stage3_max_reuse_distance": self.stage3_max_reuse_distance,
                "stage3_prefetch_bucket_size": self.stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": self.stage3_param_persistence_threshold,
                "stage3_gather_16bit_weights_on_model_save": self.stage3_gather_16bit_weights_on_model_save,
            })

        if self.offload_optimizer != OffloadDevice.NONE:
            config["offload_optimizer"] = {
                "device": self.offload_optimizer.value,
                "pin_memory": self.offload_optimizer_pin_memory,
            }
            if self.offload_optimizer == OffloadDevice.NVME:
                config["offload_optimizer"]["nvme_path"] = self.nvme_path

        if self.offload_param != OffloadDevice.NONE:
            config["offload_param"] = {
                "device": self.offload_param.value,
                "pin_memory": self.offload_param_pin_memory,
            }
            if self.offload_param == OffloadDevice.NVME:
                config["offload_param"]["nvme_path"] = self.nvme_path

        return config


@dataclass
class FP16Config:
    """Mixed precision (FP16) configuration."""

    enabled: bool = True
    loss_scale: float = 0.0  # 0 = dynamic scaling
    loss_scale_window: int = 1000
    initial_scale_power: int = 16
    hysteresis: int = 2
    min_loss_scale: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "loss_scale": self.loss_scale,
            "loss_scale_window": self.loss_scale_window,
            "initial_scale_power": self.initial_scale_power,
            "hysteresis": self.hysteresis,
            "min_loss_scale": self.min_loss_scale,
        }


@dataclass
class BF16Config:
    """BF16 configuration (preferred for A100/H100)."""

    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"enabled": self.enabled}


@dataclass
class GradientClippingConfig:
    """Gradient clipping configuration."""

    enabled: bool = True
    max_norm: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return {"gradient_clipping": self.max_norm}


@dataclass
class ActivationCheckpointingConfig:
    """Activation/gradient checkpointing configuration."""

    enabled: bool = True
    partition_activations: bool = True
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = True
    number_checkpoints: Optional[int] = None
    synchronize_checkpoint_boundary: bool = False
    profile: bool = False

    def to_dict(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        config = {
            "partition_activations": self.partition_activations,
            "cpu_checkpointing": self.cpu_checkpointing,
            "contiguous_memory_optimization": self.contiguous_memory_optimization,
            "synchronize_checkpoint_boundary": self.synchronize_checkpoint_boundary,
            "profile": self.profile,
        }

        if self.number_checkpoints:
            config["number_checkpoints"] = self.number_checkpoints

        return {"activation_checkpointing": config}


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    type: str = "AdamW"

    # Adam parameters
    lr: float = 1e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    # Fused operations (faster on GPU)
    fused: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "params": {
                "lr": self.lr,
                "betas": list(self.betas),
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            },
        }


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    type: str = "WarmupDecayLR"
    warmup_min_lr: float = 0.0
    warmup_max_lr: float = 1e-4
    warmup_num_steps: int = 1000
    total_num_steps: int = 100000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "params": {
                "warmup_min_lr": self.warmup_min_lr,
                "warmup_max_lr": self.warmup_max_lr,
                "warmup_num_steps": self.warmup_num_steps,
                "total_num_steps": self.total_num_steps,
            },
        }


@dataclass
class CommunicationConfig:
    """Communication configuration for multi-node training."""

    # Gradient compression
    compression_enabled: bool = False
    compression_type: str = "fp16"  # fp16, powersgd

    # Tensor parallelism
    tensor_parallel_degree: int = 1

    # Pipeline parallelism
    pipeline_parallel_degree: int = 1

    def to_dict(self) -> Dict[str, Any]:
        config = {}

        if self.compression_enabled:
            config["communication_data_type"] = self.compression_type

        return config


@dataclass
class DeepSpeedConfig:
    """Complete DeepSpeed configuration."""

    # Training batch size
    train_batch_size: int = 256
    train_micro_batch_size_per_gpu: int = 8
    gradient_accumulation_steps: int = 8

    # Steps per print
    steps_per_print: int = 100

    # Wall clock breakdown
    wall_clock_breakdown: bool = False

    # Sub-configurations
    zero: ZeROConfig = field(default_factory=ZeROConfig)
    fp16: Optional[FP16Config] = None
    bf16: Optional[BF16Config] = field(default_factory=BF16Config)
    gradient_clipping: GradientClippingConfig = field(default_factory=GradientClippingConfig)
    activation_checkpointing: ActivationCheckpointingConfig = field(default_factory=ActivationCheckpointingConfig)
    optimizer: Optional[OptimizerConfig] = field(default_factory=OptimizerConfig)
    scheduler: Optional[SchedulerConfig] = None
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)

    # Flops profiler
    flops_profiler_enabled: bool = False
    flops_profiler_profile_step: int = 1
    flops_profiler_module_depth: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to DeepSpeed JSON config format."""
        config = {
            "train_batch_size": self.train_batch_size,
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "steps_per_print": self.steps_per_print,
            "wall_clock_breakdown": self.wall_clock_breakdown,
            "zero_optimization": self.zero.to_dict(),
        }

        # Mixed precision (bf16 preferred on modern GPUs)
        if self.bf16 and self.bf16.enabled:
            config["bf16"] = self.bf16.to_dict()
        elif self.fp16 and self.fp16.enabled:
            config["fp16"] = self.fp16.to_dict()

        # Gradient clipping
        config.update(self.gradient_clipping.to_dict())

        # Activation checkpointing
        config.update(self.activation_checkpointing.to_dict())

        # Optimizer
        if self.optimizer:
            config["optimizer"] = self.optimizer.to_dict()

        # Scheduler
        if self.scheduler:
            config["scheduler"] = self.scheduler.to_dict()

        # Communication
        config.update(self.communication.to_dict())

        # Flops profiler
        if self.flops_profiler_enabled:
            config["flops_profiler"] = {
                "enabled": True,
                "profile_step": self.flops_profiler_profile_step,
                "module_depth": self.flops_profiler_module_depth,
                "detailed": True,
            }

        return config

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        logger.info(f"DeepSpeed config saved to: {path}")


def create_deepspeed_config(
    model_size: str = "7B",
    num_gpus: int = 8,
    num_nodes: int = 1,
    gpu_memory_gb: int = 80,
    micro_batch_size: int = 4,
    gradient_accumulation: int = 8,
    learning_rate: float = 1e-4,
    use_cpu_offload: bool = False,
    use_nvme_offload: bool = False,
) -> DeepSpeedConfig:
    """
    Create optimized DeepSpeed configuration for a given model size.

    Args:
        model_size: Model size (e.g., "3B", "7B", "13B", "30B", "70B")
        num_gpus: GPUs per node
        num_nodes: Number of nodes
        gpu_memory_gb: GPU memory in GB
        micro_batch_size: Micro batch size per GPU
        gradient_accumulation: Gradient accumulation steps
        learning_rate: Base learning rate
        use_cpu_offload: Enable CPU offloading
        use_nvme_offload: Enable NVMe offloading

    Returns:
        Optimized DeepSpeedConfig
    """
    total_gpus = num_gpus * num_nodes
    train_batch_size = micro_batch_size * gradient_accumulation * total_gpus

    # Determine ZeRO stage based on model size and available memory
    model_params = {
        "3B": 3_000_000_000,
        "7B": 7_000_000_000,
        "13B": 13_000_000_000,
        "30B": 30_000_000_000,
        "70B": 70_000_000_000,
    }

    params = model_params.get(model_size, 7_000_000_000)
    params_per_gpu = params / total_gpus
    bytes_per_param = 2  # BF16
    memory_per_gpu_gb = (params_per_gpu * bytes_per_param * 4) / (1024**3)  # x4 for optimizer states

    if memory_per_gpu_gb > gpu_memory_gb * 0.8:
        if use_nvme_offload:
            zero_stage = ZeROStage.STAGE_3
            offload_opt = OffloadDevice.NVME
            offload_param = OffloadDevice.NVME
        elif use_cpu_offload:
            zero_stage = ZeROStage.STAGE_3
            offload_opt = OffloadDevice.CPU
            offload_param = OffloadDevice.CPU
        else:
            zero_stage = ZeROStage.STAGE_3
            offload_opt = OffloadDevice.NONE
            offload_param = OffloadDevice.NONE
    elif memory_per_gpu_gb > gpu_memory_gb * 0.5:
        zero_stage = ZeROStage.STAGE_2
        offload_opt = OffloadDevice.CPU if use_cpu_offload else OffloadDevice.NONE
        offload_param = OffloadDevice.NONE
    else:
        zero_stage = ZeROStage.STAGE_1
        offload_opt = OffloadDevice.NONE
        offload_param = OffloadDevice.NONE

    logger.info(f"Creating DeepSpeed config for {model_size}: ZeRO Stage {zero_stage.value}, "
                f"batch_size={train_batch_size}, offload_opt={offload_opt.value}")

    return DeepSpeedConfig(
        train_batch_size=train_batch_size,
        train_micro_batch_size_per_gpu=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        zero=ZeROConfig(
            stage=zero_stage,
            offload_optimizer=offload_opt,
            offload_param=offload_param,
            overlap_comm=True,
            contiguous_gradients=True,
            reduce_scatter=True,
        ),
        bf16=BF16Config(enabled=True),
        gradient_clipping=GradientClippingConfig(enabled=True, max_norm=1.0),
        activation_checkpointing=ActivationCheckpointingConfig(
            enabled=True,
            partition_activations=True,
            cpu_checkpointing=use_cpu_offload,
        ),
        optimizer=OptimizerConfig(
            type="AdamW",
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        ),
        scheduler=SchedulerConfig(
            type="WarmupDecayLR",
            warmup_max_lr=learning_rate,
            warmup_num_steps=2000,
            total_num_steps=100000,
        ),
    )
