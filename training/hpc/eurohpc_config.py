"""
EuroHPC Configuration and Trainer Factory

Unified configuration for training on EuroHPC systems including:
- LUMI (Finland) - AMD MI250X
- Leonardo (Italy) - NVIDIA A100
- MareNostrum 5 (Spain) - NVIDIA H100/Intel GPUs
- JUWELS Booster (Germany) - NVIDIA A100
- Karolina (Czech Republic) - NVIDIA A100
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Literal
from pathlib import Path
from enum import Enum

from .distributed_init import DistributedConfig, init_distributed, get_rank, get_world_size
from .slurm_launcher import SlurmConfig, SlurmLauncher, detect_slurm_environment
from .deepspeed_config import DeepSpeedConfig, create_deepspeed_config, ZeROStage
from .fsdp_wrapper import FSDPConfig, FSDPWrapper, create_fsdp_config
from .checkpoint_manager import DistributedCheckpointManager, CheckpointConfig

logger = logging.getLogger(__name__)


class EuroHPCSystem(Enum):
    """EuroHPC supercomputer systems."""
    LUMI = "lumi"                   # AMD MI250X, InfiniBand
    LEONARDO = "leonardo"           # NVIDIA A100 64GB, InfiniBand
    MARENOSTRUM5 = "marenostrum5"   # NVIDIA H100, InfiniBand
    JUWELS = "juwels"               # NVIDIA A100 80GB, InfiniBand
    KAROLINA = "karolina"           # NVIDIA A100 40GB, InfiniBand
    MELUXINA = "meluxina"           # NVIDIA A100 40GB
    DISCOVERER = "discoverer"       # NVIDIA A100 80GB
    DEUCALION = "deucalion"         # NVIDIA A100 80GB
    GENERIC = "generic"             # Generic HPC system


class DistributedBackend(Enum):
    """Distributed training backend."""
    DEEPSPEED = "deepspeed"
    FSDP = "fsdp"
    DDP = "ddp"


@dataclass
class SystemSpecs:
    """Hardware specifications for an HPC system."""

    gpu_type: str
    gpu_memory_gb: int
    gpus_per_node: int
    interconnect: str
    storage_type: str
    modules: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)


# System specifications for each EuroHPC system
SYSTEM_SPECS: Dict[EuroHPCSystem, SystemSpecs] = {
    EuroHPCSystem.LUMI: SystemSpecs(
        gpu_type="AMD MI250X",
        gpu_memory_gb=128,  # 64GB per GCD, 2 GCDs per MI250X
        gpus_per_node=8,
        interconnect="Slingshot-11",
        storage_type="Lustre",
        modules=["LUMI/23.09", "rocm/5.6", "pytorch/2.2"],
        env_vars={
            "ROCR_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        }
    ),
    EuroHPCSystem.LEONARDO: SystemSpecs(
        gpu_type="NVIDIA A100",
        gpu_memory_gb=64,
        gpus_per_node=4,
        interconnect="InfiniBand HDR",
        storage_type="GPFS",
        modules=["cuda/12.1", "cudnn/8.9", "nccl/2.18", "python/3.11"],
        env_vars={
            "NCCL_IB_DISABLE": "0",
            "NCCL_NET_GDR_LEVEL": "5",
        }
    ),
    EuroHPCSystem.MARENOSTRUM5: SystemSpecs(
        gpu_type="NVIDIA H100",
        gpu_memory_gb=80,
        gpus_per_node=4,
        interconnect="InfiniBand NDR",
        storage_type="GPFS",
        modules=["cuda/12.2", "cudnn/9.0", "nccl/2.19", "python/3.11"],
        env_vars={
            "NCCL_IB_DISABLE": "0",
            "NCCL_NET_GDR_LEVEL": "5",
        }
    ),
    EuroHPCSystem.JUWELS: SystemSpecs(
        gpu_type="NVIDIA A100",
        gpu_memory_gb=80,
        gpus_per_node=4,
        interconnect="InfiniBand HDR200",
        storage_type="GPFS",
        modules=["CUDA/12.1", "cuDNN/8.9", "NCCL/2.18", "Python/3.11"],
        env_vars={
            "NCCL_IB_DISABLE": "0",
            "NCCL_NET_GDR_LEVEL": "5",
        }
    ),
    EuroHPCSystem.KAROLINA: SystemSpecs(
        gpu_type="NVIDIA A100",
        gpu_memory_gb=40,
        gpus_per_node=8,
        interconnect="InfiniBand HDR",
        storage_type="Lustre",
        modules=["CUDA/12.1", "cuDNN/8.9", "NCCL/2.18", "Python/3.11"],
        env_vars={
            "NCCL_IB_DISABLE": "0",
        }
    ),
    EuroHPCSystem.GENERIC: SystemSpecs(
        gpu_type="NVIDIA A100",
        gpu_memory_gb=80,
        gpus_per_node=4,
        interconnect="InfiniBand",
        storage_type="Lustre",
        modules=["cuda/12.1", "cudnn/8.9", "nccl/2.18", "python/3.11"],
        env_vars={}
    ),
}


@dataclass
class EuroHPCConfig:
    """
    Complete configuration for EuroHPC training.

    Combines all HPC-specific settings into a single configuration.
    """

    # System configuration
    system: EuroHPCSystem = EuroHPCSystem.GENERIC

    # Job configuration
    job_name: str = "capibara_train"
    nodes: int = 1
    time_limit: str = "24:00:00"
    partition: str = "gpu"
    account: Optional[str] = None
    qos: Optional[str] = None

    # Model configuration
    model_size: str = "7B"

    # Training configuration
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    max_steps: int = 100000
    warmup_steps: int = 2000

    # Distributed backend
    backend: DistributedBackend = DistributedBackend.DEEPSPEED

    # ZeRO configuration (DeepSpeed)
    zero_stage: int = 2
    cpu_offload: bool = False
    nvme_offload: bool = False
    nvme_path: str = "/local_nvme"

    # Mixed precision
    use_bf16: bool = True
    use_fp16: bool = False

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    max_checkpoints: int = 5

    # Logging
    log_dir: str = "logs"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Output
    output_dir: str = "output"

    # Advanced
    activation_checkpointing: bool = True
    compile_model: bool = True

    @property
    def specs(self) -> SystemSpecs:
        """Get system specifications."""
        return SYSTEM_SPECS.get(self.system, SYSTEM_SPECS[EuroHPCSystem.GENERIC])

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs."""
        return self.nodes * self.specs.gpus_per_node

    @property
    def global_batch_size(self) -> int:
        """Global batch size."""
        return self.micro_batch_size * self.gradient_accumulation_steps * self.total_gpus

    def get_slurm_config(self) -> SlurmConfig:
        """Generate SLURM configuration."""
        specs = self.specs

        return SlurmConfig(
            job_name=self.job_name,
            nodes=self.nodes,
            gpus_per_node=specs.gpus_per_node,
            cpus_per_task=8,
            memory_gb=256 if specs.gpu_memory_gb >= 80 else 128,
            time_limit=self.time_limit,
            partition=self.partition,
            account=self.account,
            qos=self.qos,
            modules=specs.modules,
            output_dir=self.log_dir,
            use_deepspeed=self.backend == DistributedBackend.DEEPSPEED,
            use_torchrun=self.backend != DistributedBackend.DEEPSPEED,
        )

    def get_deepspeed_config(self) -> DeepSpeedConfig:
        """Generate DeepSpeed configuration."""
        specs = self.specs

        return create_deepspeed_config(
            model_size=self.model_size,
            num_gpus=specs.gpus_per_node,
            num_nodes=self.nodes,
            gpu_memory_gb=specs.gpu_memory_gb,
            micro_batch_size=self.micro_batch_size,
            gradient_accumulation=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            use_cpu_offload=self.cpu_offload,
            use_nvme_offload=self.nvme_offload,
        )

    def get_fsdp_config(self) -> FSDPConfig:
        """Generate FSDP configuration."""
        specs = self.specs

        return create_fsdp_config(
            model_size=self.model_size,
            num_gpus=self.total_gpus,
            gpu_memory_gb=specs.gpu_memory_gb,
            use_cpu_offload=self.cpu_offload,
        )

    def get_checkpoint_config(self) -> CheckpointConfig:
        """Generate checkpoint configuration."""
        return CheckpointConfig(
            checkpoint_dir=self.checkpoint_dir,
            save_every_n_steps=self.save_every_n_steps,
            max_checkpoints=self.max_checkpoints,
            save_on_rank_0_only=True,
            include_metrics=True,
            verify_checkpoints=True,
        )

    def get_distributed_config(self) -> DistributedConfig:
        """Generate distributed configuration."""
        return DistributedConfig(
            nccl_timeout_minutes=30,
            nccl_async_error_handling=True,
        )

    def setup_environment(self):
        """Set up environment variables for the HPC system."""
        specs = self.specs

        # Set system-specific environment variables
        for key, value in specs.env_vars.items():
            os.environ[key] = value

        # Common optimizations
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        # CUDA settings
        if "NVIDIA" in specs.gpu_type:
            os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

        logger.info(f"Environment configured for {self.system.value}")


class EuroHPCTrainer:
    """
    High-level trainer for EuroHPC systems.

    Orchestrates distributed training with:
    - Automatic hardware detection
    - DeepSpeed or FSDP sharding
    - Distributed checkpointing
    - SLURM integration
    """

    def __init__(self, config: EuroHPCConfig, model=None):
        self.config = config
        self.model = model

        # Set up environment
        config.setup_environment()

        # Initialize distributed
        self.dist_config = config.get_distributed_config()
        self.dist_config = init_distributed(self.dist_config)

        self.rank = get_rank()
        self.world_size = get_world_size()

        # Initialize checkpoint manager
        self.checkpoint_manager = DistributedCheckpointManager(
            config.get_checkpoint_config(),
            rank=self.rank,
            world_size=self.world_size,
        )

        # Initialize distributed wrapper
        if config.backend == DistributedBackend.FSDP:
            self.fsdp_wrapper = FSDPWrapper(config.get_fsdp_config())
        else:
            self.fsdp_wrapper = None

        # DeepSpeed config
        if config.backend == DistributedBackend.DEEPSPEED:
            self.ds_config = config.get_deepspeed_config()
        else:
            self.ds_config = None

        logger.info(f"EuroHPC Trainer initialized: system={config.system.value}, "
                   f"backend={config.backend.value}, rank={self.rank}/{self.world_size}")

    def wrap_model(self, model):
        """Wrap model with distributed wrapper."""
        if self.config.backend == DistributedBackend.FSDP:
            return self.fsdp_wrapper.wrap_model(model)
        elif self.config.backend == DistributedBackend.DEEPSPEED:
            # DeepSpeed wrapping is done during training initialization
            return model
        else:
            # Standard DDP
            import torch.nn as nn
            return nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.dist_config.local_rank],
            )

    def save_checkpoint(self, step: int, epoch: int, model_state: Dict,
                       optimizer_state: Dict, metrics: Optional[Dict] = None,
                       is_best: bool = False) -> Optional[str]:
        """Save distributed checkpoint."""
        return self.checkpoint_manager.save(
            step=step,
            epoch=epoch,
            model_state=model_state,
            optimizer_state=optimizer_state,
            metrics=metrics,
            is_best=is_best,
        )

    def load_checkpoint(self, path: Optional[str] = None,
                       load_best: bool = False) -> Optional[Dict]:
        """Load checkpoint."""
        return self.checkpoint_manager.load(
            checkpoint_path=path,
            load_best=load_best,
            load_latest=path is None and not load_best,
        )

    def generate_slurm_script(self, output_path: str, training_script: str,
                             training_args: Optional[List[str]] = None) -> str:
        """Generate SLURM submission script."""
        slurm_config = self.config.get_slurm_config()
        slurm_config.training_script = training_script
        slurm_config.training_args = training_args or []

        if self.ds_config:
            ds_config_path = Path(self.config.output_dir) / "deepspeed_config.json"
            ds_config_path.parent.mkdir(parents=True, exist_ok=True)
            self.ds_config.save(ds_config_path)
            slurm_config.deepspeed_config = str(ds_config_path)

        launcher = SlurmLauncher(slurm_config)
        return launcher.generate_script(output_path)


def create_eurohpc_trainer(
    system: Union[str, EuroHPCSystem] = "generic",
    model_size: str = "7B",
    nodes: int = 1,
    backend: Union[str, DistributedBackend] = "deepspeed",
    **kwargs
) -> EuroHPCTrainer:
    """
    Factory function to create EuroHPC trainer.

    Args:
        system: EuroHPC system name
        model_size: Model size (e.g., "3B", "7B", "13B", "30B", "70B")
        nodes: Number of nodes
        backend: Distributed backend ("deepspeed", "fsdp", "ddp")
        **kwargs: Additional configuration options

    Returns:
        Configured EuroHPCTrainer
    """
    if isinstance(system, str):
        system = EuroHPCSystem(system.lower())

    if isinstance(backend, str):
        backend = DistributedBackend(backend.lower())

    config = EuroHPCConfig(
        system=system,
        model_size=model_size,
        nodes=nodes,
        backend=backend,
        **kwargs
    )

    return EuroHPCTrainer(config)


def detect_eurohpc_system() -> EuroHPCSystem:
    """Auto-detect which EuroHPC system we're running on."""
    # Check environment variables and hostnames
    hostname = os.environ.get("HOSTNAME", "").lower()

    if "lumi" in hostname or "LUMI" in os.environ.get("LMOD_SYSTEM_NAME", ""):
        return EuroHPCSystem.LUMI
    elif "leonardo" in hostname or "leo" in hostname:
        return EuroHPCSystem.LEONARDO
    elif "mn5" in hostname or "marenostrum" in hostname:
        return EuroHPCSystem.MARENOSTRUM5
    elif "juwels" in hostname or "jwb" in hostname:
        return EuroHPCSystem.JUWELS
    elif "karolina" in hostname:
        return EuroHPCSystem.KAROLINA
    elif "meluxina" in hostname:
        return EuroHPCSystem.MELUXINA
    else:
        return EuroHPCSystem.GENERIC
