"""
HPC (High Performance Computing) Module for CapibaraGPT

Provides integration with EuroHPC and similar supercomputing infrastructures:
- SLURM job scheduler integration
- DeepSpeed ZeRO optimization (Stage 1/2/3)
- PyTorch FSDP (Fully Sharded Data Parallel)
- Multi-node distributed initialization
- Robust checkpoint/restart for long-running jobs
"""

from .distributed_init import (
    DistributedConfig,
    init_distributed,
    get_rank,
    get_world_size,
    get_local_rank,
    is_main_process,
    barrier,
    cleanup_distributed,
)

from .slurm_launcher import (
    SlurmConfig,
    SlurmLauncher,
    detect_slurm_environment,
    get_slurm_job_info,
)

from .deepspeed_config import (
    DeepSpeedConfig,
    create_deepspeed_config,
    ZeROStage,
)

from .fsdp_wrapper import (
    FSDPConfig,
    FSDPWrapper,
    ShardingStrategy,
)

from .eurohpc_config import (
    EuroHPCConfig,
    EuroHPCSystem,
    create_eurohpc_trainer,
)

from .checkpoint_manager import (
    DistributedCheckpointManager,
    CheckpointConfig,
)

__all__ = [
    # Distributed
    "DistributedConfig",
    "init_distributed",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "is_main_process",
    "barrier",
    "cleanup_distributed",
    # SLURM
    "SlurmConfig",
    "SlurmLauncher",
    "detect_slurm_environment",
    "get_slurm_job_info",
    # DeepSpeed
    "DeepSpeedConfig",
    "create_deepspeed_config",
    "ZeROStage",
    # FSDP
    "FSDPConfig",
    "FSDPWrapper",
    "ShardingStrategy",
    # EuroHPC
    "EuroHPCConfig",
    "EuroHPCSystem",
    "create_eurohpc_trainer",
    # Checkpointing
    "DistributedCheckpointManager",
    "CheckpointConfig",
]
