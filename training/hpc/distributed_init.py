"""
Distributed Training Initialization for HPC Environments

Handles multi-node distributed training setup with support for:
- SLURM environment detection
- NCCL/Gloo backend initialization
- Process group management
- Rank and world size coordination
"""

import os
import socket
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
torch = None
dist = None


def _ensure_torch():
    global torch, dist
    if torch is None:
        import torch as _torch
        import torch.distributed as _dist
        torch = _torch
        dist = _dist


class DistributedBackend(Enum):
    """Supported distributed backends."""
    NCCL = "nccl"      # NVIDIA GPUs (recommended for HPC)
    GLOO = "gloo"      # CPU or fallback
    MPI = "mpi"        # MPI-based (some HPC systems)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    backend: DistributedBackend = DistributedBackend.NCCL

    # Process configuration (auto-detected from SLURM if available)
    world_size: Optional[int] = None
    rank: Optional[int] = None
    local_rank: Optional[int] = None

    # Network configuration
    master_addr: Optional[str] = None
    master_port: int = 29500

    # NCCL configuration
    nccl_timeout_minutes: int = 30
    nccl_async_error_handling: bool = True
    nccl_blocking_wait: bool = True

    # Performance tuning
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False

    # Debugging
    debug: bool = False

    def __post_init__(self):
        """Auto-detect configuration from environment."""
        self._detect_slurm_config()
        self._detect_torchrun_config()
        self._set_defaults()

    def _detect_slurm_config(self):
        """Detect configuration from SLURM environment variables."""
        if "SLURM_JOB_ID" not in os.environ:
            return

        logger.info("Detected SLURM environment")

        # World size from SLURM
        if self.world_size is None:
            ntasks = os.environ.get("SLURM_NTASKS")
            nnodes = os.environ.get("SLURM_NNODES", "1")
            gpus_per_node = os.environ.get("SLURM_GPUS_ON_NODE",
                                           os.environ.get("SLURM_GPUS_PER_NODE", "1"))

            if ntasks:
                self.world_size = int(ntasks)
            else:
                self.world_size = int(nnodes) * int(gpus_per_node.split(":")[-1] if ":" in str(gpus_per_node) else gpus_per_node)

        # Rank from SLURM
        if self.rank is None:
            self.rank = int(os.environ.get("SLURM_PROCID", "0"))

        # Local rank from SLURM
        if self.local_rank is None:
            self.local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

        # Master address from SLURM
        if self.master_addr is None:
            nodelist = os.environ.get("SLURM_NODELIST", "")
            if nodelist:
                self.master_addr = self._parse_slurm_nodelist(nodelist)

    def _detect_torchrun_config(self):
        """Detect configuration from torchrun/torch.distributed.launch."""
        if "RANK" in os.environ:
            if self.rank is None:
                self.rank = int(os.environ["RANK"])
            if self.world_size is None:
                self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if self.local_rank is None:
                self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if self.master_addr is None:
                self.master_addr = os.environ.get("MASTER_ADDR", "localhost")
            if "MASTER_PORT" in os.environ:
                self.master_port = int(os.environ["MASTER_PORT"])

    def _set_defaults(self):
        """Set default values for any unset configuration."""
        if self.world_size is None:
            self.world_size = 1
        if self.rank is None:
            self.rank = 0
        if self.local_rank is None:
            self.local_rank = 0
        if self.master_addr is None:
            self.master_addr = "localhost"

    def _parse_slurm_nodelist(self, nodelist: str) -> str:
        """Parse SLURM nodelist to get master node address."""
        try:
            import subprocess
            result = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True,
                text=True,
                check=True
            )
            nodes = result.stdout.strip().split("\n")
            if nodes:
                master_node = nodes[0]
                try:
                    return socket.gethostbyname(master_node)
                except socket.gaierror:
                    return master_node
        except Exception as e:
            logger.warning(f"Failed to parse SLURM nodelist: {e}")

        return "localhost"


# Global distributed state
_distributed_initialized = False
_config: Optional[DistributedConfig] = None


def init_distributed(config: Optional[DistributedConfig] = None) -> DistributedConfig:
    """
    Initialize distributed training environment.

    Args:
        config: Optional configuration. If None, auto-detects from environment.

    Returns:
        The effective DistributedConfig.
    """
    global _distributed_initialized, _config

    if _distributed_initialized:
        logger.warning("Distributed already initialized, returning existing config")
        return _config

    _ensure_torch()

    config = config or DistributedConfig()
    _config = config

    if config.world_size == 1:
        logger.info("Single-process mode, skipping distributed initialization")
        _distributed_initialized = True
        return config

    # Set environment variables for PyTorch
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    os.environ["WORLD_SIZE"] = str(config.world_size)
    os.environ["RANK"] = str(config.rank)
    os.environ["LOCAL_RANK"] = str(config.local_rank)

    # NCCL configuration for HPC
    if config.backend == DistributedBackend.NCCL:
        os.environ["NCCL_DEBUG"] = "WARN" if not config.debug else "INFO"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" if config.nccl_async_error_handling else "0"
        os.environ["NCCL_BLOCKING_WAIT"] = "1" if config.nccl_blocking_wait else "0"

        # InfiniBand optimizations for HPC
        if "SLURM_JOB_ID" in os.environ:
            os.environ.setdefault("NCCL_IB_DISABLE", "0")
            os.environ.setdefault("NCCL_NET_GDR_LEVEL", "5")
            os.environ.setdefault("NCCL_IB_GID_INDEX", "3")

    # Initialize process group
    timeout = torch.distributed.timedelta(minutes=config.nccl_timeout_minutes)

    try:
        dist.init_process_group(
            backend=config.backend.value,
            init_method=f"tcp://{config.master_addr}:{config.master_port}",
            world_size=config.world_size,
            rank=config.rank,
            timeout=timeout,
        )

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(config.local_rank)

        _distributed_initialized = True

        logger.info(
            f"Distributed initialized: rank={config.rank}/{config.world_size}, "
            f"local_rank={config.local_rank}, master={config.master_addr}:{config.master_port}"
        )

    except Exception as e:
        logger.error(f"Failed to initialize distributed: {e}")
        raise

    return config


def get_rank() -> int:
    """Get current process rank."""
    _ensure_torch()
    if dist.is_initialized():
        return dist.get_rank()
    return _config.rank if _config else 0


def get_world_size() -> int:
    """Get total number of processes."""
    _ensure_torch()
    if dist.is_initialized():
        return dist.get_world_size()
    return _config.world_size if _config else 1


def get_local_rank() -> int:
    """Get local rank (within node)."""
    return _config.local_rank if _config else 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    _ensure_torch()
    if dist.is_initialized() and get_world_size() > 1:
        dist.barrier()


def cleanup_distributed():
    """Clean up distributed resources."""
    global _distributed_initialized, _config

    _ensure_torch()

    if dist.is_initialized():
        dist.destroy_process_group()

    _distributed_initialized = False
    _config = None

    logger.info("Distributed cleanup complete")


def get_device() -> "torch.device":
    """Get the device for the current process."""
    _ensure_torch()

    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_local_rank()}")
    return torch.device("cpu")


def all_reduce_mean(tensor: "torch.Tensor") -> "torch.Tensor":
    """All-reduce tensor and compute mean across processes."""
    _ensure_torch()

    if not dist.is_initialized() or get_world_size() == 1:
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast Python object from src to all processes."""
    _ensure_torch()

    if not dist.is_initialized() or get_world_size() == 1:
        return obj

    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]
