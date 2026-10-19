"""
Backend Utilities

Helper functions for hardware detection, memory management,
and cross-backend operations.
"""

import functools
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from core.decorators import cached_computation

logger = logging.getLogger(__name__)


def _torch_import_works() -> bool:
    """Check torch import in a subprocess to avoid hard crashes."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


@cached_computation(maxsize=1, ttl_seconds=60.0)
def detect_available_hardware() -> Dict[str, Any]:
    """
    Detect all available compute hardware.

    Cached for 60 seconds since hardware doesn't change frequently.

    Returns:
        Dictionary with hardware information:
        - tpu: TPU availability and configuration
        - gpu: GPU availability and specifications
        - cpu: CPU information
    """
    result = {
        "tpu": {"available": False, "devices": [], "topology": None},
        "gpu": {"available": False, "devices": [], "cuda_version": None},
        "cpu": {"available": True, "cores": os.cpu_count()},
    }

    # Check TPU
    try:
        import jax
        tpu_devices = jax.devices("tpu")
        if tpu_devices:
            result["tpu"] = {
                "available": True,
                "devices": [str(d) for d in tpu_devices],
                "device_count": len(tpu_devices),
                "topology": _detect_tpu_topology(len(tpu_devices)),
            }
    except Exception:
        pass

    # Check GPU
    try:
        if not _torch_import_works():
            raise ImportError("torch import failed")
        import torch
        if torch.cuda.is_available():
            gpu_devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_devices.append({
                    "id": i,
                    "name": props.name,
                    "memory_gb": props.total_memory / (1024 ** 3),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "is_a100": "A100" in props.name,
                })
            result["gpu"] = {
                "available": True,
                "devices": gpu_devices,
                "device_count": len(gpu_devices),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
    except Exception:
        pass

    return result


@functools.lru_cache(maxsize=32)
def _detect_tpu_topology(num_devices: int) -> str:
    """Detect TPU topology from device count (cached - pure function)."""
    topologies = {
        8: "v4-8",
        32: "v4-32",
        64: "v6e-64",
        128: "v4-128",
        256: "v4-256",
    }
    return topologies.get(num_devices, f"custom-{num_devices}")


def get_device_info(backend_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get detailed information about current device.

    Args:
        backend_name: Backend to query (tpu, gpu, cpu)

    Returns:
        Device information dictionary
    """
    if backend_name is None:
        # Auto-detect
        hw = detect_available_hardware()
        if hw["tpu"]["available"]:
            backend_name = "tpu"
        elif hw["gpu"]["available"]:
            backend_name = "gpu"
        else:
            backend_name = "cpu"

    info = {"backend": backend_name}

    if backend_name == "tpu":
        try:
            import jax
            devices = jax.devices("tpu")
            info.update({
                "device_count": len(devices),
                "topology": _detect_tpu_topology(len(devices)),
                "jax_version": jax.__version__,
                "platform": jax.default_backend(),
            })
        except Exception as e:
            info["error"] = str(e)

    elif backend_name == "gpu":
        try:
            if not _torch_import_works():
                raise ImportError("torch import failed")
            import torch
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            info.update({
                "device_id": device,
                "device_name": props.name,
                "memory_total_gb": props.total_memory / (1024 ** 3),
                "memory_allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
                "memory_reserved_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
                "compute_capability": f"{props.major}.{props.minor}",
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            })
        except Exception as e:
            info["error"] = str(e)

    elif backend_name == "cpu":
        import platform
        info.update({
            "cores": os.cpu_count(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        })

    return info


def sync_device(backend_name: str = "auto") -> None:
    """
    Synchronize device operations (wait for completion).

    Args:
        backend_name: Backend to sync (auto, tpu, gpu, cpu)
    """
    if backend_name == "auto":
        hw = detect_available_hardware()
        if hw["gpu"]["available"]:
            backend_name = "gpu"
        elif hw["tpu"]["available"]:
            backend_name = "tpu"
        else:
            return

    if backend_name == "gpu":
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass

    elif backend_name == "tpu":
        try:
            import jax
            jax.block_until_ready
        except Exception:
            pass


def memory_stats(backend_name: str = "auto") -> Dict[str, float]:
    """
    Get memory statistics in GB.

    Args:
        backend_name: Backend to query (auto, tpu, gpu, cpu)

    Returns:
        Dictionary with memory stats:
        - allocated: Currently allocated memory
        - reserved: Currently reserved memory
        - total: Total device memory
    """
    if backend_name == "auto":
        hw = detect_available_hardware()
        if hw["gpu"]["available"]:
            backend_name = "gpu"
        elif hw["tpu"]["available"]:
            backend_name = "tpu"
        else:
            backend_name = "cpu"

    stats = {
        "allocated_gb": 0.0,
        "reserved_gb": 0.0,
        "total_gb": 0.0,
        "backend": backend_name,
    }

    if backend_name == "gpu":
        try:
            if not _torch_import_works():
                raise ImportError("torch import failed")
            import torch
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            stats.update({
                "allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
                "reserved_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
                "total_gb": props.total_memory / (1024 ** 3),
            })
        except Exception:
            pass

    elif backend_name == "tpu":
        try:
            import jax
            devices = jax.devices("tpu")
            if devices:
                device = devices[0]
                mem_stats = {}
                memory_stats_fn = getattr(device, "memory_stats", None)
                if callable(memory_stats_fn):
                    mem_stats = memory_stats_fn() or {}

                bytes_in_use = (
                    mem_stats.get("bytes_in_use")
                    or mem_stats.get("bytes_used")
                    or mem_stats.get("allocated_bytes")
                    or 0
                )
                bytes_limit = (
                    mem_stats.get("bytes_limit")
                    or mem_stats.get("memory_limit")
                    or mem_stats.get("bytes_reserved")
                    or 0
                )

                stats.update({
                    "allocated_gb": bytes_in_use / (1024 ** 3),
                    "reserved_gb": bytes_limit / (1024 ** 3),
                    "total_gb": bytes_limit / (1024 ** 3),
                })
            else:
                stats.update({
                    "allocated_gb": 0.0,
                    "reserved_gb": 0.0,
                    "total_gb": 0.0,
                })
        except Exception:
            stats.update({
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "total_gb": 0.0,
            })

    elif backend_name == "cpu":
        try:
            import psutil
            mem = psutil.virtual_memory()
            stats.update({
                "allocated_gb": mem.used / (1024 ** 3),
                "reserved_gb": 0.0,
                "total_gb": mem.total / (1024 ** 3),
            })
        except ImportError:
            pass

    return stats


def format_memory_stats(stats: Dict[str, float]) -> str:
    """Format memory stats as human-readable string."""
    return (
        f"Memory ({stats['backend'].upper()}): "
        f"{stats['allocated_gb']:.2f}GB allocated / "
        f"{stats['total_gb']:.2f}GB total"
    )


def estimate_model_memory(
    num_params: int,
    dtype: str = "bfloat16",
    optimizer: str = "adamw",
    batch_size: int = 1,
    seq_len: int = 2048,
) -> Dict[str, float]:
    """
    Estimate memory requirements for training a model.

    Args:
        num_params: Number of model parameters
        dtype: Data type (float32, float16, bfloat16)
        optimizer: Optimizer type (adamw, sgd)
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Estimated memory in GB
    """
    # Bytes per element
    bytes_per_element = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }.get(dtype, 2)

    # Model parameters
    param_memory = num_params * bytes_per_element

    # Gradients (same size as parameters)
    grad_memory = param_memory

    # Optimizer states
    if optimizer == "adamw":
        # AdamW: 2 states (m, v) per parameter
        optimizer_memory = 2 * num_params * 4  # Always float32
    else:
        optimizer_memory = 0

    # Activations (rough estimate)
    # Assume ~10 bytes per token per billion parameters
    activation_memory = batch_size * seq_len * (num_params / 1e9) * 10 * bytes_per_element

    # Total (add 20% overhead)
    total_memory = (param_memory + grad_memory + optimizer_memory + activation_memory) * 1.2

    return {
        "parameters_gb": param_memory / (1024 ** 3),
        "gradients_gb": grad_memory / (1024 ** 3),
        "optimizer_gb": optimizer_memory / (1024 ** 3),
        "activations_gb": activation_memory / (1024 ** 3),
        "total_gb": total_memory / (1024 ** 3),
        "dtype": dtype,
        "optimizer": optimizer,
    }
