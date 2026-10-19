"""
Distributed configuration utilities for CapibaraGPT core.

This module provides a minimal, **working** mesh configuration for JAX sharding.
It avoids implicit sys.path hacks and fails fast when the requested device type
or mesh shape is not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from jax import devices, local_devices
from jax.sharding import Mesh, PartitionSpec as P

logger = logging.getLogger(__name__)


TPU_MESH_SHAPE: Tuple[int, ...] = (1,)
TPU_MESH_AXIS_NAMES: Tuple[str, ...] = ("data",)
TPU_MESH: Optional[Mesh] = None


def _prod(shape: Sequence[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _reshape_devices(devs: Sequence[Any], shape: Sequence[int]) -> Any:
    """Reshape devices into a nested list for Mesh construction."""
    total = _prod(shape)
    if total != len(devs):
        raise ValueError(f"Device count {len(devs)} does not match mesh size {total}")

    def build(level: int, offset: int) -> Tuple[Any, int]:
        if level == len(shape) - 1:
            end = offset + shape[level]
            return list(devs[offset:end]), end
        items = []
        for _ in range(shape[level]):
            item, offset = build(level + 1, offset)
            items.append(item)
        return items, offset

    nested, _ = build(0, 0)
    return nested


def _resolve_devices(device_type: str) -> Sequence[Any]:
    device_key = device_type.lower()
    if device_key.startswith("tpu"):
        devs = devices("tpu")
    elif device_key in {"gpu", "cuda"}:
        devs = devices("gpu")
    else:
        devs = devices()

    if not devs:
        raise RuntimeError(f"No devices available for device_type='{device_type}'")
    return devs


@dataclass
class DistributedSystem:
    mesh: Mesh
    mesh_shape: Tuple[int, ...]
    mesh_axis_names: Tuple[str, ...]
    global_device_count: int
    local_device_count: int
    sharding_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TPUDistributionConfig:
    """Minimal mesh configuration for distributed execution."""

    mesh_shape: Tuple[int, ...] = TPU_MESH_SHAPE
    mesh_axis_names: Tuple[str, ...] = TPU_MESH_AXIS_NAMES
    device_type: str = "cpu"
    num_partitions: Optional[int] = None

    def setup_distributed_training(
        self, sharding_config: Optional[Dict[str, Any]] = None
    ) -> DistributedSystem:
        devs = list(_resolve_devices(self.device_type))
        mesh_size = _prod(self.mesh_shape)

        if mesh_size > len(devs):
            raise ValueError(
                f"Mesh shape {self.mesh_shape} requires {mesh_size} devices; "
                f"only {len(devs)} available for {self.device_type}."
            )

        if self.num_partitions is not None and self.num_partitions > len(devs):
            raise ValueError(
                f"num_partitions={self.num_partitions} exceeds device count {len(devs)}."
            )

        devs = devs[:mesh_size]
        mesh_devices = _reshape_devices(devs, self.mesh_shape)
        mesh = Mesh(mesh_devices, self.mesh_axis_names)
        logger.info(
            "Distributed mesh initialized: shape=%s axis_names=%s devices=%s",
            self.mesh_shape,
            self.mesh_axis_names,
            len(devs),
        )

        return DistributedSystem(
            mesh=mesh,
            mesh_shape=self.mesh_shape,
            mesh_axis_names=self.mesh_axis_names,
            global_device_count=len(devs),
            local_device_count=len(local_devices()),
            sharding_config=sharding_config or {},
        )


def create_mesh_config(
    *,
    mesh_shape: Optional[Tuple[int, ...]] = None,
    data_parallel: Optional[int] = None,
    model_parallel: Optional[int] = None,
    pipeline_parallel: Optional[int] = None,
    device_type: str = "cpu",
    mesh_axis_names: Optional[Tuple[str, ...]] = None,
) -> TPUDistributionConfig:
    """Create a mesh configuration from high-level parameters."""
    if mesh_shape is None:
        if data_parallel is None or model_parallel is None:
            raise ValueError("Provide mesh_shape or data_parallel + model_parallel.")
        if pipeline_parallel is None:
            mesh_shape = (data_parallel, model_parallel)
        else:
            mesh_shape = (data_parallel, model_parallel, pipeline_parallel)

    if mesh_axis_names is None:
        default_axes = ("data", "model", "pipeline")
        mesh_axis_names = tuple(default_axes[: len(mesh_shape)])

    return TPUDistributionConfig(
        mesh_shape=mesh_shape,
        mesh_axis_names=mesh_axis_names,
        device_type=device_type,
        num_partitions=_prod(mesh_shape),
    )


def setup_mesh(
    shape: Tuple[int, ...] = TPU_MESH_SHAPE,
    axis_names: Tuple[str, ...] = TPU_MESH_AXIS_NAMES,
    device_type: str = "cpu",
) -> Mesh:
    """Initialize and store a global mesh instance."""
    global TPU_MESH
    config = TPUDistributionConfig(
        mesh_shape=shape,
        mesh_axis_names=axis_names,
        device_type=device_type,
    )
    system = config.setup_distributed_training()
    TPU_MESH = system.mesh
    return TPU_MESH


__all__ = [
    "P",
    "TPUDistributionConfig",
    "DistributedSystem",
    "create_mesh_config",
    "setup_mesh",
    "TPU_MESH_SHAPE",
    "TPU_MESH_AXIS_NAMES",
]
