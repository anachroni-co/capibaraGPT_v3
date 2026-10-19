"""
Vim VQ Config - Configuration for Vector Quantization with hardware support.

This module provides configuration for ViM-VQ (Vector-Quantized Vision Model)
with optimized support for TPU v4-32, TPU v6, and ARM Axion processors.

Key Components:
    - VimVQConfig: Configuration dataclass for VQ settings

Features:
    - Sub-vector quantization parameters
    - Hardware-specific optimizations (TPU, ARM, CPU)
    - Memory pool configuration
    - Multimodal support (audio, vision)
    - Preset configurations for different hardware

Author: Skydesk International Dev Team.
"""

from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class VimVQConfig:
    """setup for ViM-VQ with soporte tpu v4-32 and ARM Axion"""
    
    # Parámetros básicos VQ
    d: int = 8                     # dimension de sub-vectores
    k: int = 256                   # size de codebook
    n_neighbors: int = 4           # Vecinos cercanos for convex combination
    
    # Parámetros de optimization
    max_iters: int = 20           # Iteraciones máximas
    lr: float = 1e-2              # Learning rate
    convergence_threshold: float = 1e-6  # Umbral convergencia
    incremental_threshold: float = 0.99  # Umbral for hard assignment
    
    # setup hardware
    device: Literal["tpu", "arm", "cpu"] = "tpu"
    tpu_version: Literal["v4-32", "v6"] = "v4-32"
    use_arm_sve: bool = True      # use SVE/SVE2 en ARM
    
    # Optimizaciones memory
    use_memory_pool: bool = True  # use memory pools optimizados
    cache_size_mb: int = 512      # size cache for codebook
    
    # setup multimodal
    support_audio: bool = True    # Soporte for features audio
    support_vision: bool = True   # Soporte for features visuales
    
    # tpu específico
    tpu_batch_size: int = 128
    use_bfloat16: bool = True
    tpu_async_copy: bool = True
    
    # ARM específico
    arm_sve_length: Optional[int] = None  # Auto-detectado if None
    arm_numa_node: int = 0
    use_huge_pages: bool = True
    
    @classmethod
    def from_preset(cls, preset: Literal["tpu_v4", "tpu_v6", "arm_axion"]) -> "VimVQConfig":
        """Crea setup since preset predefinido"""
        if preset == "tpu_v4":
            return cls(
                d=8, k=256, device="tpu", tpu_version="v4-32",
                use_memory_pool=True, cache_size_mb=512
            )
        elif preset == "tpu_v6":
            return cls(
                d=16, k=512, device="tpu", tpu_version="v6",
                use_memory_pool=True, cache_size_mb=1024
            )
        elif preset == "arm_axion":
            return cls(
                d=8, k=128, device="arm", use_arm_sve=True,
                use_memory_pool=True, cache_size_mb=256
            )
        else:
            raise ValueError(f"Preset desconocido: {preset}")

    @classmethod
    def for_tpu_v4_32(cls) -> "VimVQConfig":
        """setup optimizada for tpu v4-32"""
        return cls(
            d=8,
            k=256,
            n_neighbors=4,
            device="tpu",
            use_memory_pool=True,
            cache_size_mb=2048,
            tpu_batch_size=256,
            use_bfloat16=True,
            tpu_async_copy=True
        )
    
    @classmethod
    def for_arm_axion(cls) -> "VimVQConfig":
        """setup optimizada for ARM Axion"""
        return cls(
            d=8,
            k=128,  # Reducido for better rendimiento en ARM
            n_neighbors=4,
            device="arm",
            use_memory_pool=True,
            cache_size_mb=1024,
            use_huge_pages=True
        )

    def validate(self):
        """Valida la setup"""
        assert self.d > 0, "d debe ser positivo"
        assert self.k > 0, "k debe ser positivo"
        assert self.n_neighbors > 0, "n_neighbors debe ser positivo"
        assert self.n_neighbors <= self.k, "n_neighbors debe ser <= k"
        assert self.cache_size_mb > 0, "cache_size_mb debe ser positivo"
        
        if self.device == "tpu":
            assert self.tpu_batch_size > 0, "tpu_batch_size debe ser positivo"
        elif self.device == "arm":
            assert self.arm_numa_node >= 0, "arm_numa_node debe ser >= 0"

# Configuraciones predefinidas
DEFAULT_TPU_V4_CONFIG = VimVQConfig.from_preset("tpu_v4")
DEFAULT_TPU_V6_CONFIG = VimVQConfig.from_preset("tpu_v6")
DEFAULT_ARM_CONFIG = VimVQConfig.from_preset("arm_axion") 