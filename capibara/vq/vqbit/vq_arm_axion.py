"""
CapibaraGPT v3.3 - Optimizaciones ARM Axion
implementation optimizada for procesadores ARM Axion with 64 códigos VQ
"""

from __future__ import annotations

import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

try:
    import torch
    import torch_arm
    from torch_arm.sve import SVELinearFunction
    from torch_arm.quantization import QuantizedLinear
    ARM_AXION_AVAILABLE = True
except ImportError:
    ARM_AXION_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ARMAxionConfig:
    """setup for optimizaciones ARM Axion."""
    
    # VQ settings
    embedding_dim: int = 4096
    num_codes: int = 64  # Optimizado for ARM
    
    # SVE2 settings
    enable_sve2: bool = True
    sve_vector_bits: int = 512
    
    # Quantization
    enable_quantization: bool = True
    quantization_bits: int = 8
    quantization_scheme: str = "symmetric"
    
    # Memory settings
    max_batch_size: int = 32
    enable_memory_opt: bool = True
    chunk_size: int = 4096
    
    # Performance
    num_threads: int = 8
    enable_kleidi: bool = True
    enable_cache: bool = True

class ARMAxionOptimizer:
    """Optimizador específico for ARM Axion."""
    
    def __init__(self, config: Optional[ARMAxionConfig] = None):
        self.config = config or ARMAxionConfig()
        self._initialize_arm_optimizations()
        
    def _initialize_arm_optimizations(self):
        """Inicializar optimizaciones ARM."""
        if not ARM_AXION_AVAILABLE:
            logger.warning(" ARM Axion optimizations not available")
            return
            
        try:
            # configure SVE2
            if self.config.enable_sve2:
                torch_arm.config.set_vector_bits(self.config.sve_vector_bits)
                logger.info(f" SVE2 enabled with {self.config.sve_vector_bits} bits")
            
            # configure threads
            torch_arm.config.set_num_threads(self.config.num_threads)
            
            # Inicializar Kleidi if está available
            if self.config.enable_kleidi:
                try:
                    from capibara.core.kleidi import KleidiOptimizer
                except Exception as e:
                    self.config.enable_kleidi = False
                    logger.warning(" Kleidi optimization unavailable: %s", e)
                else:
                    self.kleidi = KleidiOptimizer()
                    logger.info(" Kleidi optimization enabled")
            
            logger.info(" ARM Axion optimizations initialized")
            
        except Exception as e:
            logger.error(f" Error initializing ARM optimizations: {e}")
    
    def optimize_linear(self, layer: torch.nn.Linear) -> torch.nn.Module:
        """optimize capa lineal for ARM."""
        if not ARM_AXION_AVAILABLE:
            return layer
            
        try:
            # apply SVE2
            if self.config.enable_sve2:
                layer = SVELinearFunction.apply(layer)
            
            # apply cuantización
            if self.config.enable_quantization:
                layer = QuantizedLinear(
                    layer,
                    bits=self.config.quantization_bits,
                    scheme=self.config.quantization_scheme
                )
            
            return layer
            
        except Exception as e:
            logger.warning(f"Failed to optimize linear layer: {e}")
            return layer
    
    def optimize_attention(self, query: torch.Tensor, key: torch.Tensor, 
                         value: torch.Tensor) -> torch.Tensor:
        """optimize atención for ARM."""
        if not ARM_AXION_AVAILABLE:
            return torch.matmul(
                torch.softmax(torch.matmul(query, key.transpose(-2, -1)), dim=-1),
                value
            )
        
        try:
            # use Kleidi if está available
            if self.config.enable_kleidi and hasattr(self, 'kleidi'):
                return self.kleidi.optimized_attention(query, key, value)
            
            # implementation optimizada manual
            scores = torch_arm.matmul(query, key.transpose(-2, -1))
            scores = scores / np.sqrt(query.size(-1))
            
            # optimize softmax
            if self.config.enable_memory_opt:
                # process en chunks for ahorrar memory
                chunks = torch.chunk(scores, self.config.chunk_size, dim=-1)
                weights = [torch_arm.softmax(chunk, dim=-1) for chunk in chunks]
                weights = torch.cat(weights, dim=-1)
            else:
                weights = torch_arm.softmax(scores, dim=-1)
            
            return torch_arm.matmul(weights, value)
            
        except Exception as e:
            logger.warning(f"Failed to optimize attention: {e}")
            return torch.matmul(
                torch.softmax(torch.matmul(query, key.transpose(-2, -1)), dim=-1),
                value
            )
    
    def optimize_vq_forward(self, x: torch.Tensor, codebook: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass optimizado for VQ."""
        if not ARM_AXION_AVAILABLE:
            return self._vq_forward_cpu(x, codebook)
        
        try:
            # Reshape input
            flat_x = x.reshape(-1, x.size(-1))
            
            # calculate distancias usando SVE2
            distances = -torch_arm.cdist(flat_x, codebook)
            
            # find códigos more cercanos
            encoding_indices = torch_arm.argmax(distances, dim=1)
            
            # Cuantizar
            quantized = torch.index_select(codebook, 0, encoding_indices)
            quantized = quantized.view_as(x)
            
            return quantized, encoding_indices
            
        except Exception as e:
            logger.warning(f"Failed to optimize VQ forward: {e}")
            return self._vq_forward_cpu(x, codebook)
    
    def _vq_forward_cpu(self, x: torch.Tensor, codebook: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """implementation cpu fallback for VQ."""
        flat_x = x.reshape(-1, x.size(-1))
        distances = torch.cdist(flat_x, codebook)
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = torch.index_select(codebook, 0, encoding_indices)
        return quantized.view_as(x), encoding_indices
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """obtain métricas de rendimiento."""
        metrics = {
            "arm_axion_available": ARM_AXION_AVAILABLE,
            "sve2_enabled": self.config.enable_sve2 if ARM_AXION_AVAILABLE else False,
            "quantization_enabled": self.config.enable_quantization if ARM_AXION_AVAILABLE else False,
            "kleidi_enabled": hasattr(self, 'kleidi') if ARM_AXION_AVAILABLE else False,
            "num_threads": self.config.num_threads if ARM_AXION_AVAILABLE else 1
        }
        
        if ARM_AXION_AVAILABLE:
            try:
                import psutil
                metrics.update({
                    "cpu_usage": psutil.cpu_percent(interval=1),
                    "memory_usage": psutil.Process().memory_info().rss / 1024**2,  # MB
                    "threads_active": psutil.Process().num_threads()
                })
            except:
                pass
        
        return metrics

def create_arm_axion_optimizer(config: Optional[ARMAxionConfig] = None) -> ARMAxionOptimizer:
    """create optimizador ARM Axion."""
    return ARMAxionOptimizer(config) 


# Backwards-compatible alias expected by vqbit __init__
ARMAxionVQOptimizer = ARMAxionOptimizer
