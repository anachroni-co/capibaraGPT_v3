"""Router especializado for ViM-VQ with soporte tpu v4-32 and ARM Axion"""

from typing import Dict, Optional, Union, List
import numpy as np

from jax import numpy as jnp
from jax import random
from capibara.core.routers.base import BaseRouter
from capibara.core.router import EnhancedRouter

from ..configs.vim_vq_config import VimVQConfig
from .quantizer import ViMVQQuantizer

class ViMVQRouter(BaseRouter):
    """Router especializado for cuantización ViM-VQ"""
    
    def __init__(self, 
                 config: Optional[VimVQConfig] = None,
                 key: Optional[random.PRNGKey] = None,
                 parent_router: Optional[EnhancedRouter] = None):
        """Inicializa router ViM-VQ
        
        Args:
            config: setup ViM-VQ
            key: JAX random key
            parent_router: Router padre for integration
        """
        super().__init__()
        
        self.config = config or VimVQConfig()
        self.key = key or random.PRNGKey(0)
        self.parent_router = parent_router
        
        # Inicializar quantizer
        self.quantizer = ViMVQQuantizer(self.config, self.key)
        
        # cache for modelos cuantizados
        self.quantized_models_cache = {}
        
    def should_quantize_layer(self, layer_name: str, layer_weights: jnp.ndarray) -> bool:
        """Determina if una capa debe be cuantizada
        
        Args:
            layer_name: Nombre de la capa
            layer_weights: Pesos de la capa
            
        Returns:
            bool: True if la capa debe be cuantizada
        """
        # not cuantizar capas pequeñas
        if np.prod(layer_weights.shape) < 1024:
            return False
            
        # not cuantizar capas de embedding
        if "embed" in layer_name.lower():
            return False
            
        # not cuantizar capas de normalización
        if any(x in layer_name.lower() for x in ["norm", "batch", "layer"]):
            return False
            
        return True
        
    def get_optimal_config(self, 
                          model_size: int,
                          device: str = "tpu") -> VimVQConfig:
        """Obtiene setup óptima basada en size del model
        
        Args:
            model_size: size del model en parámetros
            device: Dispositivo objetivo ('tpu' or 'arm')
            
        Returns:
            VimVQConfig: setup optimizada
        """
        if device == "tpu":
            # tpu v4-32 optimizations
            if model_size > 1e9:  # >1B params
                return VimVQConfig(
                    d=16,
                    k=512,
                    n_neighbors=8,
                    device="tpu"
                )
            else:
                return VimVQConfig(
                    d=8,
                    k=256,
                    n_neighbors=4,
                    device="tpu"
                )
        else:
            # ARM optimizations
            if model_size > 1e9:
                return VimVQConfig(
                    d=8,
                    k=256,
                    n_neighbors=4,
                    device="arm",
                    use_memory_pool=True,
                    cache_size_mb=2048
                )
            else:
                return VimVQConfig(
                    d=8,
                    k=128,
                    n_neighbors=4,
                    device="arm",
                    use_memory_pool=True,
                    cache_size_mb=1024
                )
    
    def route_model(self,
                   model_weights: Dict[str, jnp.ndarray],
                   model_config: Optional[Dict] = None) -> Dict[str, Dict]:
        """Rutea un model for cuantización
        
        Args:
            model_weights: Pesos del model
            model_config: setup optional
            
        Returns:
            Dict: Resultados de cuantización by capa
        """
        # calculate size del model
        total_params = sum(np.prod(w.shape) for w in model_weights.values())
        
        # determine dispositivo
        device = "tpu" if self.config.device == "tpu" else "arm"
        
        # obtain config óptima
        if not model_config or "vim_vq" not in model_config:
            self.config = self.get_optimal_config(total_params, device)
        
        # cache key
        cache_key = f"{total_params}_{device}"
        
        # verify cache
        if cache_key in self.quantized_models_cache:
            return self.quantized_models_cache[cache_key]
        
        # Cuantizar capas seleccionadas
        results = {}
        for layer_name, weights in model_weights.items():
            if self.should_quantize_layer(layer_name, weights):
                results[layer_name] = self.quantizer.quantize_layer(
                    weights,
                    layer_name=layer_name
                )
            else:
                results[layer_name] = {
                    'quantized_weights': weights,
                    'compression_ratio': 1.0,
                    'mse': 0.0
                }
        
        # update cache
        self.quantized_models_cache[cache_key] = results
        
        return results
    
    def get_compression_stats(self, results: Dict[str, Dict]) -> Dict:
        """Calcula estadísticas de compresión
        
        Args:
            results: Resultados de cuantización
            
        Returns:
            Dict: Estadísticas de compresión
        """
        total_params = 0
        total_compressed = 0
        total_mse = 0
        layers_quantized = 0
        
        for layer_name, layer_results in results.items():
            params = np.prod(layer_results['quantized_weights'].shape)
            total_params += params
            
            if layer_results['compression_ratio'] > 1.0:
                total_compressed += params / layer_results['compression_ratio']
                total_mse += layer_results['mse']
                layers_quantized += 1
            else:
                total_compressed += params
        
        return {
            'total_params': total_params,
            'compressed_params': total_compressed,
            'compression_ratio': total_params / total_compressed,
            'avg_mse': total_mse / layers_quantized if layers_quantized > 0 else 0.0,
            'layers_quantized': layers_quantized
        }
    
    def integrate_with_parent(self, parent_router: EnhancedRouter):
        """Integra with router padre
        
        Args:
            parent_router: Router padre for integration
        """
        self.parent_router = parent_router
        
        # Registrar callback for post-procesamiento
        if hasattr(parent_router, 'register_post_processor'):
            parent_router.register_post_processor(
                'vim_vq',
                self.route_model
            )
