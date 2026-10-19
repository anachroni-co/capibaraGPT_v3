""""AdaptiveSubmodel v3.0 – CapibaraGPT
=====================================
 Optimizado for tpu v4-32 with cache inteligente:
   • VQbit cache-aware integration
   • Adaptive encoding with sharding optimizado
   • Pipeline asíncrono for operaciones cuánticas
   • Métricas acumulativas eficientes
   • Memory layout optimizado for tpu matrix units
   • Prefetching inteligente de códigos cuánticos
"""

from __future__ import annotations

import os
import sys
import logging
from functools import partial
from typing import Any, Dict, Optional, Tuple

# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../CapibaraGPT v3
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    pass  # Using proper imports instead of sys.path manipulation

import jax
from flax import linen as nn
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from flax.linen.partitioning import with_sharding_constraint

# Imports with fallbacks
try:
    from capibara.vq.vqbit.wrapper import AdaptiveWrapper
except ImportError:
    logging.warning("AdaptiveWrapper unavailable; using fallback stub.")
    class AdaptiveWrapper:
        def __init__(self, config):
            self.config = config

try:
    from capibara.config.adaptive_config import AdaptiveConfig
except ImportError:
    logging.warning("AdaptiveConfig unavailable; using fallback stub.")
    class AdaptiveConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

try:
    from capibara.vq.vqbit.multimodal_vqbit import VQbitModule
except ImportError:
    logging.warning("VQbitModule unavailable; using fallback stub.")
    class VQbitModule(nn.Module):
        def __call__(self, x, **kwargs):
            return x

try:
    from capibara.sub_models.semiotic.mnemosyne_semio_module import MnemosyneSemioModule
except ImportError:
    logging.warning("MnemosyneSemioModule unavailable; using fallback stub.")
    class MnemosyneSemioModule(nn.Module):
        def __call__(self, x, **kwargs):
            return x

logger = logging.getLogger(__name__)

# tpu v4-32 optimized constants
ADAPTIVE_TILE_SIZE = 256  # Optimal for adaptive operations
PAULI_CACHE_SIZE = 64   # cache for operaciones Pauli frecuentes
METRIC_ACCUMULATION_WINDOW = 1000  # Ventana de acumulación de métricas

def _safe_mean_accumulative(sum_val: jnp.ndarray, count: jnp.ndarray) -> jnp.ndarray:
    """Mean secure with acumuladores."""
    return jnp.where(count > 0, sum_val / count, 0.0)

class AdaptiveSubmodel(nn.Module):
    """AdaptiveSubmodel optimizado for tpu v4-32.
    
    Mejoras principales:
    - cache-aware VQbit integration
    - Adaptive operations with sharding optimizado
    - Pipeline asíncrono for encoding cuántico
    - Métricas acumulativas without buffers circulares
    - Memory coalescing for operaciones Pauli
    - Prefetching de códigos cuánticos frecuentes
    """
    config: AdaptiveConfig

    def setup(self):
        # Core modules with setup optimizada for tpu
        self.vqbit = VQbitModule(
            config=self.config,
            adaptive_compatible=True,  # enable modo cuántico
            cache_size_ratio=0.4,     # cache more large for adaptive codes
            tile_size=ADAPTIVE_TILE_SIZE,
            enable_prefetch=True
        )
        
        self.wrapper = AdaptiveWrapper(config=self.config)
        
        # Projection layers with sharding hints
        self.dim_proj = nn.Dense(
            self.config.hidden_size, 
            use_bias=False,
            kernel_init=nn.initializers.variance_scaling(2.0, "fan_in", "normal")
        )
        
        # Layer norm optimizada for tpu
        self.norm = nn.LayerNorm(epsilon=1e-6, use_bias=False)
        
        # Adaptive number encoder with initialization cuántica
        self.qnum_encoder = nn.Dense(
            4, 
            kernel_init=nn.initializers.orthogonal(scale=1.0),
            bias_init=nn.initializers.zeros,
            name="qnum_encoder"
        )
        
        # Semiotic module
        self.semio = MnemosyneSemioModule()
        
        # Acumuladores de métricas eficientes (without buffers circulares)
        self._init_metric_accumulators()
        
        # cache for operaciones Pauli frecuentes
        self._init_pauli_cache()
        
        # Pre-compilar operaciones críticas
        self._setup_compiled_functions()
    
    def _init_metric_accumulators(self):
        """Inicializa acumuladores de métricas without buffers circulares."""
        # VQbit metrics accumulators
        self.vqbit_accum = {
            'perplexity_sum': self.variable("vqbit_metrics", "perplexity_sum", lambda: jnp.array(0.0)),
            'usage_sum': self.variable("vqbit_metrics", "usage_sum", lambda: jnp.array(0.0)),
            'commitment_sum': self.variable("vqbit_metrics", "commitment_sum", lambda: jnp.array(0.0)),
            'codebook_sum': self.variable("vqbit_metrics", "codebook_sum", lambda: jnp.array(0.0)),
            'count': self.variable("vqbit_metrics", "count", lambda: jnp.array(0, dtype=jnp.int32))
        }
        
        # Adaptive metrics accumulators
        self.adaptive_accum = {
            'coherence_sum': self.variable("adaptive_metrics", "coherence_sum", lambda: jnp.array(0.0)),
            'fidelity_sum': self.variable("adaptive_metrics", "fidelity_sum", lambda: jnp.array(0.0)),
            'entanglement_sum': self.variable("adaptive_metrics", "entanglement_sum", lambda: jnp.array(0.0)),
            'pauli_loss_sum': self.variable("adaptive_metrics", "pauli_loss_sum", lambda: jnp.array(0.0)),
            'count': self.variable("adaptive_metrics", "count", lambda: jnp.array(0, dtype=jnp.int32))
        }
        
        # Global step counter
        self.step_counter = self.variable("counters", "step", lambda: jnp.array(0, dtype=jnp.int32))
    
    def _init_pauli_cache(self):
        """Inicializa cache for operaciones Pauli frecuentes."""
        self.pauli_cache = {
            'frequent_qnums': self.variable(
                "pauli_cache", "frequent_qnums",
                lambda: jnp.zeros((PAULI_CACHE_SIZE, 4), dtype=jnp.float32)
            ),
            'cached_losses': self.variable(
                "pauli_cache", "cached_losses", 
                lambda: jnp.zeros(PAULI_CACHE_SIZE, dtype=jnp.float32)
            ),
            'access_counts': self.variable(
                "pauli_cache", "access_counts",
                lambda: jnp.zeros(PAULI_CACHE_SIZE, dtype=jnp.int32)
            ),
            'cache_valid': self.variable(
                "pauli_cache", "cache_valid",
                lambda: jnp.zeros(PAULI_CACHE_SIZE, dtype=jnp.bool_)
            )
        }
    
    def _setup_compiled_functions(self):
        """Pre-compila funciones críticas with optimizaciones tpu."""
        # Adaptive number encoding with sharding
        self._encode_qnums_compiled = jax.jit(
            self._encode_qnums_impl,
            donate_argnums=()
        )
        
        # Pauli loss with cache
        self._pauli_loss_cached = jax.jit(
            self._pauli_loss_cached_impl,
            static_argnames=("use_cache",)
        )
        
        # Modalidad concatenation optimizada
        self._concat_modalities_compiled = jax.jit(
            self._concat_modalities_impl,
            static_argnames=("modality_keys",)
        )
        
        # Spin modulation vectorizada
        self._apply_spin_modulation = jax.jit(
            self._apply_spin_modulation_impl
        )
    
    @partial(jax.jit, static_argnames=("self",))
    def _encode_qnums_impl(self, emb: jnp.ndarray) -> jnp.ndarray:
        """Encoding de embedding parameters optimizado for tpu."""
        # Apply sharding constraint for distribución óptima
        emb = with_sharding_constraint(emb, P("batch", None))
        
        # Adaptive encoding with restricciones físicas
        q_raw = self.qnum_encoder(emb)
        q_raw = with_sharding_constraint(q_raw, P("batch", None))
        
        # apply restricciones cuánticas en parallel
        n = jnp.clip(jnp.abs(q_raw[..., 0]) * 3 + 1, 1, self.config.max_adaptive_number)
        l = jnp.clip(jnp.round(jnp.abs(q_raw[..., 1])), 0, 3)
        
        # Restricción ml: -l <= ml <= l
        ml_raw = q_raw[..., 2] * l  # Scale by l
        ml = jnp.clip(jnp.round(ml_raw), -l, l)
        
        # Spin: ±1/2
        ms = jnp.sign(q_raw[..., 3]) * 0.5
        
        # Stack with memory layout optimizado
        qnums = jnp.stack([n, l, ml, ms], axis=-1)
        return with_sharding_constraint(qnums, P("batch", None))
    
    @partial(jax.jit, static_argnames=("self", "use_cache"))
    def _pauli_loss_cached_impl(self, emb: jnp.ndarray, use_cache: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Pauli loss with cache for configuraciones frecuentes."""
        qnums = self._encode_qnums_impl(emb)
        batch_size = qnums.shape[0]
        
        if use_cache and batch_size <= PAULI_CACHE_SIZE:
            # try use cache for configuraciones conocidas
            frequent_qnums = self.pauli_cache['frequent_qnums'].value
            cached_losses = self.pauli_cache['cached_losses'].value
            cache_valid = self.pauli_cache['cache_valid'].value
            
            # search matches en cache
            def compute_cache_distances():
                # Distancia L2 between qnums and cache
                diff = qnums[:, None, :] - frequent_qnums[None, :, :]  # (batch, cache, 4)
                distances = jnp.sum(diff ** 2, axis=-1)  # (batch, cache)
                return distances
            
            cache_distances = compute_cache_distances()
            cache_matches = jnp.argmin(cache_distances, axis=-1)
            match_distances = jnp.min(cache_distances, axis=-1)
            
            # Threshold for considerar un match valid
            match_threshold = 0.01
            is_cache_hit = (match_distances < match_threshold) & cache_valid[cache_matches]
            
            # Compute loss directamente for cache misses
            def compute_fresh_loss():
                # Tiled computation for efficiency
                def compute_tile_loss(start_idx):
                    end_idx = jnp.minimum(start_idx + ADAPTIVE_TILE_SIZE, batch_size)
                    tile_qnums = qnums[start_idx:end_idx]
                    
                    # Pairwise differences inside del tile
                    diff = tile_qnums[:, None, :] - tile_qnums[None, :, :]
                    distances = jnp.sum(jnp.abs(diff), axis=-1)
                    
                    # Exclusion diagonal with máscara
                    mask = jnp.eye(tile_qnums.shape[0])
                    masked_distances = distances + mask * 1e9
                    
                    # Stable softmax en log space
                    max_dist = jnp.max(-masked_distances)
                    log_weights = -masked_distances - max_dist
                    weights = jnp.exp(log_weights)
                    
                    return jnp.mean(weights) * jnp.exp(max_dist)
                
                # Process en tiles for memory efficiency
                num_tiles = (batch_size + ADAPTIVE_TILE_SIZE - 1) // ADAPTIVE_TILE_SIZE
                tile_starts = jnp.arange(0, batch_size, ADAPTIVE_TILE_SIZE)
                
                tile_losses = jax.vmap(compute_tile_loss)(tile_starts[:num_tiles])
                return jnp.mean(tile_losses) * self.config.pauli_loss_scale
            
            # use cache hits where sea posible
            cached_loss_values = cached_losses[cache_matches]
            fresh_loss = compute_fresh_loss()
            
            # combine resultados
            final_loss = jnp.where(
                jnp.any(is_cache_hit),
                jnp.mean(jnp.where(is_cache_hit, cached_loss_values, fresh_loss)),
                fresh_loss
            )
            
            return final_loss, qnums
        else:
            # Compute without cache for batches grandes
            diff = qnums[:, None, :] - qnums[None, :, :]
            distances = jnp.sum(jnp.abs(diff), axis=-1)
            
            mask = jnp.eye(batch_size) * 1e9
            masked_distances = distances + mask
            
            max_dist = jnp.max(-masked_distances)
            stable_exp = jnp.exp(-masked_distances - max_dist)
            loss = jnp.mean(stable_exp) * jnp.exp(max_dist) * self.config.pauli_loss_scale
            
            return loss, qnums
    
    def _update_pauli_cache(self, qnums: jnp.ndarray, loss: jnp.ndarray):
        """Actualiza cache de Pauli with configuraciones frecuentes."""
        # find setup more frecuente en batch
        unique_qnums, indices, counts = jnp.unique(
            qnums.reshape(-1, 4), axis=0, return_inverse=True, return_counts=True,
            size=min(len(qnums), PAULI_CACHE_SIZE)
        )
        
        # Update cache with LRU policy
        access_counts = self.pauli_cache['access_counts'].value
        
        # find slot except usado
        lru_slot = jnp.argmin(access_counts)
        
        # update if tenemos una setup suficientemente frecuente
        min_frequency = max(2, len(qnums) // 10)  # Al except 10% del batch
        most_frequent_idx = jnp.argmax(counts)
        
        if counts[most_frequent_idx] >= min_frequency:
            # Update cache entry
            self.pauli_cache['frequent_qnums'].value = \
                self.pauli_cache['frequent_qnums'].value.at[lru_slot].set(unique_qnums[most_frequent_idx])
            self.pauli_cache['cached_losses'].value = \
                self.pauli_cache['cached_losses'].value.at[lru_slot].set(loss)
            self.pauli_cache['cache_valid'].value = \
                self.pauli_cache['cache_valid'].value.at[lru_slot].set(True)
            self.pauli_cache['access_counts'].value = \
                self.pauli_cache['access_counts'].value.at[lru_slot].set(
                    access_counts[lru_slot] + counts[most_frequent_idx]
                )
    
    @partial(jax.jit, static_argnames=("self", "modality_keys"))
    def _concat_modalities_impl(self, xs: Dict[str, jnp.ndarray], modality_keys: Tuple[str, ...]) -> jnp.ndarray:
        """Concatenación optimizada de modalidades."""
        # Sort keys for determinismo
        sorted_items = [(k, xs[k]) for k in sorted(modality_keys)]
        
        # Reshape and concatenate with memory coalescing
        sequences = []
        for k, tensor in sorted_items:
            # Ensure optimal memory layout
            reshaped = tensor.reshape(-1, tensor.shape[-1])
            reshaped = with_sharding_constraint(reshaped, P("batch", None))
            sequences.append(reshaped)
        
        # Concatenate with optimal sharding
        result = jnp.concatenate(sequences, axis=0)
        return with_sharding_constraint(result, P("batch", None))
    
    @partial(jax.jit, static_argnames=("self",))
    def _apply_spin_modulation_impl(self, emb: jnp.ndarray, qnums: jnp.ndarray) -> jnp.ndarray:
        """apply spin modulation vectorizada."""
        spin_values = qnums[..., -1]  # ms values
        
        # Vectorized spin factors
        spin_up_factor = self.config.spin_up_factor
        spin_down_factor = self.config.spin_down_factor
        
        spin_factors = jnp.where(spin_values > 0, spin_up_factor, spin_down_factor)
        
        # Apply modulation with broadcasting optimizado
        modulated = emb * spin_factors[..., None]
        return with_sharding_constraint(modulated, P("batch", None))
    
    def _interpret_qnums(self, q: jnp.ndarray) -> Dict[str, Any]:
        """Interpretación de embedding parameters with caching."""
        orbital_names = ['s', 'p', 'd', 'f']
        n, l_idx, ml, ms = [q[..., i] for i in range(4)]
        l_idx = jnp.clip(l_idx.astype(int), 0, 3)
        
        return {
            "nivel": f"Nivel de energía {int(n[0])}",
            "orbital": f"Orbital {orbital_names[int(l_idx[0])]}{int(ml[0])}",
            "espín": "↑" if ms[0] > 0 else "↓",
            "configuracion_completa": {
                "n": float(n[0]),
                "l": int(l_idx[0]), 
                "ml": int(ml[0]),
                "ms": float(ms[0])
            }
        }
    
    def _update_metrics_accumulative(self, vq_metrics: Dict[str, Any], adaptive_metrics: Dict[str, Any]):
        """Actualiza métricas with acumuladores eficientes."""
        # Update VQbit metrics
        vq_accum = self.vqbit_accum
        vq_accum['perplexity_sum'].value += vq_metrics.get("perplexity", 0.0)
        vq_accum['usage_sum'].value += vq_metrics.get("usage", 0.0)
        vq_accum['commitment_sum'].value += vq_metrics.get("commitment_loss", 0.0)
        vq_accum['codebook_sum'].value += vq_metrics.get("codebook_loss", 0.0)
        vq_accum['count'].value += 1
        
        # Update adaptive metrics
        q_accum = self.adaptive_accum
        q_accum['coherence_sum'].value += adaptive_metrics.get("coherence", 0.0)
        q_accum['fidelity_sum'].value += adaptive_metrics.get("fidelity", 0.0)
        q_accum['entanglement_sum'].value += adaptive_metrics.get("entanglement", 0.0)
        q_accum['pauli_loss_sum'].value += adaptive_metrics.get("pauli_loss", 0.0)
        q_accum['count'].value += 1
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Obtiene métricas actuales since acumuladores."""
        # VQbit metrics
        vq_count = jnp.maximum(self.vqbit_accum['count'].value, 1)
        vqbit_metrics = {
            "perplexity": _safe_mean_accumulative(self.vqbit_accum['perplexity_sum'].value, vq_count),
            "usage": _safe_mean_accumulative(self.vqbit_accum['usage_sum'].value, vq_count),
            "commitment_loss": _safe_mean_accumulative(self.vqbit_accum['commitment_sum'].value, vq_count),
            "codebook_loss": _safe_mean_accumulative(self.vqbit_accum['codebook_sum'].value, vq_count),
            "cache_efficiency": float(jnp.mean(self.pauli_cache['access_counts'].value > 0))
        }
        
        # Adaptive metrics
        q_count = jnp.maximum(self.adaptive_accum['count'].value, 1)
        adaptive_metrics = {
            "coherence": _safe_mean_accumulative(self.adaptive_accum['coherence_sum'].value, q_count),
            "fidelity": _safe_mean_accumulative(self.adaptive_accum['fidelity_sum'].value, q_count),
            "entanglement": _safe_mean_accumulative(self.adaptive_accum['entanglement_sum'].value, q_count),
            "pauli_loss": _safe_mean_accumulative(self.adaptive_accum['pauli_loss_sum'].value, q_count),
            "pauli_cache_hits": float(jnp.sum(self.pauli_cache['cache_valid'].value))
        }
        
        return {
            "vqbit": vqbit_metrics,
            "adaptive": adaptive_metrics,
            "system": {
                "step": int(self.step_counter.value),
                "cache_utilization": float(jnp.mean(self.pauli_cache['access_counts'].value > 0)),
                "total_samples": int(vq_count)
            }
        }
    
    def reset_metrics(self):
        """Reset todas las métricas acumuladas."""
        # Reset VQbit accumulators
        for key in self.vqbit_accum:
            if key != 'count':
                self.vqbit_accum[key].value = 0.0
            else:
                self.vqbit_accum[key].value = 0
        
        # Reset adaptive accumulators
        for key in self.adaptive_accum:
            if key != 'count':
                self.adaptive_accum[key].value = 0.0
            else:
                self.adaptive_accum[key].value = 0
    
    def __call__(self, x: Dict[str, jnp.ndarray], *, training: bool = False, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Forward pass optimizado for tpu v4-32."""
        # Update step counter
        if training:
            self.step_counter.value += 1
        
        # Concatenación optimizada de modalidades
        modality_keys = tuple(sorted(x.keys()))
        flat_inp = self._concat_modalities_compiled(x, modality_keys)
        
        # VQbit processing with cache inteligente
        vq_out = jax.checkpoint(self.vqbit)(flat_inp, training=training)
        
        # Adaptive wrapper with pipeline optimizado
        fft_out = jax.checkpoint(self.wrapper)(vq_out["output"], deterministic=not training)
        
        # Dimension projection if es necessary
        if fft_out.shape[-1] != self.config.hidden_size:
            emb = self.dim_proj(fft_out)
        else:
            emb = fft_out
        
        # Layer normalization
        emb = self.norm(emb)
        emb = with_sharding_constraint(emb, P("batch", None))
        
        # Adaptive processing pipeline
        if training:
            # Pauli loss with cache
            pauli_loss, qnums = self._pauli_loss_cached(emb, use_cache=True)
            self.sow("intermediates", "loss.pauli", pauli_loss)
            
            # Update Pauli cache
            self._update_pauli_cache(qnums, pauli_loss)
        else:
            # only encoding without loss computation
            qnums = self._encode_qnums_compiled(emb)
            pauli_loss = jnp.array(0.0)
        
        # Spin modulation
        emb_modulated = self._apply_spin_modulation(emb, qnums)
        
        # Metrics update
        if training:
            adaptive_metrics = self.wrapper.get_metrics()
            adaptive_metrics["pauli_loss"] = float(pauli_loss)
            self._update_metrics_accumulative(vq_out["metrics"], adaptive_metrics)
        
        # Semiotic processing condicional
        semio_output = None
        if prompt and "cuántico" in prompt.lower():
            semio_output = {
                "adaptive_numbers": qnums,
                "interpretation": self._interpret_qnums(qnums),
                "cache_info": {
                    "pauli_cache_size": PAULI_CACHE_SIZE,
                    "cache_hits": int(jnp.sum(self.pauli_cache['cache_valid'].value)),
                    "adaptive_tile_size": ADAPTIVE_TILE_SIZE
                }
            }
        
        return {
            "output": emb_modulated,
            "adaptive_numbers": qnums,
            "semio": semio_output,
            "metrics": self.get_metrics() if training else {},
            "cache_info": {
                "vqbit_cache": vq_out.get("cache_info", {}),
                "pauli_cache_efficiency": float(jnp.mean(self.pauli_cache['access_counts'].value > 0))
            }
        }
