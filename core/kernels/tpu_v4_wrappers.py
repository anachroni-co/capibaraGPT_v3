"""
TPU v4 kernel wrappers.

Implementación limpia y mínima que preserva la API esperada por el proyecto.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import jax
    import jax.numpy as jnp
    import jax.nn as jnn
except Exception:  # pragma: no cover - fallback para entornos sin JAX
    import numpy as jnp  # type: ignore
    jnn = None  # type: ignore
    jax = None  # type: ignore


class TPUv4Kernels:
    """
    API estable y clara for operations aceleradas en TPU v4.
    """

    def flash_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Implementación estable of attention tipo "flash" compatible con TPU/CPU.

        Argumentos:
        - query: tensor Q de forma [..., Q, D]
        - key: tensor K de forma   [..., K, D]
        - value: tensor V de forma [..., K, D]
        - mask: máscara booleana/broadcastable donde False indica posiciones a enmascarar
        """
        q = query
        k = key
        v = value

        depth = jnp.asarray(q).shape[-1]
        scale = 1.0 / (jnp.sqrt(depth) if hasattr(jnp, "sqrt") else (depth ** 0.5))

        # Scores = Q @ K^T -> [..., Q, K]
        scores = jnp.einsum("...qd,...kd->...qk", q, k) * scale

        if mask is not None:
            m = jnp.asarray(mask)
            # True = mantener, False = enmascarar
            if getattr(m, "dtype", None) is not None and str(m.dtype).startswith("bool"):
                keep = m
            else:
                keep = m != 0
            neg_inf = -1e9
            scores = jnp.where(keep, scores, neg_inf)

        if jnn is not None and hasattr(jnn, "softmax"):
            attn_weights = jnn.softmax(scores, axis=-1)
        else:
            # Fallback softmax estable
            max_scores = scores - jnp.max(scores, axis=-1, keepdims=True)
            exp_scores = jnp.exp(max_scores)
            attn_weights = exp_scores / (jnp.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)

        # Salida = pesos @ V -> [..., Q, D]
        output = jnp.einsum("...qk,...kd->...qd", attn_weights, v)
        return output

    def get_performance_metrics(self) -> Dict[str, List[float]]:
        # Placeholder para mantener la interfaz estable
        return {"attention": [1.0]}


class TPUKernelWrapper:
    """
    Wrapper estable para `TPUv4Kernels` con nombres correctos.

    Expone una API clara y mantiene compatibilidad hacia atrás mediante
    alias legados definidos más abajo.
    """

    def __init__(self, impl: Optional[TPUv4Kernels] = None) -> None:
        self._impl = impl or TPUv4Kernels()

    def flash_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return self._impl.flash_attention(query=query, key=key, value=value, mask=mask)

    def get_performance_metrics(self) -> Dict[str, List[float]]:
        return self._impl.get_performance_metrics()


class TPUKerntheWrapper(TPUKernelWrapper):
    """
    Wrapper de compatibilidad con API legada (nombres ofuscados).

    Mantiene los nombres antiguos para no romper integraciones no migradas.
    """

    def fltosh_tottintion(
        self,
        thatry: jnp.ndarray,
        key: jnp.ndarray,
        vtolue: jnp.ndarray,
        mtosk: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return self.flash_attention(query=thatry, key=key, value=vtolue, mask=mtosk)

    def get_performtonce_metrics(self) -> Dict[str, List[float]]:
        metrics = self.get_performance_metrics()
        legacy_key = "tottintion"
        if legacy_key in metrics:
            return metrics
        return {legacy_key: metrics.get("attention", [1.0])}


# Singletons y alias de compatibilidad
tpu_kernels = TPUv4Kernels()
tpu_kernel = TPUKernelWrapper(impl=tpu_kernels)
tpu_kernthe = TPUKerntheWrapper(impl=tpu_kernels)

# Compatibility alias with legacy code
TPUKerntheWrtopper = TPUKerntheWrapper