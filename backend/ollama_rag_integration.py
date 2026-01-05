#!/usr/bin/env python3
"""
Integración entre Ollama y sistema RAG con optimización TOON

Este módulo permite que las respuestas de Ollama sean enriquecidas
con información del sistema RAG cuando se detecte que la consulta
requiere datos personales del usuario, usando TOON para reducir tokens.
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple
from ollama_client import VLLMClient  # Changed from OllamaClient to VLLMClient
from rag_client import RAGClient

logger = logging.getLogger(__name__)


class OllamaRAGIntegration:
    """Integración entre Ollama y RAG con optimización TOON automática"""

    # Patrones que indican necesidad de consultar RAG
    RAG_TRIGGERS = [
        r"\b(mi|mis|yo|he|tengo|recuerdo)\b",  # Referencias personales
        r"\b(dije|hablé|comenté|mencioné)\b",  # Referencias a conversaciones
        r"\b(guardé|guardado|archivé)\b",  # Referencias a archivos
        r"\b(conversación|chat|mensaje)\b",  # Conversaciones
        r"\b(archivo|documento|pdf|imagen)\b",  # Archivos
        r"\b(contacto|persona|amigo)\b",  # Relaciones
        r"qué (tengo|hay|dijiste|dije)",  # Preguntas sobre datos
        r"(muéstrame|busca|encuentra) (mi|mis)",  # Búsquedas personales
    ]

    def __init__(
        self,
        vllm_client: Optional[VLLMClient] = None,
        rag_client: Optional[RAGClient] = None,
        rag_threshold: float = 0.3,  # Score mínimo para usar RAG
        context_max_length: int = 1500,
        enable_toon: bool = True,  # Nuevo: habilitar TOON
    ):
        """
        Inicializar integración vLLM-RAG con soporte TOON

        Args:
            vllm_client: Cliente vLLM configurado
            rag_client: Cliente RAG configurado
            rag_threshold: Umbral para decidir si usar RAG
            context_max_length: Longitud máxima del contexto RAG
            enable_toon: Habilitar optimización TOON para reducir tokens
        """
        self.vllm_client = vllm_client
        self.rag_client = rag_client or RAGClient(enable_toon=enable_toon)
        self.rag_threshold = rag_threshold
        self.context_max_length = context_max_length
        self.enable_toon = enable_toon

        logger.info(f"OllamaRAGIntegration initialized (toon: {enable_toon})")

    def should_use_rag(self, query: str) -> Tuple[bool, float]:
        """
        Determinar si una consulta requiere información del RAG

        Args:
            query: Consulta del usuario

        Returns:
            Tuple (should_use, confidence_score)
        """
        query_lower = query.lower()
        matches = 0

        for pattern in self.RAG_TRIGGERS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                matches += 1

        # Score basado en número de coincidencias
        score = min(matches * 0.2, 1.0)

        should_use = score >= self.rag_threshold

        logger.debug(f"RAG decision for '{query[:50]}...': {should_use} (score: {score:.2f})")

        return should_use, score

    def enrich_prompt_with_rag(
        self,
        user_prompt: str,
        n_results: int = 3,
        use_toon: Optional[bool] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Enriquecer prompt con contexto del RAG (con TOON si es beneficioso)

        Args:
            user_prompt: Prompt original del usuario
            n_results: Número de resultados RAG a incluir
            use_toon: Forzar TOON (None=auto, True=forzar, False=desactivar)

        Returns:
            Tuple (enriched_prompt, rag_data)
        """
        should_use, confidence = self.should_use_rag(user_prompt)

        if not should_use:
            logger.info("RAG not needed for this query")
            return user_prompt, None

        # Obtener contexto del RAG con TOON automático
        rag_context, metadata = self.rag_client.get_context_for_llm(
            user_query=user_prompt,
            max_context_length=self.context_max_length,
            n_results=n_results,
            use_toon=use_toon if use_toon is not None else None  # Auto-detect
        )

        if not rag_context:
            logger.warning("RAG query returned empty context")
            return user_prompt, None

        # Construir prompt enriquecido
        # Si se usó TOON, el contexto ya incluye una nota explicativa
        enriched_prompt = f"""{rag_context}

---

Basándote en la información anterior, responde a la siguiente pregunta:
{user_prompt}

Si el contexto contiene información relevante, úsala en tu respuesta.
Si no hay información relevante en el contexto, responde normalmente."""

        rag_data = {
            "used_rag": True,
            "confidence": confidence,
            "context_length": len(rag_context),
            "format_used": metadata.get('format_used', 'text'),
            "sources_count": metadata.get('sources_count', 0)
        }

        # Agregar métricas de TOON si está disponible
        if metadata.get('savings_percent'):
            rag_data['toon_savings_percent'] = metadata['savings_percent']
            rag_data['original_size'] = metadata.get('original_size', 0)
            rag_data['optimized_size'] = metadata.get('formatted_size', 0)

        logger.info(
            f"Prompt enriched with RAG "
            f"(format: {rag_data['format_used']}, "
            f"sources: {rag_data['sources_count']}, "
            f"savings: {rag_data.get('toon_savings_percent', 0):.1f}%)"
        )

        return enriched_prompt, rag_data

    def generate_with_rag(
        self,
        prompt: str,
        model_tier: Optional[str] = None,
        use_rag: bool = True,
        use_toon: Optional[bool] = None,
        **vllm_options: Any
    ) -> Dict[str, Any]:
        """
        Generar respuesta con vLLM, usando RAG+TOON si es necesario

        Args:
            prompt: Prompt del usuario
            model_tier: Tier del modelo vLLM (fast_response, balanced, complex)
            use_rag: Si permitir el uso de RAG
            use_toon: Forzar uso de TOON (None=auto, True=forzar, False=desactivar)
            **vllm_options: Opciones adicionales para vLLM

        Returns:
            Dict con respuesta y metadatos incluyendo info de TOON
        """
        if not self.vllm_client:
            raise ValueError("VLLMClient no configurado")

        rag_data = None

        # Enriquecer con RAG si está habilitado
        if use_rag:
            enriched_prompt, rag_data = self.enrich_prompt_with_rag(
                prompt,
                use_toon=use_toon
            )
        else:
            enriched_prompt = prompt

        # Generar respuesta con vLLM
        vllm_response = self.vllm_client.generate_with_fallback(
            prompt=enriched_prompt,
            model_tier=model_tier,
            **vllm_options
        )

        # Combinar respuesta con metadata RAG y TOON
        response = {
            "response": vllm_response.get("response", ""),
            "success": vllm_response.get("success", False),
            "model": vllm_response.get("model"),
            "rag_used": rag_data is not None,
        }

        if rag_data:
            response["rag_metadata"] = rag_data

            # Log detallado si se usó TOON
            if rag_data.get('format_used') == 'toon':
                logger.info(
                    f"TOON optimization: {rag_data.get('toon_savings_percent', 0):.1f}% tokens saved "
                    f"({rag_data.get('original_size', 0)} → {rag_data.get('optimized_size', 0)} chars)"
                )

        if "error" in vllm_response:
            response["error"] = vllm_response["error"]

        return response

    def stream_with_rag(
        self,
        prompt: str,
        model_tier: str,
        use_rag: bool = True,
        use_toon: Optional[bool] = None,
        **vllm_options: Any
    ):
        """
        Generar respuesta en streaming con vLLM, RAG y TOON

        Args:
            prompt: Prompt del usuario
            model_tier: Tier del modelo
            use_rag: Si usar RAG
            use_toon: Forzar uso de TOON (None=auto)
            **vllm_options: Opciones adicionales

        Yields:
            Chunks de texto
        """
        if not self.vllm_client:
            raise ValueError("VLLMClient no configurado")

        # Enriquecer con RAG si está habilitado
        if use_rag:
            enriched_prompt, rag_data = self.enrich_prompt_with_rag(
                prompt,
                use_toon=use_toon
            )
            # Yield metadata inicial si se usó RAG
            if rag_data:
                format_info = f"[RAG: {rag_data['confidence']:.2f} confidence"
                if rag_data.get('format_used') == 'toon':
                    format_info += f", TOON: {rag_data.get('toon_savings_percent', 0):.0f}% saved"
                format_info += "]\n\n"
                yield format_info
        else:
            enriched_prompt = prompt

        # Stream desde vLLM
        for chunk in self.vllm_client.stream_with_model(
            prompt=enriched_prompt,
            model_tier=model_tier,
            **vllm_options
        ):
            yield chunk


def create_integrated_client(
    vllm_config: Dict[str, Any],
    rag_url: Optional[str] = None,
    enable_toon: bool = True
) -> OllamaRAGIntegration:
    """
    Factory para crear cliente integrado vLLM-RAG con TOON

    Args:
        vllm_config: Configuración de vLLM (model_config.json)
        rag_url: URL del servidor RAG (opcional)
        enable_toon: Habilitar optimización TOON

    Returns:
        Cliente integrado configurado con soporte TOON
    """
    vllm_client = VLLMClient(vllm_config)
    rag_client = RAGClient(base_url=rag_url, enable_toon=enable_toon)

    return OllamaRAGIntegration(
        vllm_client=vllm_client,
        rag_client=rag_client,
        enable_toon=enable_toon
    )


if __name__ == "__main__":
    # Demo de uso con TOON
    import json
    logging.basicConfig(level=logging.INFO)

    # Cargar configuración de Ollama
    with open("/home/elect/capibara6/model_config.json") as f:
        ollama_config = json.load(f)

    # Crear cliente integrado con TOON habilitado
    integrated_client = create_integrated_client(
        vllm_config=ollama_config,  # The config is compatible with both
        enable_toon=True
    )

    # Test: consulta que NO debería usar RAG
    print("=== Test 1: Pregunta general (sin RAG) ===")
    response1 = integrated_client.generate_with_rag(
        prompt="¿Qué es machine learning?",
        model_tier="fast_response"
    )
    print(f"RAG usado: {response1['rag_used']}")
    print(f"Respuesta: {response1['response'][:100]}...")

    # Test: consulta que SÍ debería usar RAG con TOON
    print("\n=== Test 2: Pregunta personal (con RAG+TOON) ===")
    response2 = integrated_client.generate_with_rag(
        prompt="¿Qué he comentado sobre machine learning en mis conversaciones?",
        model_tier="balanced",
        use_toon=None  # Auto-detect
    )
    print(f"RAG usado: {response2['rag_used']}")
    if response2.get('rag_metadata'):
        meta = response2['rag_metadata']
        print(f"RAG confidence: {meta['confidence']:.2f}")
        print(f"Formato usado: {meta['format_used']}")
        if meta.get('toon_savings_percent'):
            print(f"TOON ahorro: {meta['toon_savings_percent']:.1f}%")
            print(f"Tokens: {meta['original_size']} → {meta['optimized_size']}")
    print(f"Respuesta: {response2['response'][:100]}...")
