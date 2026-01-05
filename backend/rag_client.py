#!/usr/bin/env python3
"""
Cliente para consultar el sistema RAG desde otras VMs (ej: bounty2)
con soporte TOON para optimización de tokens

Este cliente permite a los servidores con Ollama consultar el sistema RAG
en RAG3 para enriquecer respuestas con datos personales del usuario,
usando TOON para reducir el consumo de tokens en 30-60%.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class RAGClient:
    """Cliente HTTP para consultar el sistema RAG en VM RAG3 con soporte TOON"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        enable_toon: bool = True
    ):
        """
        Inicializar cliente RAG

        Args:
            base_url: URL base del servidor RAG (default: desde env o IP interna)
            timeout: Timeout para requests en segundos
            max_retries: Número máximo de reintentos
            enable_toon: Habilitar optimización TOON para reducir tokens
        """
        # IP interna de RAG3 en GCloud (actualizar según vm_config.json)
        # Para obtener la IP correcta, ejecuta: python3 scripts/get_vm_info.py
        default_rag_url = os.getenv(
            "RAG3_IP_INTERNAL",
            os.getenv("RAG_API_URL", "http://10.154.0.2:8000")  # IP interna por defecto
        )
        self.base_url = base_url or default_rag_url
        self.timeout = timeout
        self.enable_toon = enable_toon

        # Intentar importar TOON si está habilitado
        self.toon_available = False
        if self.enable_toon:
            try:
                from toon_utils.format_manager import FormatManager
                self.FormatManager = FormatManager
                self.toon_available = True
                logger.info("TOON habilitado para optimización de tokens")
            except ImportError:
                logger.warning("TOON no disponible, usando JSON estándar")
                self.enable_toon = False

        # Configurar session con retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(f"RAGClient initialized (base_url: {self.base_url}, toon: {self.enable_toon})")

    def search_semantic(
        self,
        query: str,
        collection_name: Optional[str] = None,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Búsqueda semántica en colecciones vectoriales

        Args:
            query: Texto de búsqueda
            collection_name: Colección específica o None para buscar en todas
            n_results: Número de resultados a retornar

        Returns:
            Dict con resultados de búsqueda
        """
        endpoint = f"{self.base_url}/api/search/semantic"
        payload = {
            "query": query,
            "collection_name": collection_name,
            "n_results": n_results
        }

        try:
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            return {
                "query": query,
                "total_results": 0,
                "results": [],
                "error": str(e)
            }

    def search_rag(
        self,
        query: str,
        n_results: int = 5,
        use_graph: bool = True
    ) -> Dict[str, Any]:
        """
        Búsqueda RAG completa (Vector + PostgreSQL + Grafo)

        Esta es la búsqueda más completa que combina:
        - Búsqueda vectorial semántica
        - Enriquecimiento con datos de PostgreSQL
        - Exploración de grafo de conocimiento en Nebula

        Args:
            query: Pregunta en lenguaje natural
            n_results: Resultados por colección
            use_graph: Si usar exploración de grafo

        Returns:
            Dict con contexto enriquecido del RAG
        """
        endpoint = f"{self.base_url}/api/search/rag"
        payload = {
            "query": query,
            "n_results": n_results,
            "use_graph": use_graph
        }

        try:
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error en búsqueda RAG: {e}")
            return {
                "query": query,
                "context": "",
                "sources": [],
                "error": str(e)
            }

    def search_all_collections(
        self,
        query: str,
        n_results: int = 3
    ) -> Dict[str, Any]:
        """
        Buscar en todas las colecciones simultáneamente

        Args:
            query: Texto de búsqueda
            n_results: Resultados por colección

        Returns:
            Dict con resultados por colección
        """
        endpoint = f"{self.base_url}/api/search/all"
        payload = {
            "query": query,
            "n_results": n_results
        }

        try:
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error en búsqueda multi-colección: {e}")
            return {
                "query": query,
                "total_results": 0,
                "collections": [],
                "results": {},
                "error": str(e)
            }

    def health_check(self) -> Dict[str, Any]:
        """
        Verificar estado del servicio RAG

        Returns:
            Dict con estado de servicios
        """
        endpoint = f"{self.base_url}/health"

        try:
            response = self.session.get(endpoint, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error en health check: {e}")
            return {
                "status": "unavailable",
                "error": str(e)
            }

    def get_context_for_llm(
        self,
        user_query: str,
        max_context_length: int = 2000,
        n_results: int = 3,
        use_toon: Optional[bool] = None,
        format_output: str = "auto"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Obtener contexto formateado para agregar al prompt de un LLM
        con optimización TOON automática

        Esta función es útil para enriquecer las respuestas de Ollama
        con información personal del usuario, usando TOON cuando es beneficioso.

        Args:
            user_query: Consulta del usuario
            max_context_length: Longitud máxima del contexto en caracteres
            n_results: Número de resultados a incluir
            use_toon: Forzar uso de TOON (None=auto, True=siempre, False=nunca)
            format_output: Formato de salida ('auto', 'toon', 'json', 'text')

        Returns:
            Tuple (contexto_formateado, metadata)
            metadata incluye: {
                'format_used': 'toon' | 'json' | 'text',
                'original_size': int,
                'formatted_size': int,
                'savings_percent': float,
                'sources_count': int
            }
        """
        # Obtener resultados del RAG
        rag_result = self.search_rag(
            query=user_query,
            n_results=n_results,
            use_graph=True
        )

        if "error" in rag_result:
            return "", {
                "format_used": "none",
                "error": rag_result["error"],
                "sources_count": 0
            }

        sources = rag_result.get("sources", [])

        if not sources:
            return "", {
                "format_used": "none",
                "sources_count": 0,
                "message": "No se encontraron fuentes relevantes"
            }

        # Determinar si usar TOON
        should_use_toon = self._should_use_toon(
            sources=sources,
            use_toon=use_toon,
            format_output=format_output
        )

        # Formatear contexto
        if should_use_toon:
            context, metadata = self._format_with_toon(sources, max_context_length)
        else:
            context, metadata = self._format_without_toon(sources, max_context_length)

        metadata['sources_count'] = len(sources)

        return context, metadata

    def _should_use_toon(
        self,
        sources: List[Dict],
        use_toon: Optional[bool],
        format_output: str
    ) -> bool:
        """Determinar si se debe usar TOON para formatear"""

        # Si se fuerza explícitamente
        if use_toon is True:
            return self.toon_available
        if use_toon is False:
            return False
        if format_output == 'toon':
            return self.toon_available
        if format_output == 'json' or format_output == 'text':
            return False

        # Auto-detección
        if not self.toon_available:
            return False

        # Usar TOON si hay 5+ fuentes con estructura similar
        if len(sources) >= 5:
            return True

        # Analizar si TOON sería beneficioso
        try:
            data_to_analyze = {"sources": sources}
            stats = self.FormatManager.analyze_data(data_to_analyze)

            # Usar TOON si ahorra al menos 25%
            if stats.get('toon_recommended') and stats.get('savings_percent', 0) >= 25:
                logger.info(f"TOON seleccionado: ahorro estimado {stats['savings_percent']:.1f}%")
                return True
        except Exception as e:
            logger.debug(f"Error en análisis TOON: {e}")

        return False

    def _format_with_toon(
        self,
        sources: List[Dict],
        max_length: int
    ) -> Tuple[str, Dict[str, Any]]:
        """Formatear contexto usando TOON"""

        try:
            # Preparar datos para TOON
            data_to_encode = {"sources": sources}

            # Codificar con TOON
            toon_content, format_type = self.FormatManager.encode(
                data_to_encode,
                preferred_format='toon'
            )

            # Calcular ahorro
            import json
            json_content = json.dumps(data_to_encode, ensure_ascii=False)
            original_size = len(json_content)
            toon_size = len(toon_content)
            savings = ((original_size - toon_size) / original_size * 100) if original_size > 0 else 0

            # Construir contexto formateado
            context = f"""Información relevante del usuario (formato TOON):

{toon_content[:max_length]}

Nota: Formato TOON - estructura optimizada para LLMs"""

            metadata = {
                'format_used': 'toon',
                'original_size': original_size,
                'formatted_size': toon_size,
                'savings_percent': round(savings, 1)
            }

            logger.info(f"Contexto formateado con TOON: {savings:.1f}% de ahorro ({original_size} → {toon_size} chars)")

            return context, metadata

        except Exception as e:
            logger.error(f"Error al formatear con TOON, fallback a texto: {e}")
            return self._format_without_toon(sources, max_length)

    def _format_without_toon(
        self,
        sources: List[Dict],
        max_length: int
    ) -> Tuple[str, Dict[str, Any]]:
        """Formatear contexto sin TOON (texto plano)"""

        context_parts = ["Información relevante del usuario:\n"]

        for i, source in enumerate(sources[:10], 1):  # Máximo 10 fuentes
            source_text = source.get("content", source.get("text", ""))
            similarity = source.get("similarity", 0)

            if source_text:
                # Truncar si es muy largo
                if len(source_text) > 300:
                    source_text = source_text[:300] + "..."

                context_parts.append(
                    f"{i}. [{similarity:.2f}] {source_text}"
                )

        full_context = "\n".join(context_parts)

        # Truncar si excede límite
        if len(full_context) > max_length:
            full_context = full_context[:max_length] + "..."

        metadata = {
            'format_used': 'text',
            'original_size': len(full_context),
            'formatted_size': len(full_context),
            'savings_percent': 0
        }

        return full_context, metadata


# Función helper para uso rápido
def get_rag_context(query: str, rag_url: Optional[str] = None, use_toon: bool = True) -> Tuple[str, Dict]:
    """
    Función helper para obtener rápidamente contexto RAG con TOON

    Args:
        query: Consulta del usuario
        rag_url: URL del servidor RAG (opcional)
        use_toon: Usar optimización TOON

    Returns:
        Tuple (contexto, metadata)

    Uso:
        context, meta = get_rag_context("¿Qué he hablado sobre IA?")
        print(f"Formato usado: {meta['format_used']}")
        print(f"Ahorro: {meta.get('savings_percent', 0)}%")

        full_prompt = f"{context}\n\nUsuario: {user_query}"
    """
    client = RAGClient(base_url=rag_url, enable_toon=use_toon)
    return client.get_context_for_llm(query)


if __name__ == "__main__":
    # Demo de uso
    import sys
    logging.basicConfig(level=logging.INFO)

    client = RAGClient(enable_toon=True)

    # Test health
    print("=== Health Check ===")
    health = client.health_check()
    print(f"Status: {health.get('status')}")

    # Test búsqueda
    print("\n=== Búsqueda RAG ===")
    result = client.search_rag("machine learning", n_results=5)
    print(f"Query: {result.get('query')}")
    print(f"Sources: {len(result.get('sources', []))}")

    # Test contexto con TOON
    print("\n=== Contexto para LLM con TOON ===")
    context, metadata = client.get_context_for_llm(
        "machine learning y embeddings",
        n_results=5,
        use_toon=None  # Auto
    )

    print(f"Formato usado: {metadata['format_used']}")
    if metadata.get('savings_percent'):
        print(f"Ahorro de tokens: {metadata['savings_percent']}%")
        print(f"Tamaño: {metadata['original_size']} → {metadata['formatted_size']} chars")
    print(f"\nContexto (primeros 300 chars):")
    print(context[:300] + "..." if len(context) > 300 else context)
