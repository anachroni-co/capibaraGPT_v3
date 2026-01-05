#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cliente HTTP para interactuar con vLLM con selección de modelo y fallback (ajustado para compatibilidad)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, Optional

import requests  # type: ignore[import-untyped]
from openai import OpenAI

from task_classifier import TaskClassifier

logger = logging.getLogger(__name__)


class VLLMClient:
    """Cliente para gestionar peticiones a vLLM con soporte de fallback (ajustado para compatibilidad)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.endpoint = config.get("api_settings", {}).get(
            "vllm_endpoint",
            config.get("api_settings", {}).get("ollama_endpoint", "http://localhost:8000/v1"),
        )
        self.models = config.get("models", {})

        fallback_cfg = config.get("fallback_strategy", {})
        self.fallback_enabled = fallback_cfg.get(
            "enabled", os.getenv("FALLBACK_ENABLED", "true").lower() == "true"
        )
        self.fallback_order = fallback_cfg.get(
            "order",
            ["fast_response", "balanced", "complex"],
        )

        self.default_tier = os.getenv(
            "DEFAULT_MODEL_TIER",
            config.get("api_settings", {}).get("default_model", "fast_response"),
        )

        # Initialize OpenAI client with vLLM endpoint
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key="EMPTY"  # vLLM typically doesn't require an API key in basic setup
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, model_tier: str, **options: Any) -> Dict[str, Any]:
        """Generar una respuesta usando el modelo asignado a `model_tier`."""
        model_cfg = self.models.get(model_tier)
        if not model_cfg:
            raise ValueError(f"Modelo no configurado para tier: {model_tier}")

        try:
            # Prepare the chat completion request with only vLLM-compatible parameters
            # Removed top_k which isn't supported by vLLM
            request_params = {
                "model": model_cfg["name"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": options.get("temperature", 0.7),
                "max_tokens": min(options.get("max_tokens", model_cfg.get("max_tokens", 512)), 2048),
                "top_p": options.get("top_p", 0.9),
            }
            
            # Only add presence_penalty and frequency_penalty if they exist and are supported
            if "presence_penalty" in options:
                request_params["presence_penalty"] = options["presence_penalty"]
            if "frequency_penalty" in options:
                request_params["frequency_penalty"] = options["frequency_penalty"]

            response = self.client.chat.completions.create(**request_params)

            # Extract the response and metadata
            message_content = response.choices[0].message.content
            usage = response.usage

            return {
                "success": True,
                "model": model_cfg["name"],
                "response": message_content or "",
                "total_duration": getattr(response, 'response_ms', None),
                "token_count": usage.completion_tokens if usage else None,
            }
        except Exception as e:
            logger.error(f"Error con modelo {model_cfg['name']}: {e}")
            return {
                "success": False,
                "model": model_cfg["name"],
                "error": str(e),
            }

    def generate_with_fallback(self, prompt: str, model_tier: Optional[str] = None, **options: Any) -> Dict[str, Any]:
        """Generar respuesta con fallback según configuración."""

        selected_tier = model_tier or self.default_tier or "fast_response"
        tiers_to_try = self._resolve_tiers(prompt, selected_tier)
        last_error = None

        for tier in tiers_to_try:
            try:
                return self.generate(prompt, tier, **options)
            except Exception as exc:  # noqa: BLE001 - Propagamos último error
                logger.warning("Error con modelo %s: %s", tier, exc)
                last_error = exc
                continue

        return {
            "success": False,
            "model": None,
            "error": str(last_error) if last_error else "No se pudo generar respuesta",
        }

    def stream_with_model(self, prompt: str, model_tier: str, **options: Any) -> Iterable[str]:
        """Generar texto en streaming usando un modelo específico."""

        model_cfg = self.models.get(model_tier)
        if not model_cfg:
            raise ValueError(f"Modelo no configurado para tier: {model_tier}")

        try:
            # Prepare the chat completion request with streaming and only compatible parameters
            request_params = {
                "model": model_cfg["name"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": options.get("temperature", 0.7),
                "max_tokens": min(options.get("max_tokens", model_cfg.get("max_tokens", 512)), 2048),
                "top_p": options.get("top_p", 0.9),
                "stream": True,
            }
            
            # Only add presence_penalty and frequency_penalty if they exist and are supported
            if "presence_penalty" in options:
                request_params["presence_penalty"] = options["presence_penalty"]
            if "frequency_penalty" in options:
                request_params["frequency_penalty"] = options["frequency_penalty"]

            stream = self.client.chat.completions.create(**request_params)

            # Yield each chunk as it arrives
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error en streaming con modelo {model_cfg['name']}: {e}")
            raise

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_timeout(model_cfg: Dict[str, Any], options: Dict[str, Any]) -> float:
        timeout_ms = options.get("timeout") or model_cfg.get("timeout", 10000)
        return float(timeout_ms) / 1000.0

    def _resolve_tiers(self, prompt: str, preferred_tier: str) -> Iterable[str]:
        if preferred_tier == "auto":
            classification = TaskClassifier.classify(prompt)
            preferred_tier = classification.model_tier

        tiers = [preferred_tier]

        if self.fallback_enabled:
            tiers.extend([tier for tier in self.fallback_order if tier not in tiers])

        return tiers