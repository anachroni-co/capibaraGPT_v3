#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clasificador heurístico de tareas para seleccionar el modelo óptimo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ClassificationResult:
    """Resultado de la clasificación de un prompt."""

    model_tier: str
    scores: Dict[str, int]


class TaskClassifier:
    """Clasificador simple basado en palabras clave y longitud del prompt."""

    COMPLEX_KEYWORDS = (
        "análisis",
        "razonamiento",
        "comparación",
        "evaluar",
        "estrategia",
        "planificación",
        "investigación",
        "profundo",
        "detalle",
        "complejo",
        "técnico",
    )
    BALANCED_KEYWORDS = (
        "explicar",
        "qué es",
        "cómo funciona",
        "describir",
        "resumen",
        "breve",
        "ejemplo",
        "definir",
    )
    SIMPLE_KEYWORDS = (
        "qué",
        "quién",
        "cuál",
        "cuándo",
        "dónde",
        "chiste",
        "broma",
        "saludo",
        "ayuda",
    )

    ESTIMATED_TIMES_MS = {
        "fast_response": 2000,
        "balanced": 4000,
        "complex": 120000,
    }

    @classmethod
    def classify(cls, prompt: str) -> ClassificationResult:
        """Clasificar un prompt en 'fast_response', 'balanced' o 'complex'."""

        prompt_lower = prompt.lower()
        scores = {
            "complex": 0,
            "balanced": 0,
            "fast_response": 0,
        }

        for keyword in cls.COMPLEX_KEYWORDS:
            if keyword in prompt_lower:
                scores["complex"] += 2

        for keyword in cls.BALANCED_KEYWORDS:
            if keyword in prompt_lower:
                scores["balanced"] += 1

        for keyword in cls.SIMPLE_KEYWORDS:
            if keyword in prompt_lower:
                scores["fast_response"] += 1

        length = len(prompt)
        if length > 200:
            scores["complex"] += 1
        elif length > 100:
            scores["balanced"] += 1

        model_tier = max(scores, key=scores.get)
        return ClassificationResult(model_tier=model_tier, scores=scores)

    @classmethod
    def estimate_response_time(cls, model_tier: str) -> int:
        """Estimar el tiempo de respuesta esperado en milisegundos."""

        return cls.ESTIMATED_TIMES_MS.get(model_tier, cls.ESTIMATED_TIMES_MS["balanced"])


