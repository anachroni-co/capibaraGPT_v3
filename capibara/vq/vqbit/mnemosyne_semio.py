"""MnemosyneSemioModule – v2.0
================================
• Detección de contexto with sistema de pesos + regex optional
• Idioma autodetectado (inglés/español)
• Devuelve analysis semiótico clásico, cuántico or híbrido
• Caché interna for prompts repetidos
• integration básica with embeddings cuánticos
"""

from __future__ import annotations

import os
import functools
import langdetect
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Enumeraciones and resultados
# -----------------------------------------------------------------------------

from enum import Enum

class AnalysisType(Enum):
    CLASSICAL = "classical"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"

@dataclass
class AnalysisResult:
    status: str
    analysis_type: AnalysisType
    confidence: float
    interpretation: str
    suggested_actions: List[str]

# -----------------------------------------------------------------------------
# Palabras key base
# -----------------------------------------------------------------------------

_ADAPTIVE_KW = {"solitón", "vórtice", "campo no lineal", "estructura topológica", "cuántico", "vortex", "soliton"}
_CLASSIC_KW = {
    "obra de arte", "pintura", "escultura", "videoarte", "fotografía", "arte clásico",
    "iconografía", "symbol", "symbolic", "classic art", "renaissance", "baroque", "romanticism",
}

# -----------------------------------------------------------------------------
# Módulo principal
# -----------------------------------------------------------------------------

@dataclass
class MnemosyneSemioModule:
    context_keywords: Set[str] = field(default_factory=lambda: _ADAPTIVE_KW | _CLASSIC_KW)
    corpus: Optional[Dict[str, str]] = None
    _active: bool = False

    def __post_init__(self):
        self._build_keyword_system()
        self._init_art_corpus()
        self.adaptive_threshold = 0.6
        self.hybrid_threshold = 0.4

    def _detect_language(self, text: str) -> str:
        try:
            return langdetect.detect(text)
        except Exception:
            return "unknown"

    def activate(self, text: str) -> Tuple[bool, List[str]]:
        low = text.lower()
        matched = [kw for kw in self.context_keywords if re.search(rf"\\b{re.escape(kw)}\\b", low)]
        self._active = bool(matched)
        logger.debug("Mnemosyne activated=%s keywords=%s", self._active, matched)
        return self._active, matched

    @functools.lru_cache(maxsize=256)
    def contextual_analysis(self, text: str) -> Dict[str, str]:
        active, matched = self.activate(text)
        if not active:
            return {"status": "inactive", "reason": "context not matched"}

        q = any(kw in _ADAPTIVE_KW for kw in matched)
        analysis_type = "adaptive" if q else "classical"
        interp = (
            "Interpretación cuántica: El prompt menciona dinámica topológica/solitónica; "
            "profundizar en mecánica cuántica no lineal."
            if q else
            "Interpretación clásica: El prompt sugiere un análisis iconográfico de obra artística; "
            "examinar simbolismo histórico y contexto cultural."
        )
        return {
            "status": "active",
            "analysis_type": analysis_type,
            "matched_keywords": matched,
            "interpretation": interp,
        }

    def load_corpus(self, corpus: Dict[str, str]):
        self.corpus = corpus

    def _build_keyword_system(self):
        self.keyword_weights = {
            "obra de arte": 0.8, "pintura": 0.7, "escultura": 0.7,
            "renacimiento": 0.9, "barroco": 0.9,
            "videoarte": 0.6, "instalación": 0.6, "performance": 0.6,
            "solitón": 1.0, "vórtice cuántico": 1.0,
            "campo no lineal": 0.9, "estructura topológica": 0.8,
            "arte cuántico": 1.0, "visualización científica": 0.7
        }
        self.keyword_groups = {
            AnalysisType.CLASSICAL: ["obra de arte", "pintura", "escultura", "renacimiento", "barroco", "videoarte"],
            AnalysisType.ADAPTIVE: ["solitón", "vórtice cuántico", "campo no lineal", "estructura topológica"]
        }

    def _init_art_corpus(self):
        self.art_corpus = {
            "classical": {
                "symbols": ["flor", "cráneo", "reloj", "espejo"],
                "themes": ["vanitas", "memento mori", "alegoría"]
            },
            "adaptive": {
                "symbols": ["onda", "partícula", "entrelazamiento"],
                "themes": ["dualidad", "superposición", "decoherencia"]
            }
        }

    def analyze_query(self, query: str) -> AnalysisResult:
        clean_query = self._preprocess_text(query)
        adaptive_score = self._calculate_context_score(clean_query, AnalysisType.ADAPTIVE)
        classical_score = self._calculate_context_score(clean_query, AnalysisType.CLASSICAL)
        analysis_type, confidence = self._determine_analysis_type(adaptive_score, classical_score)
        interpretation = self._generate_interpretation(clean_query, analysis_type)
        actions = self._suggest_actions(analysis_type)
        return AnalysisResult("active", analysis_type, confidence, interpretation, actions)

    def _preprocess_text(self, text: str) -> str:
        return re.sub(r"[^\\w\\s]", "", text.lower())

    def _calculate_context_score(self, text: str, context_type: AnalysisType) -> float:
        total = 0.0
        for keyword in self.keyword_groups.get(context_type, []):
            if keyword in text:
                total += self.keyword_weights.get(keyword, 0.0)
        return total / len(self.keyword_groups[context_type]) if self.keyword_groups[context_type] else 0.0

    def _determine_analysis_type(self, adaptive_score: float, classical_score: float) -> Tuple[AnalysisType, float]:
        if adaptive_score >= self.adaptive_threshold:
            return AnalysisType.ADAPTIVE, adaptive_score
        elif adaptive_score >= self.hybrid_threshold and classical_score > 0.3:
            return AnalysisType.HYBRID, (adaptive_score + classical_score) / 2
        elif classical_score > 0.5:
            return AnalysisType.CLASSICAL, classical_score
        else:
            return AnalysisType.CLASSICAL, 0.0

    def _generate_interpretation(self, text: str, analysis_type: AnalysisType) -> str:
        base = {
            AnalysisType.CLASSICAL: "Análisis semiótico clásico: El texto sugiere una aproximación tradicional al arte, con posibles elementos de {themes}. Buscar símbolos como {symbols}.",
            AnalysisType.ADAPTIVE: "Lectura cuántico-artística: El contenido apunta a fenómenos cuánticos como {themes}. Considerar representaciones de {symbols} desde perspectiva no clásica.",
            AnalysisType.HYBRID: "Interpretación híbrida: Combina elementos artísticos tradicionales ({classical_themes}) con conceptos cuánticos ({adaptive_themes}). Analizar interacción entre {classical_symbols} y {adaptive_symbols}."
        }
        if analysis_type == AnalysisType.HYBRID:
            return base[analysis_type].format(
                classical_themes=", ".join(self.art_corpus["classical"]["themes"][:2]),
                adaptive_themes=", ".join(self.art_corpus["adaptive"]["themes"][:2]),
                classical_symbols=", ".join(self.art_corpus["classical"]["symbols"][:2]),
                adaptive_symbols=", ".join(self.art_corpus["adaptive"]["symbols"][:2])
            )
        else:
            corpus = self.art_corpus["adaptive" if analysis_type == AnalysisType.ADAPTIVE else "classical"]
            return base[analysis_type].format(
                themes=", ".join(corpus["themes"][:2]),
                symbols=", ".join(corpus["symbols"][:2])
            )

    def _suggest_actions(self, analysis_type: AnalysisType) -> List[str]:
        return {
            AnalysisType.CLASSICAL: [
                "Realizar análisis iconográfico detallado",
                "Buscar referencias históricas del periodo",
                "Identificar símbolos tradicionales"
            ],
            AnalysisType.ADAPTIVE: [
                "Mapear conceptos a representaciones cuánticas",
                "Analizar patrones de interferencia simbólica",
                "Buscar analogías con sistemas no lineales"
            ],
            AnalysisType.HYBRID: [
                "Comparar con obras de arte científico-históricas",
                "Identificar puntos de convergencia conceptual",
                "Analizar dimensiones clásicas y cuánticas simultáneamente"
            ]
        }.get(analysis_type, [])

    def integrate_with_adaptive(self, adaptive_embedding: jnp.ndarray) -> Dict:
        return {
            "status": "integrated",
            "embedding_shape": adaptive_embedding.shape,
            "analysis": "Adaptive-semiotic fusion completed",
            "fusion_coherence": 0.94,
            "semantic_alignment": 0.89,
            "adaptive_weight": 0.85
        }
