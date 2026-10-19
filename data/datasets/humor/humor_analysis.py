"""
Humor Analysis Tools for CapibaraGPT v3
=======================================

Tools for analyzing and categorizing humor in Spanish.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class HumorType(Enum):
    """Identifiable humor types."""
    JUEGO_PALABRAS = "juego_palabras"
    HUMOR_NEGRO = "humor_negro"
    COMPARACION = "comparacion"
    REGLA_TRES = "regla_tres"
    ANIMACION = "animacion"
    GENERAL = "general"
    IRONIA = "ironia"
    SARCASMO = "sarcasmo"

@dataclass
class HumorAnalysis:
    """Humor analysis result."""
    text: str
    humor_type: HumorType
    confidence: float
    features: Dict[str, Any]
    explanation: Optional[str] = None

class HumorAnalyzer:
    """Humor type analyzer for Spanish jokes."""
    
    def __init__(self):
        self.patterns = self._load_humor_patterns()
        self.indicators = self._load_humor_indicators()
    
    def _load_humor_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns to detect humor types."""
        return {
            'juego_palabras': [
                r'\b(\w+)\b.*\b\1\w*\b',  # Repetition of similar words
                r'¿[^?]*\?.*¡[^!]*!',      # Question followed by exclamation
                r'\b(suena?|parece|dice)\s+(como|que)',  # Sound/similarity indicators
                r'\b(dos|tres)\s+\w+\s+(van|entran|salen)',  # Typical joke setup
            ],
            'humor_negro': [
                r'\b(muerte|muerto|morir|funeral|cementerio)\b',
                r'\b(enfermedad|cáncer|depresión|suicidio)\b',
                r'\b(accidente|tragedia|desgracia)\b',
                r'\b(infierno|diablo|demonio)\b',
            ],
            'comparacion': [
                r'\btan\s+\w+\s+como\b',
                r'\bmás\s+\w+\s+que\b',
                r'\bparece\s+(un|una)\b',
                r'\bes\s+como\s+(un|una)\b',
                r'\bse\s+parece\s+a\b',
            ],
            'regla_tres': [
                r'\b(primero|segundo|tercero|1º|2º|3º)\b',
                r'\b(uno|dos|tres)\s+\w+',
                r'\ben\s+primer\s+lugar.*en\s+segundo.*en\s+tercer',
                r'\bprimera.*segunda.*tercera',
            ],
            'animacion': [
                r'\b(dice|pregunta|responde|piensa)\s+el/la\s+\w+',
                r'\bun\s+\w+\s+(habla|dice|piensa)',
                r'\bel\s+\w+\s+le\s+(dice|pregunta)',
            ],
            'ironia': [
                r'\b(qué\s+sorpresa|obviamente|por\s+supuesto)\b',
                r'\bjusto\s+lo\s+que\s+necesitaba\b',
                r'\bperfecto.*exactamente\b',
                r'\b(claro|seguro).*como\s+siempre\b',
            ],
            'sarcasmo': [
                r'\boh\s+(sí|no|claro)\b',
                r'\bqué\s+(original|gracioso|inteligente)\b',
                r'\b(fantástico|genial|perfecto).*\b',
                r'\bmuchas\s+gracias.*por\b',
            ]
        }
    
    def _load_humor_indicators(self) -> Dict[str, List[str]]:
        """Load lexical indicators for humor types."""
        return {
            'juego_palabras': [
                'calambur', 'trabalenguas', 'rima', 'sonido', 'pronuncia',
                'suena', 'parece', 'dice', 'nombre', 'palabra'
            ],
            'humor_negro': [
                'muerte', 'funeral', 'cementerio', 'enfermedad', 'hospital',
                'médico', 'doctor', 'accidente', 'tragedia', 'infierno'
            ],
            'comparacion': [
                'como', 'parece', 'igual', 'similar', 'tan', 'más', 'menos',
                'parecido', 'diferente', 'mismo'
            ],
            'regla_tres': [
                'primero', 'segundo', 'tercero', 'uno', 'dos', 'tres',
                'lista', 'orden', 'secuencia'
            ],
            'animacion': [
                'dice', 'habla', 'pregunta', 'responde', 'piensa', 'opina',
                'comenta', 'explica', 'cuenta'
            ],
            'ironia': [
                'sorpresa', 'obviamente', 'supuesto', 'claro', 'perfecto',
                'exactamente', 'justo', 'ideal'
            ],
            'sarcasmo': [
                'genial', 'fantástico', 'perfecto', 'maravilloso', 'increíble',
                'original', 'gracioso', 'inteligente'
            ]
        }
    
    def analyze_humor_type(self, text: str) -> HumorAnalysis:
        """
        Analyze the humor type in a text.

        Args:
            text: Joke text to analyze

        Returns:
            HumorAnalysis: Analysis result
        """
        text_lower = text.lower()
        scores = {}
        features = {}
        
        # Analyze each humor type
        for humor_type, patterns in self.patterns.items():
            score = 0
            matched_patterns = []
            
            # Search for regex patterns
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    score += len(matches) * 2
                    matched_patterns.extend(matches)
            
            # Search for lexical indicators
            if humor_type in self.indicators:
                for indicator in self.indicators[humor_type]:
                    if indicator in text_lower:
                        score += 1
            
            scores[humor_type] = score
            features[humor_type] = {
                'score': score,
                'patterns': matched_patterns,
                'indicators': [ind for ind in self.indicators.get(humor_type, []) 
                              if ind in text_lower]
            }
        
        # Determine predominant type
        if not any(scores.values()):
            humor_type = HumorType.GENERAL
            confidence = 0.5
        else:
            max_score = max(scores.values())
            humor_type_str = max(scores, key=scores.get)
            humor_type = HumorType(humor_type_str)
            confidence = min(max_score / 10.0, 1.0)  # Normalize to 0-1
        
        return HumorAnalysis(
            text=text,
            humor_type=humor_type,
            confidence=confidence,
            features=features
        )
    
    def batch_analyze(self, texts: List[str]) -> List[HumorAnalysis]:
        """
        Analyze multiple texts in batch.

        Args:
            texts: List of texts to analyze

        Returns:
            List[HumorAnalysis]: List of analyses
        """
        return [self.analyze_humor_type(text) for text in texts]
    
    def get_humor_distribution(self, analyses: List[HumorAnalysis]) -> Dict[str, float]:
        """
        Calculate the distribution of humor types.

        Args:
            analyses: List of humor analyses

        Returns:
            Dict: Percentage distribution of humor types
        """
        if not analyses:
            return {}
        
        type_counts = {}
        for analysis in analyses:
            humor_type = analysis.humor_type.value
            type_counts[humor_type] = type_counts.get(humor_type, 0) + 1
        
        total = len(analyses)
        return {
            humor_type: (count / total) * 100 
            for humor_type, count in type_counts.items()
        }
    
    def filter_by_confidence(self, analyses: List[HumorAnalysis],
                           min_confidence: float = 0.6) -> List[HumorAnalysis]:
        """
        Filter analyses by confidence level.

        Args:
            analyses: List of analyses
            min_confidence: Minimum required confidence

        Returns:
            List[HumorAnalysis]: Filtered analyses
        """
        return [a for a in analyses if a.confidence >= min_confidence]

class HumorMetrics:
    """Metrics for evaluating humor."""
    
    @staticmethod
    def calculate_humor_diversity(analyses: List[HumorAnalysis]) -> float:
        """
        Calculate humor diversity (Shannon entropy).

        Args:
            analyses: List of humor analyses

        Returns:
            float: Diversity index (0-1)
        """
        if not analyses:
            return 0.0
        
        type_counts = {}
        for analysis in analyses:
            humor_type = analysis.humor_type.value
            type_counts[humor_type] = type_counts.get(humor_type, 0) + 1
        
        total = len(analyses)
        entropy = 0.0
        
        for count in type_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * (p.bit_length() - 1)  # log2(p)
        
        # Normalize by maximum possible entropy
        max_entropy = len(HumorType) if len(HumorType) > 1 else 1
        return min(entropy / max_entropy, 1.0)
    
    @staticmethod
    def calculate_average_confidence(analyses: List[HumorAnalysis]) -> float:
        """Calculates the average confidence of the analyses."""
        if not analyses:
            return 0.0
        return sum(a.confidence for a in analyses) / len(analyses)

# Convenience functions
def analyze_humor_type(text: str) -> HumorAnalysis:
    """Analyzes the humor type in a text."""
    analyzer = HumorAnalyzer()
    return analyzer.analyze_humor_type(text)

def get_humor_distribution(texts: List[str]) -> Dict[str, float]:
    """Gets the distribution of humor types in a list of texts."""
    analyzer = HumorAnalyzer()
    analyses = analyzer.batch_analyze(texts)
    return analyzer.get_humor_distribution(analyses)

# Analysis dataset configuration
humor_analysis_datasets = {
    "humor_type_classifier": {
        "description": "Humor type classifier for jokes in Spanish",
        "features": [
            "juego_palabras", "humor_negro", "comparacion", 
            "regla_tres", "animacion", "ironia", "sarcasmo"
        ],
        "confidence_threshold": 0.6,
        "supported_languages": ["es"],
        "analysis_methods": ["pattern_matching", "lexical_indicators"]
    },
    "humor_metrics": {
        "description": "Metrics to evaluate humor diversity and quality",
        "metrics": ["diversity_index", "confidence_average", "type_distribution"],
        "normalization": "shannon_entropy"
    }
}