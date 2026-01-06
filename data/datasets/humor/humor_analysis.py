"""
Humor Analysis Tools for CapibaraGPT-v2
=======================================

Herramientas para analizar y categorizar humor en español.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class HumorType(Enum):
    """Tipos de humor identificables."""
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
    """Resultado del análisis de humor."""
    text: str
    humor_type: HumorType
    confidence: float
    features: Dict[str, Any]
    explanation: Optional[str] = None

class HumorAnalyzer:
    """Analizador de tipos de humor en chistes españoles."""
    
    def __init__(self):
        self.patterns = self._load_humor_patterns()
        self.indicators = self._load_humor_indicators()
    
    def _load_humor_patterns(self) -> Dict[str, List[str]]:
        """Loads patrones regex para detectar tipos de humor."""
        return {
            'juego_palabras': [
                r'\b(\w+)\b.*\b\1\w*\b',  # Repetición de palabras similares
                r'¿[^?]*\?.*¡[^!]*!',      # Pregunta seguida de exclamación
                r'\b(suena?|parece|dice)\s+(como|que)',  # Indicadores de sonido/similitud
                r'\b(dos|tres)\s+\w+\s+(van|entran|salen)',  # Setup típico de chistes
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
        """Loads indicadores léxicos de tipos de humor."""
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
        Analiza el tipo de humor en un texto.
        
        Args:
            text: Texto del chiste a analizar
            
        Returns:
            HumorAnalysis: Resultado del análisis
        """
        text_lower = text.lower()
        scores = {}
        features = {}
        
        # Analizar cada tipo de humor
        for humor_type, patterns in self.patterns.items():
            score = 0
            matched_patterns = []
            
            # Buscar patrones regex
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    score += len(matches) * 2
                    matched_patterns.extend(matches)
            
            # Buscar indicadores léxicos
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
        
        # Determinar tipo predominante
        if not any(scores.values()):
            humor_type = HumorType.GENERAL
            confidence = 0.5
        else:
            max_score = max(scores.values())
            humor_type_str = max(scores, key=scores.get)
            humor_type = HumorType(humor_type_str)
            confidence = min(max_score / 10.0, 1.0)  # Normalizar a 0-1
        
        return HumorAnalysis(
            text=text,
            humor_type=humor_type,
            confidence=confidence,
            features=features
        )
    
    def batch_analyze(self, texts: List[str]) -> List[HumorAnalysis]:
        """
        Analiza múltiples textos en lote.
        
        Args:
            texts: Lista de textos a analizar
            
        Returns:
            List[HumorAnalysis]: Lista de análisis
        """
        return [self.analyze_humor_type(text) for text in texts]
    
    def get_humor_distribution(self, analyses: List[HumorAnalysis]) -> Dict[str, float]:
        """
        Calcula la distribución de tipos de humor.
        
        Args:
            analyses: Lista de análisis de humor
            
        Returns:
            Dict: Distribución porcentual de tipos de humor
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
        Filtra análisis por nivel de confianza.
        
        Args:
            analyses: Lista de análisis
            min_confidence: Confianza mínima requerida
            
        Returns:
            List[HumorAnalysis]: Análisis filtrados
        """
        return [a for a in analyses if a.confidence >= min_confidence]

class HumorMetrics:
    """Métricas para evaluar humor."""
    
    @staticmethod
    def calculate_humor_diversity(analyses: List[HumorAnalysis]) -> float:
        """
        Calcula la diversidad de humor (entropía de Shannon).
        
        Args:
            analyses: Lista de análisis de humor
            
        Returns:
            float: Índice de diversidad (0-1)
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
        
        # Normalizar por máxima entropía posible
        max_entropy = len(HumorType) if len(HumorType) > 1 else 1
        return min(entropy / max_entropy, 1.0)
    
    @staticmethod
    def calculate_average_confidence(analyses: List[HumorAnalysis]) -> float:
        """Calculates la confianza promedio de los análisis."""
        if not analyses:
            return 0.0
        return sum(a.confidence for a in analyses) / len(analyses)

# Funciones de conveniencia
def analyze_humor_type(text: str) -> HumorAnalysis:
    """Analyzes el tipo de humor en un texto."""
    analyzer = HumorAnalyzer()
    return analyzer.analyze_humor_type(text)

def get_humor_distribution(texts: List[str]) -> Dict[str, float]:
    """Gets la distribución de tipos de humor en una lista de textos."""
    analyzer = HumorAnalyzer()
    analyses = analyzer.batch_analyze(texts)
    return analyzer.get_humor_distribution(analyses)

# Configuración de datasets de análisis
humor_analysis_datasets = {
    "humor_type_classifier": {
        "description": "Clasificador de tipos de humor para chistes en español",
        "features": [
            "juego_palabras", "humor_negro", "comparacion", 
            "regla_tres", "animacion", "ironia", "sarcasmo"
        ],
        "confidence_threshold": 0.6,
        "supported_languages": ["es"],
        "analysis_methods": ["pattern_matching", "lexical_indicators"]
    },
    "humor_metrics": {
        "description": "Métricas para evaluar diversidad y calidad del humor",
        "metrics": ["diversity_index", "confidence_average", "type_distribution"],
        "normalization": "shannon_entropy"
    }
}