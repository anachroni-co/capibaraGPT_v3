"""
Language Processing Adapter

Extiende y mejora el system de adaptación lingüística existente,
integrando el SapirWhorfAdapter y añadiendo nuevas capacidades
para procesamiento multilingüe avanzado.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

from .base_adapter import BaseAdapter, AdapterConfig
from .adapter_registry import register_adapter_decorator, AdapterType

logger = logging.getLogger(__name__)

class LanguageFamily(Enum):
    """Linguistic families para agrupación y optimización."""
    INDO_EUROPEAN = "indo_european"
    SINO_TIBETAN = "sino_tibetan"
    AFRO_ASIATIC = "afro_asiatic"
    AUSTRONESIAN = "austronesian"
    TRANS_NEW_GUINEA = "trans_new_guinea"
    NIGER_CONGO = "niger_congo"
    AUSTROASIATIC = "austroasiatic"
    NILO_SAHARAN = "nilo_saharan"
    KHOISAN = "khoisan"
    NATIVE_AMERICAN = "native_american"

class ProcessingMode(Enum):
    """Linguistic processing modes."""
    MONOLINGUAL = "monolingual"
    MULTILINGUAL = "multilingual"
    CODE_SWITCHING = "code_switching"
    TRANSLATION = "translation"
    CROSS_LINGUAL = "cross_lingual"

class CulturalContext(Enum):
    """Cultural contexts for adaptation."""
    WESTERN_INDIVIDUALISTIC = "western_individualistic"
    EASTERN_COLLECTIVE = "eastern_collective"
    AFRICAN_COMMUNAL = "african_communal"
    LATIN_HIERARCHICAL = "latin_hierarchical"
    NORDIC_EGALITARIAN = "nordic_egalitarian"
    MIDDLE_EASTERN_TRADITIONAL = "middle_eastern_traditional"
    INDIGENOUS_HOLISTIC = "indigenous_holistic"

@dataclass
class LanguageProfile:
    """Perfil completo de un idioma."""
    language_code: str
    language_name: str
    family: LanguageFamily
    writing_system: str
    directionality: str = "ltr"  # ltr, rtl, ttb
    morphology: str = "analytic"  # analytic, synthetic, agglutinative
    word_order: str = "svo"  # svo, sov, vso, etc.
    cultural_context: CulturalContext = CulturalContext.WESTERN_INDIVIDUALISTIC
    cognitive_features: Dict[str, str] = field(default_factory=dict)
    processing_complexity: float = 1.0  # Factor de complejidad de procesamiento
    available_resources: List[str] = field(default_factory=list)
    special_adaptations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MultilingualContext:
    """Context for multilingual processing."""
    primary_language: str
    secondary_languages: List[str] = field(default_factory=list)
    code_switching_probability: float = 0.0
    cultural_adaptation_level: float = 0.5  # 0 = mínima, 1 = máxima
    processing_mode: ProcessingMode = ProcessingMode.MONOLINGUAL
    context_metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedLanguageDetector:
    """Detector of language avanzado con soporte multilingüe."""
    
    def __init__(self):
        self.language_patterns = self._initialize_language_patterns()
        self.confidence_threshold = 0.7
        self.code_switching_detector = CodeSwitchingDetector()
        
    def detect_language_advanced(self, text: str) -> Dict[str, Any]:
        """Advanced detection of language con múltiples métricas."""
        if not isinstance(text, str) or not text.strip():
            return {
                'primary_language': 'en',
                'confidence': 0.0,
                'languages_detected': [],
                'is_multilingual': False,
                'code_switching': False
            }
        
        # Detección básica using patterns
        basic_detection = self._detect_basic_language(text)
        
        # Detección de code-switching
        code_switching_result = self.code_switching_detector.detect(text)
        
        # Análisis de características lingüísticas
        linguistic_features = self._analyze_linguistic_features(text)
        
        # Combinar resultados
        result = {
            'primary_language': basic_detection['language'],
            'confidence': basic_detection['confidence'],
            'languages_detected': code_switching_result['languages'],
            'is_multilingual': len(code_switching_result['languages']) > 1,
            'code_switching': code_switching_result['detected'],
            'linguistic_features': linguistic_features,
            'script_analysis': self._analyze_script(text),
            'complexity_score': self._calculate_complexity_score(text)
        }
        
        return result
    
    def _initialize_language_patterns(self) -> Dict[str, List[str]]:
        """Initializes patterns extendidos for detection of languages."""
        return {
            'es': [
                'que', 'de', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son',
                'con', 'para', 'una', 'tiene', 'él', 'fue', 'puede', 'como', 'tiempo', 'muy', 'cuando',
                'hola', 'gracias', 'por favor', 'cómo', 'dónde', 'qué', 'cuándo', 'porqué', 'también'
            ],
            'en': [
                'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on',
                'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
                'hello', 'thank', 'please', 'how', 'where', 'what', 'when', 'why', 'also', 'very'
            ],
            'fr': [
                'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans',
                'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par', 'grand',
                'bonjour', 'merci', 'sil vous plaît', 'comment', 'où', 'quoi', 'quand', 'pourquoi'
            ],
            'de': [
                'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für',
                'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus',
                'hallo', 'danke', 'bitte', 'wie', 'wo', 'was', 'wann', 'warum', 'auch', 'sehr'
            ],
            'zh': [
                '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '个', '上', '也', '很',
                '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '年', '那', '以',
                '你好', '谢谢', '请', '怎么', '哪里', '什么', '什么时候', '为什么', '也', '非常'
            ],
            'ja': [
                'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'する',
                'です', 'ます', 'だ', 'である', 'から', 'まで', 'より', 'など', 'として', 'について',
                'こんにちは', 'ありがとう', 'お願いします', 'どう', 'どこ', '何', 'いつ', 'なぜ', 'も'
            ],
            'ar': [
                'في', 'من', 'إلى', 'على', 'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'كان', 'كانت',
                'يكون', 'تكون', 'له', 'لها', 'بها', 'فيها', 'عليها', 'منها', 'إليها', 'معها',
                'مرحبا', 'شكرا', 'من فضلك', 'كيف', 'أين', 'ماذا', 'متى', 'لماذا', 'أيضا'
            ],
            'ru': [
                'в', 'и', 'не', 'на', 'я', 'быть', 'тот', 'он', 'оно', 'она', 'а', 'то', 'все', 'она',
                'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее',
                'привет', 'спасибо', 'пожалуйста', 'как', 'где', 'что', 'когда', 'почему', 'также'
            ],
            'pt': [
                'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma',
                'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele',
                'olá', 'obrigado', 'por favor', 'como', 'onde', 'o que', 'quando', 'por que', 'também'
            ]
        }
    
    def _detect_basic_language(self, text: str) -> Dict[str, Any]:
        """Basic detection using patterns de palabras."""
        text_lower = text.lower()
        language_scores = defaultdict(float)
        
        for lang, patterns in self.language_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Peso based en longitud del patrón (patrones más largos = más específicos)
                    weight = len(pattern.split()) * 2 if ' ' in pattern else 1
                    language_scores[lang] += weight
        
        if not language_scores:
            return {'language': 'en', 'confidence': 0.1}
        
        # Normalizar scores
        total_score = sum(language_scores.values())
        best_lang = max(language_scores, key=language_scores.get)
        confidence = language_scores[best_lang] / total_score if total_score > 0 else 0.1
        
        return {'language': best_lang, 'confidence': min(confidence, 1.0)}
    
    def _analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyzes linguistic features of text."""
        features = {}
        
        # Análisis de longitud promedio de palabras
        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
            features['word_count'] = len(words)
        else:
            features['avg_word_length'] = 0
            features['word_count'] = 0
        
        # Análisis de caracteres especiales
        features['has_accents'] = any(ord(char) > 127 for char in text)
        features['has_chinese_chars'] = any('\u4e00' <= char <= '\u9fff' for char in text)
        features['has_arabic_chars'] = any('\u0600' <= char <= '\u06ff' for char in text)
        features['has_cyrillic_chars'] = any('\u0400' <= char <= '\u04ff' for char in text)
        
        # Análisis de puntuación
        punctuation_chars = '.,!?;:'
        features['punctuation_density'] = sum(1 for char in text if char in punctuation_chars) / len(text) if text else 0
        
        return features
    
    def _analyze_script(self, text: str) -> Dict[str, str]:
        """Analyzes writing system of text."""
        script_analysis = {
            'primary_script': 'latin',
            'mixed_scripts': False,
            'direction': 'ltr'
        }
        
        # Detectar scripts
        has_latin = any('a' <= char.lower() <= 'z' for char in text)
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        has_arabic = any('\u0600' <= char <= '\u06ff' for char in text)
        has_cyrillic = any('\u0400' <= char <= '\u04ff' for char in text)
        has_japanese = any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text)
        
        scripts = []
        if has_latin: scripts.append('latin')
        if has_chinese: scripts.append('chinese')
        if has_arabic: scripts.append('arabic')
        if has_cyrillic: scripts.append('cyrillic')
        if has_japanese: scripts.append('japanese')
        
        if scripts:
            script_analysis['primary_script'] = scripts[0]
            script_analysis['mixed_scripts'] = len(scripts) > 1
        
        # Determinar dirección
        if has_arabic:
            script_analysis['direction'] = 'rtl'
        elif has_chinese:
            script_analysis['direction'] = 'ttb'  # Top-to-bottom (tradicional)
        
        return script_analysis
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculates linguistic complexity score."""
        if not text:
            return 0.0
        
        complexity = 0.0
        
        # Complejidad por longitud de palabras
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            complexity += min(avg_word_length / 10.0, 1.0)  # Normalizar a 1.0
        
        # Complejidad por caracteres especiales
        special_chars = sum(1 for char in text if ord(char) > 127)
        complexity += min(special_chars / len(text), 0.5) if text else 0
        
        # Complejidad por scripts mixtos
        script_analysis = self._analyze_script(text)
        if script_analysis['mixed_scripts']:
            complexity += 0.3
        
        return min(complexity, 2.0)  # Cap at 2.0

class CodeSwitchingDetector:
    """Detector de cambio de código (code-switching) en texto multilingüe."""
    
    def __init__(self):
        self.window_size = 10  # Palabras a analizar por ventana
        
    def detect(self, text: str) -> Dict[str, Any]:
        """Detecta cambio de código en el texto."""
        words = text.split()
        if len(words) < 2:
            return {
                'detected': False,
                'languages': [],
                'switch_points': [],
                'confidence': 0.0
            }
        
        # Analizar ventanas deslizantes
        languages_detected = set()
        switch_points = []
        
        detector = AdvancedLanguageDetector()
        
        for i in range(0, len(words), self.window_size):
            window = ' '.join(words[i:i + self.window_size])
            result = detector._detect_basic_language(window)
            
            if result['confidence'] > 0.3:  # Umbral mínimo de confianza
                languages_detected.add(result['language'])
                
                if len(languages_detected) > 1:
                    switch_points.append(i)
        
        return {
            'detected': len(languages_detected) > 1,
            'languages': list(languages_detected),
            'switch_points': switch_points,
            'confidence': min(len(switch_points) / (len(words) / self.window_size), 1.0)
        }

class CulturalAdaptationEngine:
    """Cultural adaptation engine for contextualized processing."""
    
    def __init__(self):
        self.cultural_mappings = self._initialize_cultural_mappings()
        self.adaptation_strategies = self._initialize_adaptation_strategies()
    
    def _initialize_cultural_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initializes mapeos culturales."""
        return {
            'privacy_concepts': {
                'western_individualistic': {'weight': 1.0, 'emphasis': 'individual_rights'},
                'eastern_collective': {'weight': 0.6, 'emphasis': 'group_harmony'},
                'african_communal': {'weight': 0.4, 'emphasis': 'community_benefit'},
                'latin_hierarchical': {'weight': 0.8, 'emphasis': 'respect_authority'},
                'nordic_egalitarian': {'weight': 0.9, 'emphasis': 'transparency'},
                'middle_eastern_traditional': {'weight': 0.7, 'emphasis': 'family_honor'},
                'indigenous_holistic': {'weight': 0.5, 'emphasis': 'interconnectedness'}
            },
            'time_concepts': {
                'western_individualistic': {'structure': 'linear', 'precision': 'high'},
                'eastern_collective': {'structure': 'cyclical', 'precision': 'contextual'},
                'african_communal': {'structure': 'event_based', 'precision': 'flexible'},
                'latin_hierarchical': {'structure': 'relationship_based', 'precision': 'moderate'},
                'nordic_egalitarian': {'structure': 'efficient', 'precision': 'high'},
                'middle_eastern_traditional': {'structure': 'traditional_cycles', 'precision': 'contextual'},
                'indigenous_holistic': {'structure': 'seasonal_natural', 'precision': 'intuitive'}
            },
            'communication_styles': {
                'western_individualistic': {'directness': 0.8, 'formality': 0.5},
                'eastern_collective': {'directness': 0.3, 'formality': 0.8},
                'african_communal': {'directness': 0.6, 'formality': 0.6},
                'latin_hierarchical': {'directness': 0.4, 'formality': 0.9},
                'nordic_egalitarian': {'directness': 0.9, 'formality': 0.3},
                'middle_eastern_traditional': {'directness': 0.5, 'formality': 0.8},
                'indigenous_holistic': {'directness': 0.4, 'formality': 0.4}
            }
        }
    
    def _initialize_adaptation_strategies(self) -> Dict[str, Callable]:
        """Initializes adaptation strategies."""
        return {
            'adjust_formality': self._adjust_formality,
            'adapt_time_references': self._adapt_time_references,
            'modify_directness': self._modify_directness,
            'contextualize_privacy': self._contextualize_privacy
        }
    
    def adapt_content(self, 
                     content: str, 
                     source_culture: CulturalContext,
                     target_culture: CulturalContext) -> Dict[str, Any]:
        """Adapts content between cultural contexts."""
        
        adaptations = {}
        
        # Aplicar strategys de adaptación
        for strategy_name, strategy_func in self.adaptation_strategies.items():
            try:
                adaptation = strategy_func(content, source_culture, target_culture)
                adaptations[strategy_name] = adaptation
            except Exception as e:
                logger.warning(f"Cultural adaptation strategy {strategy_name} failed: {e}")
                adaptations[strategy_name] = {'adapted_content': content, 'changes': []}
        
        # Combinar adaptaciones
        final_content = content
        all_changes = []
        
        for adaptation in adaptations.values():
            if 'adapted_content' in adaptation:
                final_content = adaptation['adapted_content']
            if 'changes' in adaptation:
                all_changes.extend(adaptation['changes'])
        
        return {
            'original_content': content,
            'adapted_content': final_content,
            'source_culture': source_culture.value,
            'target_culture': target_culture.value,
            'adaptations_applied': list(adaptations.keys()),
            'changes_made': all_changes,
            'adaptation_score': len(all_changes) / max(len(content.split()), 1)
        }
    
    def _adjust_formality(self, content: str, source: CulturalContext, target: CulturalContext) -> Dict[str, Any]:
        """Ajusta el nivel de formalidad."""
        source_formality = self.cultural_mappings['communication_styles'].get(source.value, {}).get('formality', 0.5)
        target_formality = self.cultural_mappings['communication_styles'].get(target.value, {}).get('formality', 0.5)
        
        changes = []
        adapted_content = content
        
        if target_formality > source_formality + 0.2:  # Aumentar formalidad
            # Reemplazos para aumentar formalidad
            formal_replacements = {
                'hi': 'hello',
                'thanks': 'thank you',
                'ok': 'very well',
                'yeah': 'yes',
                'gonna': 'going to',
                'wanna': 'want to'
            }
            
            for informal, formal in formal_replacements.items():
                if informal in adapted_content.lower():
                    adapted_content = adapted_content.replace(informal, formal)
                    changes.append(f"Increased formality: '{informal}' → '{formal}'")
        
        elif source_formality > target_formality + 0.2:  # Reducir formalidad
            # Reemplazos para reducir formalidad
            informal_replacements = {
                'greetings': 'hi',
                'thank you very much': 'thanks',
                'I would like to': "I'd like to",
                'it is': "it's",
                'cannot': "can't"
            }
            
            for formal, informal in informal_replacements.items():
                if formal in adapted_content:
                    adapted_content = adapted_content.replace(formal, informal)
                    changes.append(f"Reduced formality: '{formal}' → '{informal}'")
        
        return {
            'adapted_content': adapted_content,
            'changes': changes
        }
    
    def _adapt_time_references(self, content: str, source: CulturalContext, target: CulturalContext) -> Dict[str, Any]:
        """Adapts temporal references according to cultural context."""
        changes = []
        adapted_content = content
        
        source_time = self.cultural_mappings['time_concepts'].get(source.value, {})
        target_time = self.cultural_mappings['time_concepts'].get(target.value, {})
        
        # Adaptaciones básicas de tiempo
        if target_time.get('structure') == 'event_based' and 'at 3 PM' in content:
            adapted_content = adapted_content.replace('at 3 PM', 'after lunch')
            changes.append("Adapted time reference to event-based: '3 PM' → 'after lunch'")
        
        if target_time.get('precision') == 'flexible' and 'exactly' in content:
            adapted_content = adapted_content.replace('exactly', 'around')
            changes.append("Reduced time precision: 'exactly' → 'around'")
        
        return {
            'adapted_content': adapted_content,
            'changes': changes
        }
    
    def _modify_directness(self, content: str, source: CulturalContext, target: CulturalContext) -> Dict[str, Any]:
        """Modifies the directness level in communication."""
        changes = []
        adapted_content = content
        
        source_directness = self.cultural_mappings['communication_styles'].get(source.value, {}).get('directness', 0.5)
        target_directness = self.cultural_mappings['communication_styles'].get(target.value, {}).get('directness', 0.5)
        
        if target_directness < source_directness - 0.2:  # Reducir directness
            # Hacer el lenguaje más indirecto
            indirect_replacements = {
                'you should': 'you might want to consider',
                'you must': 'it would be advisable to',
                'do this': 'perhaps you could do this',
                'wrong': 'not quite right'
            }
            
            for direct, indirect in indirect_replacements.items():
                if direct in adapted_content.lower():
                    adapted_content = adapted_content.replace(direct, indirect)
                    changes.append(f"Reduced directness: '{direct}' → '{indirect}'")
        
        return {
            'adapted_content': adapted_content,
            'changes': changes
        }
    
    def _contextualize_privacy(self, content: str, source: CulturalContext, target: CulturalContext) -> Dict[str, Any]:
        """Contextualiza conceptos de privacidad según la cultura."""
        changes = []
        adapted_content = content
        
        source_privacy = self.cultural_mappings['privacy_concepts'].get(source.value, {})
        target_privacy = self.cultural_mappings['privacy_concepts'].get(target.value, {})
        
        if target_privacy.get('emphasis') == 'community_benefit' and 'personal data' in content:
            adapted_content = adapted_content.replace('personal data', 'community information')
            changes.append("Adapted privacy concept for community context: 'personal data' → 'community information'")
        
        return {
            'adapted_content': adapted_content,
            'changes': changes
        }

@register_adapter_decorator(
    adapter_type=AdapterType.LANGUAGE_PROCESSING,
    priority=75,
    capabilities=["multilingual_processing", "cultural_adaptation", "code_switching_detection"],
    metadata={"version": "2.0", "extends_sapir_whorf": True}
)
class LanguageProcessingAdapter(BaseAdapter):
    """
    Adapter avanzado para procesamiento de lenguaje que extiende
    las capacidades del SapirWhorfAdapter existente.
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self.language_detector = AdvancedLanguageDetector()
        self.cultural_engine = CulturalAdaptationEngine()
        self.language_profiles = self._initialize_language_profiles()
        self.sapir_whorf_adapter = None
        
    def _initialize_impl(self) -> bool:
        """Initializes language processing adapter."""
        try:
            # Intentar importar y usar el SapirWhorfAdapter existente
            try:
                from capibara.sub_models.semiotic.sapir_whorf_adapter import SapirWhorfAdapter
                self.sapir_whorf_adapter = SapirWhorfAdapter()
                self.logger.info("Integrated with existing SapirWhorfAdapter")
            except ImportError:
                self.logger.warning("SapirWhorfAdapter not available, using standalone implementation")
            
            self.logger.info("Language processing adapter initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize language processing adapter: {e}")
            return False
    
    def _execute_impl(self, 
                     operation: str = "process", 
                     text: str = "",
                     context: Optional[MultilingualContext] = None,
                     **kwargs) -> Dict[str, Any]:
        """Executes language processing operation."""
        
        if operation == "detect":
            return self._execute_language_detection(text, **kwargs)
        elif operation == "adapt_cultural":
            return self._execute_cultural_adaptation(text, context, **kwargs)
        elif operation == "process_multilingual":
            return self._execute_multilingual_processing(text, context, **kwargs)
        elif operation == "code_switching":
            return self._execute_code_switching_detection(text, **kwargs)
        elif operation == "get_profiles":
            return self._get_language_profiles()
        elif operation == "sapir_whorf":
            return self._execute_sapir_whorf_processing(text, **kwargs)
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    def _execute_language_detection(self, text: str, **kwargs) -> Dict[str, Any]:
        """Executes advanced language detection."""
        detection_result = self.language_detector.detect_language_advanced(text)
        
        # Enriquecer con información de perfiles
        primary_lang = detection_result['primary_language']
        if primary_lang in self.language_profiles:
            profile = self.language_profiles[primary_lang]
            detection_result['language_profile'] = {
                'family': profile.family.value,
                'writing_system': profile.writing_system,
                'cultural_context': profile.cultural_context.value,
                'processing_complexity': profile.processing_complexity
            }
        
        return {
            'detection_result': detection_result,
            'processing_recommendations': self._get_processing_recommendations(detection_result)
        }
    
    def _execute_cultural_adaptation(self, 
                                   text: str, 
                                   context: Optional[MultilingualContext],
                                   **kwargs) -> Dict[str, Any]:
        """Executes cultural adaptation of text."""
        
        source_culture = kwargs.get('source_culture', CulturalContext.WESTERN_INDIVIDUALISTIC)
        target_culture = kwargs.get('target_culture', CulturalContext.EASTERN_COLLECTIVE)
        
        if isinstance(source_culture, str):
            source_culture = CulturalContext(source_culture)
        if isinstance(target_culture, str):
            target_culture = CulturalContext(target_culture)
        
        adaptation_result = self.cultural_engine.adapt_content(text, source_culture, target_culture)
        
        return {
            'adaptation_result': adaptation_result,
            'context_used': context.__dict__ if context else None
        }
    
    def _execute_multilingual_processing(self, 
                                       text: str,
                                       context: Optional[MultilingualContext],
                                       **kwargs) -> Dict[str, Any]:
        """Executes complete multilingual processing."""
        
        # Detección of languages
        detection = self.language_detector.detect_language_advanced(text)
        
        # Análisis de code-switching si es multilingüe
        code_switching = None
        if detection['is_multilingual']:
            code_switching = self.language_detector.code_switching_detector.detect(text)
        
        # Adaptación cultural si se especifica contexto
        cultural_adaptation = None
        if context and context.cultural_adaptation_level > 0.3:
            # Determinar culturas basadas en idiomas
            primary_culture = self._map_language_to_culture(detection['primary_language'])
            if context.secondary_languages:
                secondary_culture = self._map_language_to_culture(context.secondary_languages[0])
                cultural_adaptation = self.cultural_engine.adapt_content(
                    text, primary_culture, secondary_culture
                )
        
        # Procesamiento con SapirWhorf si está disponible
        sapir_whorf_result = None
        if self.sapir_whorf_adapter:
            try:
                sapir_whorf_result = self.sapir_whorf_adapter(text)
            except Exception as e:
                self.logger.warning(f"SapirWhorf processing failed: {e}")
        
        return {
            'language_detection': detection,
            'code_switching_analysis': code_switching,
            'cultural_adaptation': cultural_adaptation,
            'sapir_whorf_processing': sapir_whorf_result,
            'processing_context': context.__dict__ if context else None,
            'recommendations': self._generate_processing_recommendations(detection, context)
        }
    
    def _execute_code_switching_detection(self, text: str, **kwargs) -> Dict[str, Any]:
        """Executes code-switching specific detection."""
        result = self.language_detector.code_switching_detector.detect(text)
        
        # Enriquecer con análisis adicional
        if result['detected']:
            result['analysis'] = {
                'switch_frequency': len(result['switch_points']) / len(text.split()) if text.split() else 0,
                'dominant_language': max(result['languages'], key=lambda x: text.lower().count(x)) if result['languages'] else 'unknown',
                'complexity_score': len(result['languages']) * result['confidence']
            }
        
        return result
    
    def _execute_sapir_whorf_processing(self, text: str, **kwargs) -> Dict[str, Any]:
        """Executes SapirWhorf specific processing."""
        if not self.sapir_whorf_adapter:
            return {"error": "SapirWhorf adapter not available"}
        
        try:
            # Usar el adapter existente
            result = self.sapir_whorf_adapter(text, **kwargs)
            
            # Enriquecer con análisis adicional
            if isinstance(result, dict):
                result['enhanced_analysis'] = {
                    'linguistic_relativity_score': self._calculate_relativity_score(text),
                    'cognitive_load_estimate': self._estimate_cognitive_load(text),
                    'cultural_markers': self._identify_cultural_markers(text)
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"SapirWhorf processing failed: {e}")
            return {"error": str(e)}
    
    def _get_language_profiles(self) -> Dict[str, Any]:
        """Gets available language profiles."""
        profiles_data = {}
        
        for lang_code, profile in self.language_profiles.items():
            profiles_data[lang_code] = {
                'language_name': profile.language_name,
                'family': profile.family.value,
                'writing_system': profile.writing_system,
                'directionality': profile.directionality,
                'morphology': profile.morphology,
                'word_order': profile.word_order,
                'cultural_context': profile.cultural_context.value,
                'processing_complexity': profile.processing_complexity,
                'available_resources': profile.available_resources
            }
        
        return {
            'language_profiles': profiles_data,
            'total_languages': len(self.language_profiles),
            'supported_families': list(set(p.family.value for p in self.language_profiles.values()))
        }
    
    def _initialize_language_profiles(self) -> Dict[str, LanguageProfile]:
        """Initializes perfiles extendidos of languages."""
        profiles = {
            'en': LanguageProfile(
                language_code='en',
                language_name='English',
                family=LanguageFamily.INDO_EUROPEAN,
                writing_system='latin',
                directionality='ltr',
                morphology='analytic',
                word_order='svo',
                cultural_context=CulturalContext.WESTERN_INDIVIDUALISTIC,
                cognitive_features={
                    'time': 'linear',
                    'space': 'cartesian',
                    'agency': 'individual',
                    'gender': 'minimal'
                },
                processing_complexity=1.0,
                available_resources=['tokenizer', 'pos_tagger', 'parser']
            ),
            'es': LanguageProfile(
                language_code='es',
                language_name='Spanish',
                family=LanguageFamily.INDO_EUROPEAN,
                writing_system='latin',
                directionality='ltr',
                morphology='synthetic',
                word_order='svo',
                cultural_context=CulturalContext.LATIN_HIERARCHICAL,
                cognitive_features={
                    'time': 'past-focused',
                    'space': 'relational',
                    'agency': 'contextual',
                    'gender': 'grammatical'
                },
                processing_complexity=1.2,
                available_resources=['tokenizer', 'pos_tagger']
            ),
            'zh': LanguageProfile(
                language_code='zh',
                language_name='Chinese',
                family=LanguageFamily.SINO_TIBETAN,
                writing_system='chinese_characters',
                directionality='ltr',
                morphology='analytic',
                word_order='svo',
                cultural_context=CulturalContext.EASTERN_COLLECTIVE,
                cognitive_features={
                    'time': 'cyclical',
                    'space': 'hierarchical',
                    'agency': 'collective',
                    'gender': 'neutral'
                },
                processing_complexity=1.8,
                available_resources=['tokenizer', 'segmenter']
            ),
            'ar': LanguageProfile(
                language_code='ar',
                language_name='Arabic',
                family=LanguageFamily.AFRO_ASIATIC,
                writing_system='arabic',
                directionality='rtl',
                morphology='synthetic',
                word_order='vso',
                cultural_context=CulturalContext.MIDDLE_EASTERN_TRADITIONAL,
                cognitive_features={
                    'time': 'eternal-present',
                    'space': 'symbolic',
                    'agency': 'divine-modulated',
                    'gender': 'dualistic'
                },
                processing_complexity=2.0,
                available_resources=['tokenizer', 'morphological_analyzer']
            ),
            'fr': LanguageProfile(
                language_code='fr',
                language_name='French',
                family=LanguageFamily.INDO_EUROPEAN,
                writing_system='latin',
                directionality='ltr',
                morphology='synthetic',
                word_order='svo',
                cultural_context=CulturalContext.WESTERN_INDIVIDUALISTIC,
                cognitive_features={
                    'time': 'precise',
                    'space': 'geometric',
                    'agency': 'individual',
                    'gender': 'grammatical'
                },
                processing_complexity=1.3,
                available_resources=['tokenizer', 'pos_tagger', 'parser']
            )
        }
        
        return profiles
    
    def _get_processing_recommendations(self, detection_result: Dict[str, Any]) -> List[str]:
        """Generates processing recommendations based on detection."""
        recommendations = []
        
        # Recomendaciones basadas en complejidad
        complexity = detection_result.get('complexity_score', 1.0)
        if complexity > 1.5:
            recommendations.append("Use advanced tokenization for complex linguistic structure")
            recommendations.append("Enable morphological analysis")
        
        # Recomendaciones basadas en multilingualismo
        if detection_result.get('is_multilingual', False):
            recommendations.append("Enable code-switching detection")
            recommendations.append("Use multilingual embedding models")
        
        # Recomendaciones basadas en script
        script_analysis = detection_result.get('script_analysis', {})
        if script_analysis.get('direction') == 'rtl':
            recommendations.append("Configure right-to-left text processing")
        if script_analysis.get('mixed_scripts', False):
            recommendations.append("Use script-aware text processing")
        
        return recommendations
    
    def _generate_processing_recommendations(self, 
                                           detection: Dict[str, Any],
                                           context: Optional[MultilingualContext]) -> List[str]:
        """Generates complete processing recommendations."""
        recommendations = []
        
        # Recomendaciones básicas de detección
        recommendations.extend(self._get_processing_recommendations(detection))
        
        # Recomendaciones basadas en contexto
        if context:
            if context.processing_mode == ProcessingMode.CODE_SWITCHING:
                recommendations.append("Optimize for code-switching scenarios")
            elif context.processing_mode == ProcessingMode.TRANSLATION:
                recommendations.append("Enable translation-aware processing")
            
            if context.cultural_adaptation_level > 0.7:
                recommendations.append("Apply high-level cultural adaptations")
            elif context.cultural_adaptation_level > 0.3:
                recommendations.append("Apply moderate cultural adaptations")
        
        # Recomendaciones basadas en recursos disponibles
        primary_lang = detection.get('primary_language', 'en')
        if primary_lang in self.language_profiles:
            profile = self.language_profiles[primary_lang]
            if 'parser' in profile.available_resources:
                recommendations.append("Utilize syntactic parsing for better understanding")
            if 'morphological_analyzer' in profile.available_resources:
                recommendations.append("Apply morphological analysis")
        
        return list(set(recommendations))  # Eliminar duplicados
    
    def _map_language_to_culture(self, language_code: str) -> CulturalContext:
        """Mapea código of language a contexto cultural."""
        if language_code in self.language_profiles:
            return self.language_profiles[language_code].cultural_context
        
        # Mapeo por defecto
        cultural_mapping = {
            'en': CulturalContext.WESTERN_INDIVIDUALISTIC,
            'es': CulturalContext.LATIN_HIERARCHICAL,
            'zh': CulturalContext.EASTERN_COLLECTIVE,
            'ar': CulturalContext.MIDDLE_EASTERN_TRADITIONAL,
            'fr': CulturalContext.WESTERN_INDIVIDUALISTIC,
            'de': CulturalContext.WESTERN_INDIVIDUALISTIC,
            'ja': CulturalContext.EASTERN_COLLECTIVE,
            'pt': CulturalContext.LATIN_HIERARCHICAL
        }
        
        return cultural_mapping.get(language_code, CulturalContext.WESTERN_INDIVIDUALISTIC)
    
    def _calculate_relativity_score(self, text: str) -> float:
        """Calculates linguistic relativity score."""
        # Simplified implementation
        complexity_indicators = [
            'metaphor', 'analogy', 'cultural', 'context', 'perspective',
            'viewpoint', 'interpretation', 'meaning', 'significance'
        ]
        
        text_lower = text.lower()
        score = sum(1 for indicator in complexity_indicators if indicator in text_lower)
        
        return min(score / 10.0, 1.0)  # Normalizar a 1.0
    
    def _estimate_cognitive_load(self, text: str) -> float:
        """Estima la carga cognitiva del texto."""
        words = text.split()
        if not words:
            return 0.0
        
        # Factores de carga cognitiva
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = len(words) / max(sentence_count, 1)
        
        # Combinar factores
        load = (avg_word_length / 10.0) + (avg_sentence_length / 20.0)
        
        return min(load, 2.0)
    
    def _identify_cultural_markers(self, text: str) -> List[str]:
        """Identifica marcadores culturales en el texto."""
        markers = []
        
        cultural_indicators = {
            'time_markers': ['punctuality', 'schedule', 'deadline', 'on time'],
            'hierarchy_markers': ['respect', 'authority', 'senior', 'junior'],
            'collectivism_markers': ['we', 'our', 'together', 'community'],
            'individualism_markers': ['I', 'my', 'personal', 'individual'],
            'formality_markers': ['please', 'thank you', 'sir', 'madam']
        }
        
        text_lower = text.lower()
        
        for category, indicators in cultural_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    markers.append(f"{category}: {indicator}")
        
        return markers
    
    # Métodos de conveniencia
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Convenience method for language detection."""
        return self.execute("detect", text=text)
    
    def adapt_culturally(self, 
                        text: str,
                        source_culture: Union[str, CulturalContext],
                        target_culture: Union[str, CulturalContext]) -> Dict[str, Any]:
        """Convenience method for cultural adaptation."""
        return self.execute("adapt_cultural", text=text, 
                          source_culture=source_culture, target_culture=target_culture)
    
    def process_multilingual(self, 
                           text: str,
                           context: Optional[MultilingualContext] = None) -> Dict[str, Any]:
        """Convenience method for multilingual processing."""
        return self.execute("process_multilingual", text=text, context=context)
    
    def detect_code_switching(self, text: str) -> Dict[str, Any]:
        """Convenience method for code-switching detection."""
        return self.execute("code_switching", text=text)


# Global adapter instance
language_adapter = LanguageProcessingAdapter()

# Funciones de utilidad globales
def detect_language_advanced(text: str) -> Dict[str, Any]:
    """Global function for advanced language detection."""
    return language_adapter.detect_language(text)

def adapt_cultural_context(text: str, 
                          source_culture: str,
                          target_culture: str) -> Dict[str, Any]:
    """Global function for cultural adaptation."""
    return language_adapter.adapt_culturally(text, source_culture, target_culture)

def process_multilingual_text(text: str, 
                             primary_language: str = "",
                             secondary_languages: List[str] = None) -> Dict[str, Any]:
    """Global function for multilingual processing."""
    context = MultilingualContext(
        primary_language=primary_language,
        secondary_languages=secondary_languages or [],
        processing_mode=ProcessingMode.MULTILINGUAL
    )
    return language_adapter.process_multilingual(text, context)

def get_language_processing_info():
    """Gets information from the language processing system."""
    return {
        'adapter_status': language_adapter.get_status().value,
        'language_profiles': language_adapter.execute("get_profiles"),
        'sapir_whorf_available': language_adapter.sapir_whorf_adapter is not None,
        'adapter_metrics': language_adapter.get_metrics().__dict__
    }