"""
Specialized Processors Module for CapibaraGPT

This module implements procesadores especializados para diferentes tipos de datos
y tareas, incluyendo procesamiento multimodal, análisis de sentimientos,
extracción de entidades y más.

Características:
- Procesadores multimodales (texto, imagen, audio)
- Análisis de sentimientos avanzado
- Extracción de entidades nombradas
- Procesamiento de código fuente
- Análisis de documentos estructurados
- Procesadores de dominio específico
"""

import os
import sys
import logging
import re
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ProcessorType(Enum):
    """Tipos de procesadores especializados."""
    TEXT_ANALYSIS = "text_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ENTITY_EXTRACTION = "entity_extraction"
    CODE_ANALYSIS = "code_analysis"
    DOCUMENT_STRUCTURE = "document_structure"
    MULTIMODAL_FUSION = "multimodal_fusion"
    DOMAIN_SPECIFIC = "domain_specific"

@dataclass
class ProcessorConfig:
    """Base configuration for specialized processors."""
    
    processor_type: ProcessorType
    input_format: str = "text"
    output_format: str = "structured"
    max_input_length: int = 2048
    
    # Analysis settings
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    confidence_threshold: float = 0.5
    
    # Performance settings
    batch_processing: bool = True
    parallel_processing: bool = False
    cache_results: bool = True

class TextAnalysisProcessor:
    """Procesador especializado para análisis de texto avanzado."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.language_patterns = self._load_language_patterns()
        self.analysis_cache = {}
        
        logger.info("📝 TextAnalysisProcessor initialized")
    
    def _load_language_patterns(self) -> Dict[str, List[str]]:
        """Loads patrones de lenguaje para análisis."""
        return {
            'questions': [r'\?$', r'^(what|how|why|when|where|who)', r'could you', r'can you'],
            'commands': [r'^(please|kindly)', r'(do|make|create|generate)', r'!$'],
            'statements': [r'\.$', r'^(i think|i believe|in my opinion)', r'(is|are|was|were)'],
            'emotions': [r'(happy|sad|angry|excited|frustrated|amazed)', r'(love|hate|like|dislike)'],
            'technical': [r'(algorithm|function|class|method)', r'(import|export|compile)', r'(database|server|api)']
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyzes text and extracts features."""
        if not text or len(text.strip()) == 0:
            return {'error': 'Empty text provided'}
        
        # Check cache first
        cache_key = hash(text)
        if self.config.cache_results and cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        analysis = {
            'text': text,
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'paragraph_count': len(text.split('\n\n')),
            'patterns': {},
            'complexity': self._calculate_complexity(text),
            'readability': self._calculate_readability(text),
            'language_features': self._extract_language_features(text)
        }
        
        # Analyze patterns
        for pattern_type, patterns in self.language_patterns.items():
            matches = 0
            for pattern in patterns:
                matches += len(re.findall(pattern, text, re.IGNORECASE))
            analysis['patterns'][pattern_type] = matches
        
        # Determine dominant pattern
        if analysis['patterns']:
            dominant_pattern = max(analysis['patterns'].items(), key=lambda x: x[1])
            analysis['dominant_pattern'] = dominant_pattern[0]
            analysis['pattern_confidence'] = dominant_pattern[1] / analysis['word_count'] if analysis['word_count'] > 0 else 0
        
        # Cache result
        if self.config.cache_results:
            self.analysis_cache[cache_key] = analysis
        
        return analysis
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculates complejidad del texto."""
        words = text.split()
        if not words:
            return 0.0
        
        # Factores de complejidad
        avg_word_length = np.mean([len(word) for word in words])
        unique_words_ratio = len(set(words)) / len(words)
        sentence_length_variance = self._calculate_sentence_variance(text)
        
        # Complejidad normalizada
        complexity = (avg_word_length / 10 + unique_words_ratio + sentence_length_variance / 100) / 3
        return min(complexity, 1.0)
    
    def _calculate_readability(self, text: str) -> float:
        """Calculates índice de legibilidad (Flesch-like)."""
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(word) for word in words])
        
        # Flesch-like score (simplified)
        readability = 206.835 - 1.015 * avg_sentence_length - 84.6 * (avg_word_length / 4.7)
        return max(0, min(100, readability)) / 100  # Normalize to 0-1
    
    def _calculate_sentence_variance(self, text: str) -> float:
        """Calculates varianza en longitud de oraciones."""
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        
        if len(sentence_lengths) < 2:
            return 0.0
        
        return np.var(sentence_lengths)
    
    def _extract_language_features(self, text: str) -> Dict[str, Any]:
        """Extrae características lingüísticas del texto."""
        words = text.split()
        
        # POS tag simulation (simplified)
        pos_counts = {
            'nouns': len([w for w in words if w.endswith(('tion', 'ness', 'ment', 'ing'))]),
            'verbs': len([w for w in words if w.endswith(('ed', 'ing', 'es', 's'))]),
            'adjectives': len([w for w in words if w.endswith(('ly', 'ful', 'less', 'ous'))]),
            'adverbs': len([w for w in words if w.endswith('ly')])
        }
        
        # Linguistic complexity
        features = {
            'pos_distribution': pos_counts,
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
            'function_words_ratio': len([w for w in words if w.lower() in 
                                       ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at']]) / len(words) if words else 0,
            'punctuation_density': len(re.findall(r'[.!?,:;]', text)) / len(text) if text else 0,
            'capitalization_ratio': len(re.findall(r'[A-Z]', text)) / len(text) if text else 0
        }
        
        return features

class SentimentAnalysisProcessor:
    """Procesador especializado para análisis de sentimientos."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.sentiment_lexicon = self._load_sentiment_lexicon()
        
        logger.info("😊 SentimentAnalysisProcessor initialized")
    
    def _load_sentiment_lexicon(self) -> Dict[str, float]:
        """Loads léxico de sentimientos."""
        # Léxico básico de sentimientos (en implementación real sería más extenso)
        return {
            # Positive words
            'excellent': 0.8, 'amazing': 0.9, 'wonderful': 0.8, 'great': 0.7,
            'good': 0.6, 'nice': 0.5, 'pleasant': 0.6, 'beautiful': 0.7,
            'fantastic': 0.9, 'awesome': 0.8, 'perfect': 0.9, 'brilliant': 0.8,
            
            # Negative words  
            'terrible': -0.8, 'awful': -0.9, 'horrible': -0.8, 'bad': -0.7,
            'poor': -0.6, 'unpleasant': -0.6, 'ugly': -0.7, 'disappointing': -0.7,
            'frustrated': -0.6, 'angry': -0.8, 'sad': -0.7, 'depressed': -0.9,
            
            # Neutral words
            'okay': 0.1, 'fine': 0.2, 'normal': 0.0, 'average': 0.0
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyzes text sentiment."""
        if not text or len(text.strip()) == 0:
            return {'error': 'Empty text provided'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Calculate sentiment scores
        sentiment_scores = []
        matched_words = []
        
        for word in words:
            if word in self.sentiment_lexicon:
                score = self.sentiment_lexicon[word]
                sentiment_scores.append(score)
                matched_words.append((word, score))
        
        if not sentiment_scores:
            overall_sentiment = 0.0
            confidence = 0.0
        else:
            overall_sentiment = np.mean(sentiment_scores)
            confidence = len(sentiment_scores) / len(words) if words else 0
        
        # Classify sentiment
        if overall_sentiment > 0.2:
            sentiment_label = "positive"
        elif overall_sentiment < -0.2:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return {
            'text': text,
            'overall_sentiment': float(overall_sentiment),
            'sentiment_label': sentiment_label,
            'confidence': float(confidence),
            'matched_words': matched_words,
            'word_count': len(words),
            'sentiment_word_count': len(sentiment_scores),
            'sentiment_distribution': {
                'positive_words': len([s for s in sentiment_scores if s > 0]),
                'negative_words': len([s for s in sentiment_scores if s < 0]),
                'neutral_words': len([s for s in sentiment_scores if s == 0])
            }
        }

class EntityExtractionProcessor:
    """Procesador para extracción de entidades nombradas."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.entity_patterns = self._load_entity_patterns()
        
        logger.info("🏷️ EntityExtractionProcessor initialized")
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Loads patrones para reconocimiento de entidades."""
        return {
            'person': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b(Mr|Mrs|Ms|Dr|Prof)\.? [A-Z][a-z]+\b',  # Title Name
            ],
            'organization': [
                r'\b[A-Z][a-z]+ (Inc|LLC|Corp|Ltd|Company)\b',
                r'\b(Google|Microsoft|Apple|Amazon|Meta|OpenAI)\b',
            ],
            'location': [
                r'\b[A-Z][a-z]+ (City|County|State|Country)\b',
                r'\b(New York|Los Angeles|London|Paris|Tokyo)\b',
            ],
            'date': [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b',
            ],
            'money': [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',
                r'\b\d+ (dollars|euros|pounds|yen)\b',
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ],
            'phone': [
                r'\b\d{3}-\d{3}-\d{4}\b',
                r'\(\d{3}\) \d{3}-\d{4}\b',
            ]
        }
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extrae entidades del texto."""
        if not text or len(text.strip()) == 0:
            return {'error': 'Empty text provided'}
        
        entities = {}
        all_entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            type_entities = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = {
                        'text': match.group(),
                        'type': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8  # Simple confidence score
                    }
                    type_entities.append(entity)
                    all_entities.append(entity)
            
            entities[entity_type] = type_entities
        
        # Calculate statistics
        entity_stats = {
            'total_entities': len(all_entities),
            'entities_by_type': {etype: len(ents) for etype, ents in entities.items()},
            'entity_density': len(all_entities) / len(text.split()) if text.split() else 0,
            'coverage': sum(len(e['text']) for e in all_entities) / len(text) if text else 0
        }
        
        return {
            'text': text,
            'entities': entities,
            'all_entities': all_entities,
            'statistics': entity_stats,
            'processing_timestamp': datetime.now().isoformat()
        }

class CodeAnalysisProcessor:
    """Procesador especializado para análisis de código fuente."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.language_keywords = self._load_language_keywords()
        
        logger.info("💻 CodeAnalysisProcessor initialized")
    
    def _load_language_keywords(self) -> Dict[str, List[str]]:
        """Loads palabras clave por lenguaje de programación."""
        return {
            'python': ['def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 'try', 'except'],
            'javascript': ['function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'try', 'catch'],
            'java': ['public', 'private', 'class', 'interface', 'if', 'else', 'for', 'while', 'try', 'catch'],
            'cpp': ['#include', 'class', 'struct', 'public', 'private', 'if', 'else', 'for', 'while'],
            'sql': ['SELECT', 'FROM', 'WHERE', 'JOIN', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP']
        }
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyzes source code."""
        if not code or len(code.strip()) == 0:
            return {'error': 'Empty code provided'}
        
        # Detect programming language
        detected_language = self._detect_language(code)
        
        # Extract code elements
        functions = self._extract_functions(code, detected_language)
        classes = self._extract_classes(code, detected_language)
        imports = self._extract_imports(code, detected_language)
        comments = self._extract_comments(code)
        
        # Calculate complexity metrics
        complexity = self._calculate_code_complexity(code)
        
        analysis = {
            'code': code,
            'detected_language': detected_language,
            'line_count': len(code.split('\n')),
            'character_count': len(code),
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'comments': comments,
            'complexity': complexity,
            'code_quality': self._assess_code_quality(code, functions, classes, comments),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def _detect_language(self, code: str) -> str:
        """Detecta el lenguaje de programación."""
        language_scores = {}
        
        for language, keywords in self.language_keywords.items():
            score = 0
            for keyword in keywords:
                score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', code, re.IGNORECASE))
            language_scores[language] = score
        
        if not language_scores or max(language_scores.values()) == 0:
            return "unknown"
        
        return max(language_scores.items(), key=lambda x: x[1])[0]
    
    def _extract_functions(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extrae funciones del código."""
        functions = []
        
        if language == 'python':
            pattern = r'def\s+(\w+)\s*\([^)]*\):'
        elif language == 'javascript':
            pattern = r'function\s+(\w+)\s*\([^)]*\)'
        elif language in ['java', 'cpp']:
            pattern = r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*\([^)]*\)'
        else:
            pattern = r'(\w+)\s*\([^)]*\)'
        
        matches = re.finditer(pattern, code, re.MULTILINE)
        for match in matches:
            functions.append({
                'name': match.group(1),
                'start_pos': match.start(),
                'line_number': code[:match.start()].count('\n') + 1
            })
        
        return functions
    
    def _extract_classes(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extracts classes from code."""
        classes = []
        
        if language == 'python':
            pattern = r'class\s+(\w+)(?:\([^)]*\))?:'
        elif language == 'java':
            pattern = r'(?:public|private)?\s*class\s+(\w+)'
        elif language == 'cpp':
            pattern = r'class\s+(\w+)'
        else:
            return classes
        
        matches = re.finditer(pattern, code, re.MULTILINE)
        for match in matches:
            classes.append({
                'name': match.group(1),
                'start_pos': match.start(),
                'line_number': code[:match.start()].count('\n') + 1
            })
        
        return classes
    
    def _extract_imports(self, code: str, language: str) -> List[str]:
        """Extrae imports/includes del código."""
        imports = []
        
        if language == 'python':
            patterns = [r'import\s+(\w+)', r'from\s+(\w+)\s+import']
        elif language == 'javascript':
            patterns = [r'import.*from\s+[\'"]([^\'"]+)[\'"]', r'require\([\'"]([^\'"]+)[\'"]\)']
        elif language in ['java', 'cpp']:
            patterns = [r'#include\s*[<"]([^>"]+)[>"]', r'import\s+([^;]+);']
        else:
            patterns = []
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            imports.extend(matches)
        
        return list(set(imports))  # Remove duplicates
    
    def _extract_comments(self, code: str) -> Dict[str, Any]:
        """Extrae comentarios del código."""
        # Single line comments
        single_line = re.findall(r'//.*$|#.*$', code, re.MULTILINE)
        
        # Multi-line comments
        multi_line = re.findall(r'/\*.*?\*/', code, re.DOTALL)
        multi_line.extend(re.findall(r'""".*?"""', code, re.DOTALL))
        multi_line.extend(re.findall(r"'''.*?'''", code, re.DOTALL))
        
        total_comment_chars = sum(len(c) for c in single_line + multi_line)
        comment_ratio = total_comment_chars / len(code) if code else 0
        
        return {
            'single_line_comments': len(single_line),
            'multi_line_comments': len(multi_line),
            'total_comments': len(single_line) + len(multi_line),
            'comment_ratio': comment_ratio,
            'well_commented': comment_ratio > 0.1
        }
    
    def _calculate_code_complexity(self, code: str) -> Dict[str, Any]:
        """Calculates métricas de complejidad del código."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Cyclomatic complexity (simplified)
        decision_points = len(re.findall(r'\b(if|else|elif|while|for|switch|case)\b', code, re.IGNORECASE))
        
        # Nesting depth
        max_nesting = 0
        current_nesting = 0
        for line in lines:
            stripped = line.strip()
            if stripped.endswith(':') or stripped.endswith('{'):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif stripped in ['}', 'end'] or stripped.startswith(('else', 'elif', 'except', 'finally')):
                current_nesting = max(0, current_nesting - 1)
        
        return {
            'cyclomatic_complexity': decision_points + 1,
            'max_nesting_depth': max_nesting,
            'lines_of_code': len(non_empty_lines),
            'complexity_score': (decision_points + max_nesting) / len(non_empty_lines) if non_empty_lines else 0
        }
    
    def _assess_code_quality(self, code: str, functions: List, classes: List, comments: Dict) -> Dict[str, Any]:
        """Evalúa la calidad del código."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Quality metrics
        has_functions = len(functions) > 0
        has_classes = len(classes) > 0
        well_commented = comments['well_commented']
        reasonable_length = len(non_empty_lines) < 1000  # Not too long
        
        # Calculate quality score
        quality_factors = [has_functions, has_classes, well_commented, reasonable_length]
        quality_score = sum(quality_factors) / len(quality_factors)
        
        return {
            'has_functions': has_functions,
            'has_classes': has_classes,
            'well_commented': well_commented,
            'reasonable_length': reasonable_length,
            'quality_score': quality_score,
            'quality_level': 'high' if quality_score > 0.7 else 'medium' if quality_score > 0.4 else 'low'
        }

class MultimodalFusionProcessor:
    """Procesador para fusión de datos multimodales."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.modality_weights = {'text': 0.4, 'image': 0.3, 'audio': 0.3}
        
        logger.info("🔀 MultimodalFusionProcessor initialized")
    
    def fuse_modalities(self, modality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fusiona datos de múltiples modalidades."""
        if not modality_data:
            return {'error': 'No modality data provided'}
        
        # Normalize each modality
        normalized_data = {}
        fusion_weights = {}
        
        for modality, data in modality_data.items():
            if modality in self.modality_weights:
                # Simulate feature extraction and normalization
                if isinstance(data, str):  # Text
                    features = self._extract_text_features(data)
                elif isinstance(data, (list, np.ndarray)):  # Numerical data
                    features = self._normalize_numerical_features(data)
                else:
                    features = np.random.randn(768).astype(np.float32)  # Fallback
                
                normalized_data[modality] = features
                fusion_weights[modality] = self.modality_weights[modality]
        
        # Weighted fusion
        if normalized_data:
            fused_features = self._weighted_fusion(normalized_data, fusion_weights)
            
            return {
                'fused_features': fused_features,
                'input_modalities': list(normalized_data.keys()),
                'fusion_weights': fusion_weights,
                'feature_dimension': len(fused_features),
                'fusion_quality': self._assess_fusion_quality(normalized_data),
                'processing_timestamp': datetime.now().isoformat()
            }
        else:
            return {'error': 'No valid modalities found'}
    
    def _extract_text_features(self, text: str) -> np.ndarray:
        """Extrae características de texto (simulado)."""
        # Simple text feature extraction
        words = text.split()
        
        # Create feature vector
        features = np.zeros(768, dtype=np.float32)
        
        # Word count features
        features[0] = len(words)
        features[1] = len(set(words))  # Unique words
        features[2] = np.mean([len(word) for word in words]) if words else 0
        
        # Character-level features
        features[3] = len(text)
        features[4] = text.count(' ')
        features[5] = text.count('.')
        
        # Fill remaining with hash-based features
        for i, word in enumerate(words[:100]):  # First 100 words
            feature_idx = (hash(word) % 662) + 6  # Use remaining slots
            features[feature_idx] = 1.0
        
        return features
    
    def _normalize_numerical_features(self, data: Union[List, np.ndarray]) -> np.ndarray:
        """Normaliza características numéricas."""
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        
        # Ensure it's 1D
        if data.ndim > 1:
            data = data.flatten()
        
        # Normalize to 768 dimensions
        if len(data) > 768:
            data = data[:768]
        elif len(data) < 768:
            padding = np.zeros(768 - len(data), dtype=np.float32)
            data = np.concatenate([data, padding])
        
        # Z-score normalization
        if np.std(data) > 0:
            data = (data - np.mean(data)) / np.std(data)
        
        return data
    
    def _weighted_fusion(self, normalized_data: Dict[str, np.ndarray], 
                        weights: Dict[str, float]) -> np.ndarray:
        """Fusión ponderada de características."""
        fused = np.zeros(768, dtype=np.float32)
        total_weight = 0
        
        for modality, features in normalized_data.items():
            weight = weights.get(modality, 1.0)
            fused += weight * features
            total_weight += weight
        
        if total_weight > 0:
            fused /= total_weight
        
        return fused
    
    def _assess_fusion_quality(self, normalized_data: Dict[str, np.ndarray]) -> float:
        """Evalúa la calidad de la fusión."""
        if len(normalized_data) < 2:
            return 0.5  # Single modality
        
        # Calculate correlation between modalities
        modalities = list(normalized_data.values())
        correlations = []
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                corr = np.corrcoef(modalities[i], modalities[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if correlations:
            # Good fusion has moderate correlation (not too high, not too low)
            avg_correlation = np.mean(correlations)
            quality = 1.0 - abs(avg_correlation - 0.3)  # Optimal around 0.3
            return max(0, min(1, quality))
        
        return 0.5

class SpecializedProcessorManager:
    """Manager para coordinar múltiples procesadores especializados."""
    
    def __init__(self):
        self.processors = {}
        self.processing_history = []
        
        logger.info("🎯 SpecializedProcessorManager initialized")
    
    def register_processor(self, name: str, processor_type: ProcessorType, 
                         config: Optional[ProcessorConfig] = None):
        """Registers a specialized processor."""
        if config is None:
            config = ProcessorConfig(processor_type=processor_type)
        
        if processor_type == ProcessorType.TEXT_ANALYSIS:
            processor = TextAnalysisProcessor(config)
        elif processor_type == ProcessorType.SENTIMENT_ANALYSIS:
            processor = SentimentAnalysisProcessor(config)
        elif processor_type == ProcessorType.ENTITY_EXTRACTION:
            processor = EntityExtractionProcessor(config)
        elif processor_type == ProcessorType.CODE_ANALYSIS:
            processor = CodeAnalysisProcessor(config)
        elif processor_type == ProcessorType.MULTIMODAL_FUSION:
            processor = MultimodalFusionProcessor(config)
        else:
            raise ValueError(f"Unsupported processor type: {processor_type}")
        
        self.processors[name] = processor
        logger.info(f"📝 Registered processor: {name} ({processor_type.value})")
    
    def process(self, processor_name: str, data: Any, **kwargs) -> Dict[str, Any]:
        """Processes data using a specific processor."""
        if processor_name not in self.processors:
            return {'error': f'Processor {processor_name} not found'}
        
        processor = self.processors[processor_name]
        
        try:
            start_time = datetime.now()
            
            # Route to appropriate processing method
            if isinstance(processor, TextAnalysisProcessor):
                result = processor.analyze_text(data)
            elif isinstance(processor, SentimentAnalysisProcessor):
                result = processor.analyze_sentiment(data)
            elif isinstance(processor, EntityExtractionProcessor):
                result = processor.extract_entities(data)
            elif isinstance(processor, CodeAnalysisProcessor):
                result = processor.analyze_code(data)
            elif isinstance(processor, MultimodalFusionProcessor):
                result = processor.fuse_modalities(data)
            else:
                return {'error': 'Unknown processor type'}
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Add processing metadata
            result['processing_metadata'] = {
                'processor_name': processor_name,
                'processing_time': processing_time,
                'timestamp': end_time.isoformat()
            }
            
            # Record in history
            self.processing_history.append({
                'processor': processor_name,
                'timestamp': end_time.isoformat(),
                'processing_time': processing_time,
                'success': True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Processing failed in {processor_name}: {e}")
            
            # Record failure in history
            self.processing_history.append({
                'processor': processor_name,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            })
            
            return {'error': str(e), 'processor': processor_name}
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Gets processor statistics."""
        stats = {
            'total_processors': len(self.processors),
            'processor_types': {},
            'processing_history_size': len(self.processing_history),
            'recent_success_rate': self._calculate_recent_success_rate()
        }
        
        # Count processor types
        for name, processor in self.processors.items():
            processor_type = type(processor).__name__
            stats['processor_types'][processor_type] = stats['processor_types'].get(processor_type, 0) + 1
        
        return stats
    
    def _calculate_recent_success_rate(self, window: int = 100) -> float:
        """Calculates tasa de éxito reciente."""
        recent_history = self.processing_history[-window:]
        if not recent_history:
            return 1.0
        
        successful = sum(1 for h in recent_history if h['success'])
        return successful / len(recent_history)

# Factory functions
def create_processor_manager() -> SpecializedProcessorManager:
    """Creates a specialized processor manager."""
    return SpecializedProcessorManager()

def create_default_processors(manager: SpecializedProcessorManager):
    """Creates default processors."""
    default_configs = [
        ('text_analyzer', ProcessorType.TEXT_ANALYSIS),
        ('sentiment_analyzer', ProcessorType.SENTIMENT_ANALYSIS),
        ('entity_extractor', ProcessorType.ENTITY_EXTRACTION),
        ('code_analyzer', ProcessorType.CODE_ANALYSIS),
        ('multimodal_fusion', ProcessorType.MULTIMODAL_FUSION)
    ]
    
    for name, processor_type in default_configs:
        manager.register_processor(name, processor_type)

# Global manager instance
_global_processor_manager: Optional[SpecializedProcessorManager] = None

def get_global_processor_manager() -> SpecializedProcessorManager:
    """Gets global processor manager instance."""
    global _global_processor_manager
    if _global_processor_manager is None:
        _global_processor_manager = create_processor_manager()
        create_default_processors(_global_processor_manager)
    return _global_processor_manager

def main():
    """Main function for testing."""
    logger.info("🔍 Specialized Processors Module - Testing Mode")
    
    # Create manager
    manager = create_processor_manager()
    create_default_processors(manager)
    
    # Test text analysis
    text_result = manager.process('text_analyzer', "This is a wonderful example of text analysis!")
    logger.info(f"Text analysis result: {text_result.get('dominant_pattern', 'N/A')}")
    
    # Test sentiment analysis
    sentiment_result = manager.process('sentiment_analyzer', "I love this amazing product!")
    logger.info(f"Sentiment: {sentiment_result.get('sentiment_label', 'N/A')} ({sentiment_result.get('confidence', 0):.2f})")
    
    # Test entity extraction
    entity_result = manager.process('entity_extractor', "John Doe works at Google Inc. in New York.")
    logger.info(f"Entities found: {entity_result.get('statistics', {}).get('total_entities', 0)}")
    
    # Test code analysis
    code_sample = """
def hello_world():
    print("Hello, World!")
    return True

class MyClass:
    def __init__(self):
        self.value = 42
"""
    code_result = manager.process('code_analyzer', code_sample)
    logger.info(f"Code language: {code_result.get('detected_language', 'N/A')}")
    
    # Show stats
    stats = manager.get_processor_stats()
    logger.info(f"Processor stats: {stats}")
    
    logger.info("✅ Specialized processors testing completed")

if __name__ == "__main__":
    main()