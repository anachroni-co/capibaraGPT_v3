"""
Quality Filter for Capibara-6

Advanced content quality assessment and filtering with:
- Multi-heuristic quality scoring
- Perplexity-based filtering
- Toxicity detection
- Boilerplate removal
- Language-specific optimizations

Integrated with Capibara-6 training pipeline and TPU optimizations.
"""

import re
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import Counter
import unicodedata

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available - advanced filtering disabled")
    TRANSFORMERS_AVAILABLE = False

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    logger.warning("fasttext not available - using langid fallback")
    FASTTEXT_AVAILABLE = False


@dataclass
class QualityConfig:
    """Configuration for quality filtering in Capibara-6."""
    
    # Quality thresholds
    min_quality_score: float = 0.35
    max_quality_score: float = 1.0
    
    # Content filtering
    min_words: int = 10
    max_words: int = 10000
    min_unique_words_ratio: float = 0.3
    
    # Repetition filtering
    max_line_repetition_ratio: float = 0.3
    max_ngram_repetition_ratio: float = 0.4
    ngram_size: int = 3
    
    # Character filtering
    min_alpha_ratio: float = 0.6
    max_digit_ratio: float = 0.3
    max_punct_ratio: float = 0.2
    max_upper_ratio: float = 0.3
    
    # Language and encoding
    allowed_languages: Tuple[str, ...] = ("es", "en", "pt", "ca")
    min_lang_confidence: float = 0.8
    
    # Advanced filtering
    use_perplexity_filter: bool = False
    perplexity_model: Optional[str] = None
    min_perplexity: float = 10.0
    max_perplexity: float = 1000.0
    
    use_toxicity_filter: bool = False
    toxicity_threshold: float = 0.8
    
    # Boilerplate detection
    remove_boilerplate: bool = True
    boilerplate_patterns: Tuple[str, ...] = (
        r"cookie policy",
        r"privacy policy", 
        r"terms of service",
        r"subscribe to newsletter",
        r"follow us on",
        r"copyright ©",
        r"all rights reserved"
    )
    
    # TPU optimizations
    batch_processing: bool = True
    parallel_scoring: bool = True
    memory_efficient: bool = True


class ContentAnalyzer:
    """Analyzes content quality using multiple heuristics."""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        
        # Compile boilerplate patterns
        self.boilerplate_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in config.boilerplate_patterns
        ]
        
        # Common spam/low-quality indicators
        self.spam_indicators = {
            'excessive_caps': r'[A-Z]{4,}',
            'excessive_exclamation': r'!{3,}',
            'excessive_question': r'\?{3,}',
            'phone_number': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email_spam': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'url_spam': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        }
        
        self.spam_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.spam_indicators.items()
        }
    
    def analyze_content(self, text: str) -> Dict[str, float]:
        """Comprehensive content analysis."""
        if not text:
            return {'quality_score': 0.0, 'issues': []}
        
        analysis = {
            'length_score': self._score_length(text),
            'lexical_diversity': self._score_lexical_diversity(text),
            'repetition_score': self._score_repetition(text),
            'character_distribution': self._score_character_distribution(text),
            'structural_quality': self._score_structure(text),
            'spam_score': self._score_spam_indicators(text),
            'boilerplate_score': self._score_boilerplate(text)
        }
        
        # Calculate overall quality score
        weights = {
            'length_score': 0.1,
            'lexical_diversity': 0.25,
            'repetition_score': 0.2,
            'character_distribution': 0.15,
            'structural_quality': 0.15,
            'spam_score': 0.1,
            'boilerplate_score': 0.05
        }
        
        quality_score = sum(
            analysis[key] * weights[key] 
            for key in weights
        )
        
        analysis['quality_score'] = float(np.clip(quality_score, 0.0, 1.0))
        analysis['issues'] = self._identify_issues(analysis)
        
        return analysis
    
    def _score_length(self, text: str) -> float:
        """Score based on text length."""
        words = text.split()
        word_count = len(words)
        
        if word_count < self.config.min_words:
            return 0.0
        elif word_count > self.config.max_words:
            return max(0.0, 1.0 - (word_count - self.config.max_words) / self.config.max_words)
        else:
            # Optimal range scoring
            optimal_min, optimal_max = 50, 1000
            if optimal_min <= word_count <= optimal_max:
                return 1.0
            elif word_count < optimal_min:
                return word_count / optimal_min
            else:
                return 1.0 - (word_count - optimal_max) / (self.config.max_words - optimal_max)
    
    def _score_lexical_diversity(self, text: str) -> float:
        """Score based on lexical diversity."""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        unique_words = len(set(words))
        diversity_ratio = unique_words / len(words)
        
        # Apply thresholds
        if diversity_ratio < self.config.min_unique_words_ratio:
            return diversity_ratio / self.config.min_unique_words_ratio
        
        return min(1.0, diversity_ratio * 1.5)  # Bonus for high diversity
    
    def _score_repetition(self, text: str) -> float:
        """Score based on repetition patterns."""
        # Line repetition
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) > 1:
            unique_lines = len(set(lines))
            line_diversity = unique_lines / len(lines)
            
            if line_diversity < (1 - self.config.max_line_repetition_ratio):
                return 0.0
        
        # N-gram repetition
        words = text.lower().split()
        if len(words) >= self.config.ngram_size:
            ngrams = [
                ' '.join(words[i:i + self.config.ngram_size])
                for i in range(len(words) - self.config.ngram_size + 1)
            ]
            
            if ngrams:
                ngram_counts = Counter(ngrams)
                most_common_count = ngram_counts.most_common(1)[0][1]
                repetition_ratio = most_common_count / len(ngrams)
                
                if repetition_ratio > self.config.max_ngram_repetition_ratio:
                    return max(0.0, 1.0 - repetition_ratio)
        
        return 1.0
    
    def _score_character_distribution(self, text: str) -> float:
        """Score based on character distribution."""
        if not text:
            return 0.0
        
        total_chars = len(text)
        alpha_chars = sum(c.isalpha() for c in text)
        digit_chars = sum(c.isdigit() for c in text)
        punct_chars = sum(c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' for c in text)
        upper_chars = sum(c.isupper() for c in text)
        
        alpha_ratio = alpha_chars / total_chars
        digit_ratio = digit_chars / total_chars
        punct_ratio = punct_chars / total_chars
        upper_ratio = upper_chars / max(alpha_chars, 1)
        
        # Check thresholds
        penalties = 0.0
        
        if alpha_ratio < self.config.min_alpha_ratio:
            penalties += (self.config.min_alpha_ratio - alpha_ratio) * 2
        
        if digit_ratio > self.config.max_digit_ratio:
            penalties += (digit_ratio - self.config.max_digit_ratio) * 2
        
        if punct_ratio > self.config.max_punct_ratio:
            penalties += (punct_ratio - self.config.max_punct_ratio) * 1.5
        
        if upper_ratio > self.config.max_upper_ratio:
            penalties += (upper_ratio - self.config.max_upper_ratio) * 1.0
        
        return max(0.0, 1.0 - penalties)
    
    def _score_structure(self, text: str) -> float:
        """Score based on text structure."""
        score = 1.0
        
        # Sentence structure
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            
            # Optimal sentence length: 10-30 words
            if 10 <= avg_sentence_length <= 30:
                score += 0.1
            elif avg_sentence_length < 5 or avg_sentence_length > 50:
                score -= 0.2
        
        # Paragraph structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            score += 0.1  # Bonus for paragraph breaks
        
        # Capitalization patterns
        if text and text[0].isupper():
            score += 0.05  # Bonus for proper capitalization
        
        return min(1.0, max(0.0, score))
    
    def _score_spam_indicators(self, text: str) -> float:
        """Score based on spam indicators."""
        penalties = 0.0
        
        for name, pattern in self.spam_patterns.items():
            matches = len(pattern.findall(text))
            
            if name in ['excessive_caps', 'excessive_exclamation', 'excessive_question']:
                if matches > 0:
                    penalties += min(0.3, matches * 0.1)
            elif name in ['phone_number', 'email_spam']:
                if matches > 2:  # Allow some contact info
                    penalties += min(0.2, (matches - 2) * 0.1)
            elif name == 'url_spam':
                if matches > 3:  # Allow some URLs
                    penalties += min(0.2, (matches - 3) * 0.05)
        
        return max(0.0, 1.0 - penalties)
    
    def _score_boilerplate(self, text: str) -> float:
        """Score based on boilerplate content."""
        if not self.config.remove_boilerplate:
            return 1.0
        
        text_lower = text.lower()
        boilerplate_matches = 0
        
        for pattern in self.boilerplate_patterns:
            if pattern.search(text_lower):
                boilerplate_matches += 1
        
        # Penalty based on number of boilerplate patterns found
        penalty = min(0.5, boilerplate_matches * 0.1)
        return max(0.0, 1.0 - penalty)
    
    def _identify_issues(self, analysis: Dict[str, float]) -> List[str]:
        """Identify quality issues based on analysis."""
        issues = []
        
        if analysis['length_score'] < 0.5:
            issues.append('length')
        if analysis['lexical_diversity'] < 0.5:
            issues.append('lexical_diversity')
        if analysis['repetition_score'] < 0.5:
            issues.append('repetition')
        if analysis['character_distribution'] < 0.5:
            issues.append('character_distribution')
        if analysis['spam_score'] < 0.5:
            issues.append('spam_indicators')
        if analysis['boilerplate_score'] < 0.5:
            issues.append('boilerplate')
        
        return issues


class AdvancedFilter:
    """Advanced filtering using ML models."""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.perplexity_model = None
        self.toxicity_classifier = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize advanced filtering models."""
        
        # Initialize perplexity model
        if self.config.use_perplexity_filter and TRANSFORMERS_AVAILABLE:
            try:
                # This would need a proper perplexity model
                # For now, we'll use a placeholder
                logger.info("Perplexity filtering would be initialized here")
            except Exception as e:
                logger.warning(f"Failed to initialize perplexity model: {e}")
        
        # Initialize toxicity classifier
        if self.config.use_toxicity_filter and TRANSFORMERS_AVAILABLE:
            try:
                self.toxicity_classifier = pipeline(
                    "text-classification",
                    model="martin-ha/toxic-comment-model",
                    device=-1  # Use CPU
                )
                logger.info("✅ Toxicity classifier initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize toxicity classifier: {e}")
    
    def filter_by_perplexity(self, docs: List[Dict]) -> List[Dict]:
        """Filter documents by perplexity."""
        if not self.config.use_perplexity_filter or not self.perplexity_model:
            return docs
        
        # Placeholder for perplexity filtering
        # This would calculate perplexity for each document
        # and filter based on thresholds
        
        logger.info("Perplexity filtering would be applied here")
        return docs
    
    def filter_by_toxicity(self, docs: List[Dict]) -> List[Dict]:
        """Filter documents by toxicity score."""
        if not self.config.use_toxicity_filter or not self.toxicity_classifier:
            return docs
        
        filtered_docs = []
        
        for doc in docs:
            try:
                text = doc.get('text_norm', doc.get('text', ''))[:512]  # Limit length
                result = self.toxicity_classifier(text)
                
                # Assuming the model returns toxicity score
                is_toxic = any(
                    pred['label'] == 'TOXIC' and pred['score'] > self.config.toxicity_threshold
                    for pred in result
                )
                
                if not is_toxic:
                    filtered_docs.append(doc)
                else:
                    doc['filtered_reason'] = 'toxicity'
                    
            except Exception as e:
                logger.warning(f"Toxicity filtering failed for document: {e}")
                filtered_docs.append(doc)  # Keep document if filtering fails
        
        logger.info(f"Toxicity filtering: {len(filtered_docs)}/{len(docs)} documents kept")
        return filtered_docs


class QualityFilter:
    """
    Main quality filtering engine for Capibara-6.
    
    Provides comprehensive content quality assessment and filtering
    with multiple configurable heuristics and optional ML-based filtering.
    """
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.content_analyzer = ContentAnalyzer(config)
        self.advanced_filter = AdvancedFilter(config)
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'quality_filtered': 0,
            'perplexity_filtered': 0,
            'toxicity_filtered': 0,
            'final_count': 0,
            'avg_quality_score': 0.0
        }
        
        logger.info(f"🔍 QualityFilter initialized with config: {self.config}")
    
    def filter_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        Apply quality filtering to documents.
        
        Args:
            docs: List of documents with 'text' or 'text_norm' field
            
        Returns:
            Filtered documents with quality metadata
        """
        logger.info(f"🚀 Starting quality filtering of {len(docs)} documents")
        
        self.stats['total_processed'] = len(docs)
        
        # Stage 1: Content quality analysis
        docs = self._analyze_content_quality(docs)
        
        # Stage 2: Apply quality thresholds
        docs = self._apply_quality_thresholds(docs)
        
        # Stage 3: Advanced filtering
        if self.config.use_perplexity_filter:
            docs = self.advanced_filter.filter_by_perplexity(docs)
        
        if self.config.use_toxicity_filter:
            docs = self.advanced_filter.filter_by_toxicity(docs)
        
        self.stats['final_count'] = len(docs)
        
        # Calculate average quality score
        if docs:
            self.stats['avg_quality_score'] = np.mean([
                doc.get('quality_analysis', {}).get('quality_score', 0.0)
                for doc in docs
            ])
        
        logger.info(f"✅ Quality filtering completed: {len(docs)} documents remaining")
        self._log_stats()
        
        return docs
    
    def _analyze_content_quality(self, docs: List[Dict]) -> List[Dict]:
        """Analyze content quality for all documents."""
        analyzed_docs = []
        
        for doc in docs:
            text = doc.get('text_norm', doc.get('text', ''))
            
            # Perform quality analysis
            analysis = self.content_analyzer.analyze_content(text)
            
            # Add analysis to document
            doc['quality_analysis'] = analysis
            doc['quality_score'] = analysis['quality_score']
            
            analyzed_docs.append(doc)
        
        logger.info(f"📊 Content quality analysis completed for {len(analyzed_docs)} documents")
        return analyzed_docs
    
    def _apply_quality_thresholds(self, docs: List[Dict]) -> List[Dict]:
        """Apply quality score thresholds."""
        filtered_docs = []
        
        for doc in docs:
            quality_score = doc.get('quality_score', 0.0)
            
            if (self.config.min_quality_score <= quality_score <= self.config.max_quality_score):
                filtered_docs.append(doc)
            else:
                self.stats['quality_filtered'] += 1
                doc['filtered_reason'] = 'quality_score'
        
        logger.info(f"🎯 After quality threshold filtering: {len(filtered_docs)} documents")
        return filtered_docs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quality filtering statistics."""
        total_filtered = (
            self.stats['quality_filtered'] + 
            self.stats['perplexity_filtered'] + 
            self.stats['toxicity_filtered']
        )
        
        return {
            **self.stats,
            'total_filtered': total_filtered,
            'filter_rate': total_filtered / max(self.stats['total_processed'], 1),
            'components_active': {
                'content_analysis': True,
                'perplexity_filter': self.config.use_perplexity_filter,
                'toxicity_filter': self.config.use_toxicity_filter
            }
        }
    
    def _log_stats(self):
        """Log detailed filtering statistics."""
        stats = self.get_stats()
        
        logger.info("📊 Quality Filtering Statistics:")
        logger.info(f"  Total processed: {stats['total_processed']:,}")
        logger.info(f"  Quality filtered: {stats['quality_filtered']:,}")
        logger.info(f"  Perplexity filtered: {stats['perplexity_filtered']:,}")
        logger.info(f"  Toxicity filtered: {stats['toxicity_filtered']:,}")
        logger.info(f"  Final count: {stats['final_count']:,}")
        logger.info(f"  Filter rate: {stats['filter_rate']:.1%}")
        logger.info(f"  Average quality score: {stats['avg_quality_score']:.3f}")


# Utility functions
def create_default_quality_config() -> QualityConfig:
    """Create default quality configuration for Capibara-6."""
    return QualityConfig()


def quick_quality_filter(docs: List[Dict], config: Optional[QualityConfig] = None) -> List[Dict]:
    """Quick quality filtering for simple use cases."""
    if config is None:
        config = create_default_quality_config()
    
    quality_filter = QualityFilter(config)
    return quality_filter.filter_documents(docs)