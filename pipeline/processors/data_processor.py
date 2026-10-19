#!/usr/bin/env python3
"""
Data Processor
==============

Processes raw downloaded data into clean, training-ready formats.
Handles text cleaning, tokenization, deduplication, and quality filtering.
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class ProcessingTask:
    """Represents a data processing task."""
    task_id: str
    source_path: str
    target_path: str
    data_type: str  # 'news', 'academic', 'legal', 'general'
    processing_steps: List[str]
    status: str = "pending"
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    input_size_mb: float = 0.0
    output_size_mb: float = 0.0
    records_processed: int = 0
    records_kept: int = 0
    quality_score: float = 0.0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

@dataclass 
class ProcessingStats:
    """Processing statistics."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_input_size_gb: float = 0.0
    total_output_size_gb: float = 0.0
    total_records_processed: int = 0
    total_records_kept: int = 0
    average_quality_score: float = 0.0
    processing_time_seconds: float = 0.0

# Pre-compiled text cleaning patterns (compiled once at module load)
_TEXT_CLEANING_PATTERNS = {
    "html_tags": re.compile(r'<[^>]+>'),
    "urls": re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
    "emails": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "whitespace": re.compile(r'\s+'),
    "excess_punct": re.compile(r'[.]{3,}|[!]{2,}|[?]{2,}'),
    "social_artifacts": re.compile(r'#\w+|@\w+|RT\s|via\s@'),
    "phone_numbers": re.compile(r'(\+34|0034|34)?[\s-]?[6789]\d{2}[\s-]?\d{3}[\s-]?\d{3}'),
}


class TextCleaner:
    """Advanced text cleaning and normalization."""

    def __init__(self):
        self.spanish_patterns = _TEXT_CLEANING_PATTERNS
        
        # Quality indicators
        self.quality_indicators = {
            "min_length": 50,
            "max_length": 10000,
            "min_words": 10,
            "max_duplicate_chars": 0.3,
            "min_letters_ratio": 0.7,
            "max_caps_ratio": 0.3
        }
    
    def clean_text(self, text: str) -> Tuple[str, float]:
        """Clean text and return quality score."""
        if not text or not isinstance(text, str):
            return "", 0.0
        
        original_text = text
        quality_score = 1.0
        
        # Remove HTML tags
        text = self.spanish_patterns["html_tags"].sub(" ", text)
        
        # Remove URLs and emails
        text = self.spanish_patterns["urls"].sub(" ", text)
        text = self.spanish_patterns["emails"].sub(" ", text)
        
        # Remove social media artifacts
        text = self.spanish_patterns["social_artifacts"].sub(" ", text)
        
        # Remove phone numbers
        text = self.spanish_patterns["phone_numbers"].sub(" ", text)
        
        # Normalize excessive punctuation
        text = self.spanish_patterns["excess_punct"].sub("...", text)
        
        # Normalize whitespace
        text = self.spanish_patterns["whitespace"].sub(" ", text).strip()
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(text)
        
        return text, quality_score
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate text quality score (0.0 to 1.0)."""
        if not text:
            return 0.0
        
        score = 1.0
        
        # Length check
        if len(text) < self.quality_indicators["min_length"]:
            score -= 0.3
        elif len(text) > self.quality_indicators["max_length"]:
            score -= 0.2
        
        # Word count check
        words = text.split()
        if len(words) < self.quality_indicators["min_words"]:
            score -= 0.3
        
        # Character diversity
        char_counts = Counter(text.lower())
        total_chars = len(text)
        max_char_ratio = max(char_counts.values()) / total_chars if total_chars > 0 else 0
        if max_char_ratio > self.quality_indicators["max_duplicate_chars"]:
            score -= 0.4
        
        # Letter ratio (vs numbers/symbols)
        letters = sum(1 for c in text if c.isalpha())
        letter_ratio = letters / total_chars if total_chars > 0 else 0
        if letter_ratio < self.quality_indicators["min_letters_ratio"]:
            score -= 0.3
        
        # Caps ratio
        caps = sum(1 for c in text if c.isupper())
        caps_ratio = caps / total_chars if total_chars > 0 else 0
        if caps_ratio > self.quality_indicators["max_caps_ratio"]:
            score -= 0.2
        
        return max(0.0, score)

class DuplicationDetector:
    """Detect and handle duplicate content."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: set = set()
        self.content_fingerprints: Dict[str, str] = {}
    
    def is_duplicate(self, text: str, record_id: str = None) -> bool:
        """Check if text is duplicate."""
        if not text:
            return True
        
        # Generate content hash
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Check exact duplicates
        if content_hash in self.seen_hashes:
            return True
        
        # Generate fingerprint for near-duplicate detection
        fingerprint = self._generate_fingerprint(text)
        
        # Check near-duplicates
        for existing_fp in self.content_fingerprints.values():
            if self._calculate_similarity(fingerprint, existing_fp) > self.similarity_threshold:
                return True
        
        # Store if not duplicate
        self.seen_hashes.add(content_hash)
        if record_id:
            self.content_fingerprints[record_id] = fingerprint
        
        return False
    
    def _generate_fingerprint(self, text: str) -> str:
        """Generate text fingerprint for similarity comparison."""
        # Normalize text
        normalized = re.sub(r'\W+', ' ', text.lower()).strip()
        words = normalized.split()
        
        # Use first 50 words as fingerprint
        fingerprint_words = words[:50] if len(words) > 50 else words
        return ' '.join(fingerprint_words)
    
    def _calculate_similarity(self, fp1: str, fp2: str) -> float:
        """Calculate similarity between two fingerprints."""
        words1 = set(fp1.split())
        words2 = set(fp2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

class DataProcessor:
    """Main data processing orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.text_cleaner = TextCleaner()
        self.duplication_detector = DuplicationDetector()
        
        # Processing configuration
        self.min_quality_score = config.get("processing", {}).get("min_quality_score", 0.6)
        self.batch_size = config.get("processing", {}).get("batch_size", 1000)
        self.enable_deduplication = config.get("processing", {}).get("enable_deduplication", True)
        
        # Storage paths
        self.processed_data_path = Path(config.get("storage", {}).get("processed_data_path", "data/processed"))
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = ProcessingStats()
    
    async def process_spanish_news(self, source_path: str, target_path: str) -> ProcessingTask:
        """Process Spanish news data."""
        task = ProcessingTask(
            task_id=f"news_processing_{int(datetime.now().timestamp())}",
            source_path=source_path,
            target_path=target_path,
            data_type="news",
            processing_steps=["clean_text", "extract_metadata", "deduplicate", "quality_filter"]
        )
        
        logger.info(f" Processing Spanish news: {source_path}")
        task.status = "running"
        task.started_at = datetime.now().isoformat()
        
        try:
            # Read source data
            source_file = Path(source_path)
            if not source_file.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            
            task.input_size_mb = source_file.stat().st_size / (1024 * 1024)
            
            processed_records = []
            records_processed = 0
            records_kept = 0
            total_quality_score = 0.0
            
            # Process data line by line (assuming JSONL format)
            with open(source_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            records_processed += 1
                            
                            # Extract text content
                            text_content = self._extract_news_content(record)
                            
                            if text_content:
                                # Clean text
                                cleaned_text, quality_score = self.text_cleaner.clean_text(text_content)
                                total_quality_score += quality_score
                                
                                # Check quality threshold
                                if quality_score >= self.min_quality_score:
                                    # Check for duplicates
                                    if not self.enable_deduplication or not self.duplication_detector.is_duplicate(cleaned_text):
                                        # Create processed record
                                        processed_record = {
                                            "text": cleaned_text,
                                            "source": record.get("source", "unknown"),
                                            "title": record.get("title", ""),
                                            "date": record.get("date", ""),
                                            "category": record.get("category", "general"),
                                            "quality_score": quality_score,
                                            "processed_at": datetime.now().isoformat()
                                        }
                                        processed_records.append(processed_record)
                                        records_kept += 1
                        
                        except json.JSONDecodeError:
                            continue  # Skip malformed records
            
            # Save processed data
            target_file = Path(target_path)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(target_file, 'w', encoding='utf-8') as f:
                for record in processed_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            task.output_size_mb = target_file.stat().st_size / (1024 * 1024)
            task.records_processed = records_processed
            task.records_kept = records_kept
            task.quality_score = total_quality_score / records_processed if records_processed > 0 else 0.0
            task.status = "completed"
            task.completed_at = datetime.now().isoformat()
            
            logger.info(f" News processing completed: {records_kept}/{records_processed} records kept")
            
        except Exception as e:
            task.status = "failed"
            logger.error(f" News processing failed: {e}")
            raise
        
        return task
    
    async def process_spanish_academic(self, source_path: str, target_path: str) -> ProcessingTask:
        """Process Spanish academic data."""
        task = ProcessingTask(
            task_id=f"academic_processing_{int(datetime.now().timestamp())}",
            source_path=source_path,
            target_path=target_path,
            data_type="academic",
            processing_steps=["clean_text", "extract_citations", "deduplicate", "quality_filter"]
        )
        
        logger.info(f" Processing Spanish academic data: {source_path}")
        task.status = "running"
        task.started_at = datetime.now().isoformat()
        
        try:
            # Similar processing to news but with academic-specific handling
            source_file = Path(source_path)
            if not source_file.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            
            task.input_size_mb = source_file.stat().st_size / (1024 * 1024)
            
            processed_records = []
            records_processed = 0
            records_kept = 0
            total_quality_score = 0.0
            
            # Process academic papers
            with open(source_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            records_processed += 1
                            
                            # Extract academic content
                            text_content = self._extract_academic_content(record)
                            
                            if text_content:
                                # Clean text (academic papers may need different cleaning)
                                cleaned_text, quality_score = self.text_cleaner.clean_text(text_content)
                                total_quality_score += quality_score
                                
                                # Academic papers have higher quality threshold
                                if quality_score >= max(self.min_quality_score, 0.7):
                                    # Check for duplicates
                                    if not self.enable_deduplication or not self.duplication_detector.is_duplicate(cleaned_text):
                                        processed_record = {
                                            "text": cleaned_text,
                                            "title": record.get("title", ""),
                                            "abstract": record.get("abstract", ""),
                                            "authors": record.get("authors", []),
                                            "publication_date": record.get("publication_date", ""),
                                            "source": record.get("source", ""),
                                            "field": record.get("field", ""),
                                            "quality_score": quality_score,
                                            "processed_at": datetime.now().isoformat()
                                        }
                                        processed_records.append(processed_record)
                                        records_kept += 1
                        
                        except json.JSONDecodeError:
                            continue
            
            # Save processed academic data
            target_file = Path(target_path)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(target_file, 'w', encoding='utf-8') as f:
                for record in processed_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            task.output_size_mb = target_file.stat().st_size / (1024 * 1024)
            task.records_processed = records_processed
            task.records_kept = records_kept
            task.quality_score = total_quality_score / records_processed if records_processed > 0 else 0.0
            task.status = "completed"
            task.completed_at = datetime.now().isoformat()
            
            logger.info(f" Academic processing completed: {records_kept}/{records_processed} records kept")
            
        except Exception as e:
            task.status = "failed"
            logger.error(f" Academic processing failed: {e}")
            raise
        
        return task
    
    def _extract_news_content(self, record: Dict[str, Any]) -> str:
        """Extract text content from news record."""
        content_parts = []
        
        # Title
        if record.get("title"):
            content_parts.append(record["title"])
        
        # Main content
        if record.get("content"):
            content_parts.append(record["content"])
        elif record.get("text"):
            content_parts.append(record["text"])
        
        return "\n\n".join(content_parts)
    
    def _extract_academic_content(self, record: Dict[str, Any]) -> str:
        """Extract text content from academic record."""
        content_parts = []
        
        # Title
        if record.get("title"):
            content_parts.append(record["title"])
        
        # Abstract
        if record.get("abstract"):
            content_parts.append(record["abstract"])
        
        # Full text if available
        if record.get("full_text"):
            content_parts.append(record["full_text"])
        elif record.get("content"):
            content_parts.append(record["content"])
        
        return "\n\n".join(content_parts)
    
    async def process_batch(self, raw_data_path: str) -> Dict[str, Any]:
        """Process a batch of raw data files."""
        logger.info(f" Starting batch processing: {raw_data_path}")
        
        processing_tasks = []
        raw_path = Path(raw_data_path)
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data path not found: {raw_data_path}")
        
        # Find all data files to process
        for data_dir in raw_path.iterdir():
            if data_dir.is_dir():
                data_type = data_dir.name
                
                # Process based on data type
                for source_dir in data_dir.iterdir():
                    if source_dir.is_dir():
                        source_name = source_dir.name
                        
                        # Find data files
                        for data_file in source_dir.glob("*.jsonl"):
                            target_path = self.processed_data_path / data_type / source_name / f"processed_{data_file.name}"
                            
                            if "news" in data_type:
                                task = await self.process_spanish_news(str(data_file), str(target_path))
                            elif "academic" in data_type:
                                task = await self.process_spanish_academic(str(data_file), str(target_path))
                            else:
                                # Generic processing
                                task = await self.process_spanish_news(str(data_file), str(target_path))  # Use news processing as default
                            
                            processing_tasks.append(task)
        
        # Update overall statistics
        self.stats.total_tasks = len(processing_tasks)
        self.stats.completed_tasks = sum(1 for task in processing_tasks if task.status == "completed")
        self.stats.failed_tasks = sum(1 for task in processing_tasks if task.status == "failed")
        self.stats.total_input_size_gb = sum(task.input_size_mb for task in processing_tasks) / 1024
        self.stats.total_output_size_gb = sum(task.output_size_mb for task in processing_tasks) / 1024
        self.stats.total_records_processed = sum(task.records_processed for task in processing_tasks)
        self.stats.total_records_kept = sum(task.records_kept for task in processing_tasks)
        
        completed_tasks = [task for task in processing_tasks if task.status == "completed"]
        if completed_tasks:
            self.stats.average_quality_score = sum(task.quality_score for task in completed_tasks) / len(completed_tasks)
        
        result = {
            "processing_summary": asdict(self.stats),
            "task_details": [asdict(task) for task in processing_tasks],
            "processed_at": datetime.now().isoformat()
        }
        
        logger.info(f" Batch processing completed!")
        logger.info(f"   Tasks: {self.stats.completed_tasks}/{self.stats.total_tasks} completed")
        logger.info(f"   Data: {self.stats.total_input_size_gb:.1f}GB → {self.stats.total_output_size_gb:.1f}GB")
        logger.info(f"   Records: {self.stats.total_records_kept}/{self.stats.total_records_processed} kept")
        logger.info(f"   Quality: {self.stats.average_quality_score:.2f}/1.0")
        
        return result

# Example usage
async def demo_data_processing():
    """Demonstrate data processing capabilities."""
    config = {
        "storage": {
            "processed_data_path": "data/processed"
        },
        "processing": {
            "min_quality_score": 0.6,
            "batch_size": 1000,
            "enable_deduplication": True
        }
    }
    
    processor = DataProcessor(config)
    
    # Process batch (assuming raw data exists)
    result = await processor.process_batch("data/raw")
    
    return result

if __name__ == "__main__":
    asyncio.run(demo_data_processing())