#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Data Pipeline - CapibaraGPT-v2

Complete unified data processing pipeline for cascade training system.
Handles preprocessing, validation, and integration of datasets across all stages.
"""

import os
import json
import logging
import asyncio
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import time
import re
from collections import defaultdict

# Data processing libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import gzip
import bz2
import lzma

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline."""
    # General settings
    max_workers: int = multiprocessing.cpu_count()
    chunk_size: int = 10000
    max_file_size_gb: float = 10.0
    
    # Text processing
    min_text_length: int = 50
    max_text_length: int = 10000
    remove_html: bool = True
    remove_urls: bool = True
    normalize_whitespace: bool = True
    
    # Quality filters
    min_quality_score: float = 0.7
    deduplication_threshold: float = 0.95
    language_detection: bool = True
    content_filtering: bool = True
    
    # Output settings
    output_format: str = "jsonl"
    compression: str = "gzip"
    include_metadata: bool = True
    validation_checksums: bool = True

@dataclass
class ProcessingResult:
    """Result of data processing operation."""
    dataset_name: str
    stage: str
    category: str
    input_files: List[str]
    output_file: str
    processed_count: int
    filtered_count: int
    error_count: int
    processing_time: float
    output_size_gb: float
    quality_score: float
    checksum: Optional[str] = None
    errors: List[str] = None

class UnifiedDataPipeline:
    """
    Unified data processing pipeline for CapibaraGPT-v2 cascade training.
    
    Features:
    - Multi-format support (JSONL, CSV, XML, HTML, etc.)
    - Parallel processing
    - Quality filtering and validation
    - Deduplication
    - Stage-specific processing
    - Progress tracking
    """
    
    def __init__(self, base_dir: Union[str, Path] = "data/cascade_datasets", 
                 config: Optional[ProcessingConfig] = None):
        """Initialize the unified data pipeline."""
        self.base_dir = Path(base_dir)
        self.config = config or ProcessingConfig()
        
        # Directory structure
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.validated_dir = self.base_dir / "validated"
        self.final_dir = self.base_dir / "final"
        
        for dir_path in [self.processed_dir, self.validated_dir, self.final_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Processing state
        self.processing_results = {}
        self.quality_metrics = defaultdict(list)
        
        logger.info(f"Unified Data Pipeline initialized at {self.base_dir}")
    
    def detect_file_format(self, file_path: Path) -> str:
        """Detect the format of a file based on extension and content."""
        if not file_path.exists():
            return "unknown"
        
        # Check extension first
        ext = file_path.suffix.lower()
        if ext in ['.jsonl', '.json']:
            return 'jsonl'
        elif ext in ['.csv']:
            return 'csv'
        elif ext in ['.xml']:
            return 'xml'
        elif ext in ['.html', '.htm']:
            return 'html'
        elif ext in ['.txt']:
            return 'text'
        elif ext in ['.mbox']:
            return 'mbox'
        elif ext in ['.parquet']:
            return 'parquet'
        
        # Try to detect from content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
                if first_line.startswith('{') or first_line.startswith('['):
                    return 'jsonl'
                elif first_line.startswith('<'):
                    return 'xml'
                elif ',' in first_line and len(first_line.split(',')) > 3:
                    return 'csv'
                else:
                    return 'text'
        except:
            return 'unknown'
    
    def load_file(self, file_path: Path, format_type: str) -> List[Dict[str, Any]]:
        """Load data from file based on format."""
        data = []
        
        try:
            if format_type == 'jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            
            elif format_type == 'csv':
                df = pd.read_csv(file_path)
                data = df.to_dict('records')
            
            elif format_type == 'xml':
                # Simple XML parsing - in practice, use proper XML parser
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract text content (simplified)
                    text_content = re.sub(r'<[^>]+>', '', content)
                    data.append({"text": text_content, "source": str(file_path)})
            
            elif format_type == 'html':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract text content (simplified)
                    text_content = re.sub(r'<[^>]+>', '', content)
                    data.append({"text": text_content, "source": str(file_path)})
            
            elif format_type == 'text':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    data.append({"text": content, "source": str(file_path)})
            
            elif format_type == 'mbox':
                # Simple mbox parsing
                with open(file_path, 'r', encoding='utf-8') as f:
                    current_message = []
                    for line in f:
                        if line.startswith('From '):
                            if current_message:
                                data.append({"text": ''.join(current_message), "source": str(file_path)})
                            current_message = [line]
                        else:
                            current_message.append(line)
                    if current_message:
                        data.append({"text": ''.join(current_message), "source": str(file_path)})
            
            elif format_type == 'parquet':
                df = pd.read_parquet(file_path)
                data = df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
        
        return data
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text content."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        if self.config.remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        if self.config.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
    
    def calculate_quality_score(self, item: Dict[str, Any]) -> float:
        """Calculate quality score for a data item."""
        score = 0.0
        
        # Text length score
        text = item.get('text', '')
        if text:
            length = len(text)
            if self.config.min_text_length <= length <= self.config.max_text_length:
                score += 0.3
            elif length > self.config.max_text_length:
                score += 0.2  # Penalty for very long text
        
        # Content quality indicators
        if text:
            # Check for meaningful content (not just whitespace/punctuation)
            meaningful_chars = len(re.sub(r'[\s\W]', '', text))
            if meaningful_chars > 20:
                score += 0.3
            
            # Check for diversity (not repetitive)
            words = text.split()
            if len(set(words)) > len(words) * 0.5:
                score += 0.2
            
            # Check for proper formatting
            if text.count('.') > 0 and text.count(' ') > 10:
                score += 0.2
        
        return min(score, 1.0)
    
    def filter_item(self, item: Dict[str, Any]) -> bool:
        """Filter data item based on quality criteria."""
        # Calculate quality score
        quality_score = self.calculate_quality_score(item)
        
        # Check minimum quality threshold
        if quality_score < self.config.min_quality_score:
            return False
        
        # Check text length
        text = item.get('text', '')
        if len(text) < self.config.min_text_length:
            return False
        
        # Content filtering
        if self.config.content_filtering:
            # Filter out common low-quality patterns
            low_quality_patterns = [
                r'^\s*$',  # Empty or whitespace only
                r'^[^\w]*$',  # No alphanumeric characters
                r'^(.)\1+$',  # Repeated characters
            ]
            
            for pattern in low_quality_patterns:
                if re.match(pattern, text):
                    return False
        
        return True
    
    def process_chunk(self, chunk: List[Dict[str, Any]], dataset_name: str) -> Tuple[List[Dict[str, Any]], int, int]:
        """Process a chunk of data items."""
        processed_items = []
        filtered_count = 0
        error_count = 0
        
        for item in chunk:
            try:
                # Preprocess text
                if 'text' in item:
                    item['text'] = self.preprocess_text(item['text'])
                
                # Add metadata
                if self.config.include_metadata:
                    item['metadata'] = {
                        'dataset': dataset_name,
                        'processed_at': time.time(),
                        'quality_score': self.calculate_quality_score(item)
                    }
                
                # Filter based on quality
                if self.filter_item(item):
                    processed_items.append(item)
                else:
                    filtered_count += 1
                    
            except Exception as e:
                error_count += 1
                logger.debug(f"Error processing item: {e}")
        
        return processed_items, filtered_count, error_count
    
    def process_dataset(self, dataset_path: Path, dataset_name: str, stage: str, category: str) -> ProcessingResult:
        """Process a single dataset."""
        start_time = time.time()
        
        # Detect format
        format_type = self.detect_file_format(dataset_path)
        logger.info(f"Processing {dataset_name} ({format_type} format)")
        
        # Load data
        raw_data = self.load_file(dataset_path, format_type)
        if not raw_data:
            logger.warning(f"No data loaded from {dataset_path}")
            return ProcessingResult(
                dataset_name=dataset_name,
                stage=stage,
                category=category,
                input_files=[str(dataset_path)],
                output_file="",
                processed_count=0,
                filtered_count=0,
                error_count=0,
                processing_time=time.time() - start_time,
                output_size_gb=0.0,
                quality_score=0.0
            )
        
        # Process in chunks
        processed_items = []
        total_filtered = 0
        total_errors = 0
        
        chunks = [raw_data[i:i + self.config.chunk_size] 
                 for i in range(0, len(raw_data), self.config.chunk_size)]
        
        for chunk in tqdm(chunks, desc=f"Processing {dataset_name}"):
            chunk_processed, chunk_filtered, chunk_errors = self.process_chunk(chunk, dataset_name)
            processed_items.extend(chunk_processed)
            total_filtered += chunk_filtered
            total_errors += chunk_errors
        
        # Calculate quality metrics
        quality_scores = [item.get('metadata', {}).get('quality_score', 0) for item in processed_items]
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # Save processed data
        output_dir = self.processed_dir / stage / category
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{dataset_name.replace(' ', '_').lower()}.jsonl"
        
        # Write with compression if specified
        if self.config.compression == 'gzip':
            output_file = output_file.with_suffix('.jsonl.gz')
            open_func = gzip.open
            mode = 'wt'
        elif self.config.compression == 'bz2':
            output_file = output_file.with_suffix('.jsonl.bz2')
            open_func = bz2.open
            mode = 'wt'
        elif self.config.compression == 'lzma':
            output_file = output_file.with_suffix('.jsonl.xz')
            open_func = lzma.open
            mode = 'wt'
        else:
            open_func = open
            mode = 'w'
        
        with open_func(output_file, mode, encoding='utf-8') as f:
            for item in processed_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Calculate output size
        output_size_gb = output_file.stat().st_size / (1024**3)
        
        # Calculate checksum if requested
        checksum = None
        if self.config.validation_checksums:
            checksum = self.calculate_file_checksum(output_file)
        
        processing_time = time.time() - start_time
        
        result = ProcessingResult(
            dataset_name=dataset_name,
            stage=stage,
            category=category,
            input_files=[str(dataset_path)],
            output_file=str(output_file),
            processed_count=len(processed_items),
            filtered_count=total_filtered,
            error_count=total_errors,
            processing_time=processing_time,
            output_size_gb=output_size_gb,
            quality_score=avg_quality,
            checksum=checksum
        )
        
        logger.info(f"✅ Processed {dataset_name}: {len(processed_items)} items, "
                   f"{total_filtered} filtered, {total_errors} errors, "
                   f"quality: {avg_quality:.3f}, time: {processing_time:.1f}s")
        
        return result
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def process_stage(self, stage_name: str) -> Dict[str, ProcessingResult]:
        """Process all datasets for a specific stage."""
        logger.info(f"🚀 Processing stage: {stage_name}")
        
        stage_dir = self.raw_dir / stage_name
        if not stage_dir.exists():
            logger.warning(f"Stage directory not found: {stage_dir}")
            return {}
        
        results = {}
        
        # Process each category
        for category_dir in stage_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category = category_dir.name
            logger.info(f"📁 Processing category: {category}")
            
            # Process each dataset file
            for dataset_file in category_dir.iterdir():
                if not dataset_file.is_file():
                    continue
                
                dataset_name = dataset_file.stem.replace('_', ' ').title()
                
                try:
                    result = self.process_dataset(dataset_file, dataset_name, stage_name, category)
                    results[f"{category}_{dataset_name}"] = result
                    
                    # Store quality metrics
                    self.quality_metrics[stage_name].append(result.quality_score)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {dataset_name}: {e}")
        
        logger.info(f"✅ Stage {stage_name} processing completed: {len(results)} datasets")
        return results
    
    def process_all_stages(self) -> Dict[str, Dict[str, ProcessingResult]]:
        """Process all stages."""
        logger.info("🚀 Starting processing of all stages")
        
        all_results = {}
        
        # Get all stage directories
        stage_dirs = [d for d in self.raw_dir.iterdir() if d.is_dir() and d.name.startswith('stage_')]
        
        for stage_dir in sorted(stage_dirs):
            stage_name = stage_dir.name
            try:
                results = self.process_stage(stage_name)
                all_results[stage_name] = results
                
                # Add delay between stages
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error processing stage {stage_name}: {e}")
                all_results[stage_name] = {}
        
        return all_results
    
    def validate_processed_data(self, stage_name: str) -> Dict[str, bool]:
        """Validates processed data for a stage."""
        logger.info(f"🔍 Validating processed data for {stage_name}")
        
        validation_results = {}
        processed_dir = self.processed_dir / stage_name
        
        if not processed_dir.exists():
            logger.warning(f"Processed directory not found: {processed_dir}")
            return {}
        
        for category_dir in processed_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            for file_path in category_dir.iterdir():
                if not file_path.is_file():
                    continue
                
                try:
                    # Basic validation
                    is_valid = self.validate_file(file_path)
                    validation_results[str(file_path)] = is_valid
                    
                    if is_valid:
                        logger.info(f"✅ Validated: {file_path.name}")
                    else:
                        logger.warning(f"⚠️ Validation failed: {file_path.name}")
                        
                except Exception as e:
                    logger.error(f"❌ Error validating {file_path}: {e}")
                    validation_results[str(file_path)] = False
        
        return validation_results
    
    def validate_file(self, file_path: Path) -> bool:
        """Validates a processed file."""
        try:
            # Check file size
            if file_path.stat().st_size == 0:
                return False
            
            # Check format (JSONL)
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            json.loads(line)
                            line_count += 1
                            if line_count > 10:  # Check first 10 lines
                                break
                        except json.JSONDecodeError:
                            return False
            
            return line_count > 0
            
        except Exception:
            return False
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results."""
        summary = {
            "total_stages": len(self.processing_results),
            "total_datasets": sum(len(results) for results in self.processing_results.values()),
            "total_processed": 0,
            "total_filtered": 0,
            "total_errors": 0,
            "total_size_gb": 0.0,
            "avg_quality": 0.0,
            "stages": {}
        }
        
        all_quality_scores = []
        
        for stage_name, results in self.processing_results.items():
            stage_summary = {
                "datasets": len(results),
                "processed": 0,
                "filtered": 0,
                "errors": 0,
                "size_gb": 0.0,
                "avg_quality": 0.0
            }
            
            stage_quality_scores = []
            
            for result in results.values():
                stage_summary["processed"] += result.processed_count
                stage_summary["filtered"] += result.filtered_count
                stage_summary["errors"] += result.error_count
                stage_summary["size_gb"] += result.output_size_gb
                stage_quality_scores.append(result.quality_score)
            
            if stage_quality_scores:
                stage_summary["avg_quality"] = np.mean(stage_quality_scores)
                all_quality_scores.extend(stage_quality_scores)
            
            summary["stages"][stage_name] = stage_summary
            summary["total_processed"] += stage_summary["processed"]
            summary["total_filtered"] += stage_summary["filtered"]
            summary["total_errors"] += stage_summary["errors"]
            summary["total_size_gb"] += stage_summary["size_gb"]
        
        if all_quality_scores:
            summary["avg_quality"] = np.mean(all_quality_scores)
        
        return summary
    
    def save_processing_report(self, output_file: Optional[Path] = None):
        """Save processing report to file."""
        if output_file is None:
            output_file = self.base_dir / "processing_report.json"
        
        report = {
            "timestamp": time.time(),
            "config": asdict(self.config),
            "summary": self.get_processing_summary(),
            "results": {
                stage: {name: asdict(result) for name, result in results.items()}
                for stage, results in self.processing_results.items()
            },
            "quality_metrics": dict(self.quality_metrics)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Processing report saved to {output_file}")

def create_unified_pipeline(base_dir: Optional[str] = None, 
                          config: Optional[ProcessingConfig] = None) -> UnifiedDataPipeline:
    """Creates a unified data pipeline instance."""
    if base_dir is None:
        base_dir = "data/cascade_datasets"
    
    return UnifiedDataPipeline(base_dir, config)

def main():
    """Main function for testing the unified data pipeline."""
    logger.info("🦫 CapibaraGPT-v2 Unified Data Pipeline")
    
    # Create pipeline
    pipeline = create_unified_pipeline()
    
    # Process all stages
    results = pipeline.process_all_stages()
    
    # Save report
    pipeline.save_processing_report()
    
    # Show summary
    summary = pipeline.get_processing_summary()
    logger.info("📊 Processing Summary:")
    logger.info(f"  Total stages: {summary['total_stages']}")
    logger.info(f"  Total datasets: {summary['total_datasets']}")
    logger.info(f"  Total processed: {summary['total_processed']:,}")
    logger.info(f"  Total filtered: {summary['total_filtered']:,}")
    logger.info(f"  Total errors: {summary['total_errors']:,}")
    logger.info(f"  Total size: {summary['total_size_gb']:.2f}GB")
    logger.info(f"  Average quality: {summary['avg_quality']:.3f}")
    
    return True

if __name__ == "__main__":
    main()
