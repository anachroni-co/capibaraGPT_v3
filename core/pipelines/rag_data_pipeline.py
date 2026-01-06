"""
RAG Data Pipeline for document ingestion and processing.

This module handles the ingestion of documents into the RAG system,
including batch processing, format conversion, and metadata extraction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time
import json

from .advanced_rag_pipeline import CapibaraAdvancedRAG

logger = logging.getLogger(__name__)

class RAGDataPipeline:
    """Pipeline for ingesting and processing documents for RAG."""
    
    def __init__(self, rag_system: CapibaraAdvancedRAG):
        self.rag_system = rag_system
        self.supported_formats = {'.txt', '.md', '.json', '.csv'}
        self.processed_files = set()
        
    def ingest_document(self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Ingest a single document into the RAG system.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata to associate with the document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
                
            if file_path.suffix.lower() not in self.supported_formats:
                logger.warning(f"Unsupported format: {file_path.suffix}")
                return False
                
            # Read document content
            text = self._read_document(file_path)
            if not text:
                logger.warning(f"No content extracted from: {file_path}")
                return False
                
            # Prepare metadata
            doc_metadata = {
                "filename": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "format": file_path.suffix.lower(),
                "ingestion_time": time.time()
            }
            
            if metadata:
                doc_metadata.update(metadata)
                
            # Add to RAG system
            self.rag_system.add_document(text, doc_metadata, compress_immediately=True)
            self.processed_files.add(str(file_path))
            
            logger.info(f"Successfully ingested: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return False
            
    def ingest_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> Dict[str, Any]:
        """
        Ingest all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            
        Returns:
            Dictionary with ingestion statistics
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return {"success": False, "error": "Directory not found"}
            
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        all_files = list(directory_path.glob(pattern))
        
        supported_files = [
            f for f in all_files 
            if f.is_file() and f.suffix.lower() in self.supported_formats
        ]
        
        # Process files
        successful = 0
        failed = 0
        start_time = time.time()
        
        for file_path in supported_files:
            if self.ingest_document(file_path):
                successful += 1
            else:
                failed += 1
                
        processing_time = time.time() - start_time
        
        stats = {
            "success": True,
            "total_files_found": len(supported_files),
            "successful_ingestions": successful,
            "failed_ingestions": failed,
            "processing_time": processing_time,
            "files_per_second": successful / max(processing_time, 0.001)
        }
        
        logger.info(f"Directory ingestion complete: {stats}")
        return stats
        
    def ingest_batch(self, file_paths: List[Union[str, Path]], metadata_list: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Ingest a batch of documents.
        
        Args:
            file_paths: List of file paths to ingest
            metadata_list: Optional list of metadata dictionaries (one per file)
            
        Returns:
            Dictionary with batch ingestion statistics
        """
        if metadata_list and len(metadata_list) != len(file_paths):
            logger.error("Metadata list length must match file paths length")
            return {"success": False, "error": "Metadata length mismatch"}
            
        successful = 0
        failed = 0
        start_time = time.time()
        
        for i, file_path in enumerate(file_paths):
            metadata = metadata_list[i] if metadata_list else None
            
            if self.ingest_document(file_path, metadata):
                successful += 1
            else:
                failed += 1
                
        processing_time = time.time() - start_time
        
        stats = {
            "success": True,
            "total_files": len(file_paths),
            "successful_ingestions": successful,
            "failed_ingestions": failed,
            "processing_time": processing_time,
            "files_per_second": successful / max(processing_time, 0.001)
        }
        
        logger.info(f"Batch ingestion complete: {stats}")
        return stats
        
    def _read_document(self, file_path: Path) -> str:
        """Read document content based on file format."""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix in {'.txt', '.md'}:
                return self._read_text_file(file_path)
            elif suffix == '.json':
                return self._read_json_file(file_path)
            elif suffix == '.csv':
                return self._read_csv_file(file_path)
            else:
                logger.warning(f"Unsupported format for reading: {suffix}")
                return ""
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
            
    def _read_text_file(self, file_path: Path) -> str:
        """Read plain text or markdown file."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
                
        logger.error(f"Could not decode file with any encoding: {file_path}")
        return ""
        
    def _read_json_file(self, file_path: Path) -> str:
        """Read JSON file and convert to text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert JSON to readable text
            if isinstance(data, dict):
                text_parts = []
                for key, value in data.items():
                    if isinstance(value, (str, int, float)):
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, list):
                        text_parts.append(f"{key}: {', '.join(map(str, value))}")
                    else:
                        text_parts.append(f"{key}: {str(value)}")
                return "\n".join(text_parts)
            elif isinstance(data, list):
                return "\n".join(str(item) for item in data)
            else:
                return str(data)
                
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return ""
            
    def _read_csv_file(self, file_path: Path) -> str:
        """Read CSV file and convert to text."""
        try:
            # Simple CSV reading without pandas dependency
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if not lines:
                return ""
                
            # Convert CSV to readable text
            text_parts = []
            header = lines[0].strip().split(',')
            
            for line in lines[1:]:
                values = line.strip().split(',')
                if len(values) == len(header):
                    row_text = []
                    for h, v in zip(header, values):
                        row_text.append(f"{h.strip()}: {v.strip()}")
                    text_parts.append("; ".join(row_text))
                    
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return ""
            
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics about the ingestion process."""
        return {
            "total_processed_files": len(self.processed_files),
            "supported_formats": list(self.supported_formats),
            "rag_system_stats": self.rag_system.get_stats()
        }
        
    def clear_processed_files(self):
        """Clear the list of processed files."""
        self.processed_files.clear()
        logger.info("Cleared processed files list")

class DocumentValidator:
    """Validator for documents before ingestion."""
    
    def __init__(self, min_content_length: int = 50, max_content_length: int = 1_000_000):
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        
    def validate_document(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a document before ingestion.
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Check content length
        if len(text) < self.min_content_length:
            issues.append(f"Content too short: {len(text)} < {self.min_content_length}")
        elif len(text) > self.max_content_length:
            warnings.append(f"Content very long: {len(text)} > {self.max_content_length}")
            
        # Check for mostly non-text content
        non_text_chars = sum(1 for c in text if not c.isprintable() and c not in '\n\r\t')
        if non_text_chars > len(text) * 0.1:
            warnings.append(f"High non-text character ratio: {non_text_chars/len(text):.2%}")
            
        # Check for required metadata
        required_fields = ['filename', 'path']
        for field in required_fields:
            if field not in metadata:
                issues.append(f"Missing required metadata field: {field}")
                
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "content_length": len(text),
            "metadata_fields": list(metadata.keys())
        }