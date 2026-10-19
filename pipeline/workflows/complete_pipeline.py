#!/usr/bin/env python3
"""
Complete Data-to-Training Pipeline
==================================

Orchestrates the complete workflow from data download/scraping to training-ready datasets.
Integrates with the training module for seamless model training.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import time

# Import pipeline components
from ..downloaders.download_orchestrator import DownloadOrchestrator
from ..processors.data_processor import DataProcessor

logger = logging.getLogger(__name__)

@dataclass
class PipelineStage:
    """Represents a pipeline stage."""
    stage_name: str
    status: str = "pending"  # pending, running, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    output_path: Optional[str] = None
    metrics: Dict[str, Any] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    pipeline_id: str
    total_duration_seconds: float
    stages: List[PipelineStage]
    final_dataset_path: str
    dataset_stats: Dict[str, Any]
    ready_for_training: bool
    completed_at: str

class CompletePipeline:
    """Orchestrates the complete data-to-training pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline_id = f"pipeline_{int(datetime.now().timestamp())}"
        
        # Initialize components
        self.download_orchestrator = DownloadOrchestrator(config)
        self.data_processor = DataProcessor(config)
        
        # Pipeline stages
        self.stages = [
            PipelineStage("data_download"),
            PipelineStage("data_processing"),
            PipelineStage("dataset_preparation"),
            PipelineStage("training_integration")
        ]
        
        # Storage paths
        self.raw_data_path = Path(config.get("storage", {}).get("raw_data_path", "data/raw"))
        self.processed_data_path = Path(config.get("storage", {}).get("processed_data_path", "data/processed"))
        self.training_data_path = Path(config.get("storage", {}).get("training_data_path", "data/training"))
        
        # Create directories
        for path in [self.raw_data_path, self.processed_data_path, self.training_data_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Pipeline start time
        self.start_time = None
    
    async def execute_complete_pipeline(self) -> PipelineResult:
        """Execute the complete pipeline from data acquisition to training preparation."""
        logger.info(f" Starting complete pipeline: {self.pipeline_id}")
        self.start_time = time.time()
        
        try:
            # Stage 1: Data Download
            await self._execute_download_stage()
            
            # Stage 2: Data Processing
            await self._execute_processing_stage()
            
            # Stage 3: Dataset Preparation
            await self._execute_dataset_preparation_stage()
            
            # Stage 4: Training Integration
            await self._execute_training_integration_stage()
            
            # Create final result
            total_duration = time.time() - self.start_time
            
            # Calculate final dataset statistics
            dataset_stats = await self._calculate_final_dataset_stats()
            
            result = PipelineResult(
                pipeline_id=self.pipeline_id,
                total_duration_seconds=total_duration,
                stages=self.stages,
                final_dataset_path=str(self.training_data_path),
                dataset_stats=dataset_stats,
                ready_for_training=all(stage.status == "completed" for stage in self.stages),
                completed_at=datetime.now().isoformat()
            )
            
            logger.info(f" Complete pipeline finished!")
            logger.info(f"   Duration: {total_duration:.1f} seconds")
            logger.info(f"   Dataset ready: {result.ready_for_training}")
            logger.info(f"   Final dataset: {result.final_dataset_path}")
            
            return result
            
        except Exception as e:
            logger.error(f" Pipeline failed: {e}")
            # Mark current stage as failed
            current_stage = next((s for s in self.stages if s.status == "running"), None)
            if current_stage:
                current_stage.status = "failed"
                current_stage.error_message = str(e)
                current_stage.completed_at = datetime.now().isoformat()
            raise
    
    async def _execute_download_stage(self):
        """Execute data download stage."""
        stage = self.stages[0]  # data_download
        stage.status = "running"
        stage.started_at = datetime.now().isoformat()
        stage_start_time = time.time()
        
        logger.info(" Stage 1: Data Download")
        
        try:
            # Setup download pipeline
            pipeline_info = self.download_orchestrator.setup_complete_download_pipeline()
            
            # Execute downloads
            download_result = await self.download_orchestrator.run_download_pipeline()
            
            # Update stage
            stage.status = "completed"
            stage.completed_at = datetime.now().isoformat()
            stage.duration_seconds = time.time() - stage_start_time
            stage.output_path = str(self.raw_data_path)
            stage.metrics = {
                "setup_info": pipeline_info,
                "download_result": download_result,
                "total_downloaded_gb": download_result.get("total_data_downloaded_gb", 0),
                "success_rate": download_result.get("success_rate", 0)
            }
            
            logger.info(f" Download stage completed: {stage.metrics['total_downloaded_gb']:.1f}GB downloaded")
            
        except Exception as e:
            stage.status = "failed"
            stage.error_message = str(e)
            stage.completed_at = datetime.now().isoformat()
            stage.duration_seconds = time.time() - stage_start_time
            raise
    
    async def _execute_processing_stage(self):
        """Execute data processing stage."""
        stage = self.stages[1]  # data_processing
        stage.status = "running"
        stage.started_at = datetime.now().isoformat()
        stage_start_time = time.time()
        
        logger.info(" Stage 2: Data Processing")
        
        try:
            # Process downloaded data
            processing_result = await self.data_processor.process_batch(str(self.raw_data_path))
            
            # Update stage
            stage.status = "completed"
            stage.completed_at = datetime.now().isoformat()
            stage.duration_seconds = time.time() - stage_start_time
            stage.output_path = str(self.processed_data_path)
            stage.metrics = {
                "processing_result": processing_result,
                "records_processed": processing_result.get("processing_summary", {}).get("total_records_processed", 0),
                "records_kept": processing_result.get("processing_summary", {}).get("total_records_kept", 0),
                "average_quality": processing_result.get("processing_summary", {}).get("average_quality_score", 0),
                "data_reduction_ratio": self._calculate_data_reduction_ratio(processing_result)
            }
            
            logger.info(f" Processing stage completed: {stage.metrics['records_kept']} records kept")
            
        except Exception as e:
            stage.status = "failed"
            stage.error_message = str(e)
            stage.completed_at = datetime.now().isoformat()
            stage.duration_seconds = time.time() - stage_start_time
            raise
    
    async def _execute_dataset_preparation_stage(self):
        """Execute dataset preparation stage."""
        stage = self.stages[2]  # dataset_preparation
        stage.status = "running"
        stage.started_at = datetime.now().isoformat()
        stage_start_time = time.time()
        
        logger.info(" Stage 3: Dataset Preparation")
        
        try:
            # Prepare training datasets
            preparation_result = await self._prepare_training_datasets()
            
            # Update stage
            stage.status = "completed"
            stage.completed_at = datetime.now().isoformat()
            stage.duration_seconds = time.time() - stage_start_time
            stage.output_path = str(self.training_data_path)
            stage.metrics = preparation_result
            
            logger.info(f" Dataset preparation completed: {len(preparation_result.get('datasets', []))} datasets ready")
            
        except Exception as e:
            stage.status = "failed"
            stage.error_message = str(e)
            stage.completed_at = datetime.now().isoformat()
            stage.duration_seconds = time.time() - stage_start_time
            raise
    
    async def _execute_training_integration_stage(self):
        """Execute training integration stage."""
        stage = self.stages[3]  # training_integration
        stage.status = "running"
        stage.started_at = datetime.now().isoformat()
        stage_start_time = time.time()
        
        logger.info(" Stage 4: Training Integration")
        
        try:
            # Integrate with training module
            integration_result = await self._integrate_with_training()
            
            # Update stage
            stage.status = "completed"
            stage.completed_at = datetime.now().isoformat()
            stage.duration_seconds = time.time() - stage_start_time
            stage.output_path = str(self.training_data_path)
            stage.metrics = integration_result
            
            logger.info(f" Training integration completed: Ready for model training")
            
        except Exception as e:
            stage.status = "failed"
            stage.error_message = str(e)
            stage.completed_at = datetime.now().isoformat()
            stage.duration_seconds = time.time() - stage_start_time
            raise
    
    async def _prepare_training_datasets(self) -> Dict[str, Any]:
        """Prepare training-ready datasets from processed data."""
        logger.info(" Preparing training datasets...")
        
        # Find all processed data files
        processed_files = []
        for data_type_dir in self.processed_data_path.iterdir():
            if data_type_dir.is_dir():
                for source_dir in data_type_dir.iterdir():
                    if source_dir.is_dir():
                        for processed_file in source_dir.glob("processed_*.jsonl"):
                            processed_files.append({
                                "path": processed_file,
                                "data_type": data_type_dir.name,
                                "source": source_dir.name
                            })
        
        # Group by data type for training dataset creation
        datasets_by_type = {}
        for file_info in processed_files:
            data_type = file_info["data_type"]
            if data_type not in datasets_by_type:
                datasets_by_type[data_type] = []
            datasets_by_type[data_type].append(file_info)
        
        # Create training datasets
        training_datasets = []
        total_training_records = 0
        
        for data_type, files in datasets_by_type.items():
            # Combine all files of the same type
            combined_records = []
            
            for file_info in files:
                with open(file_info["path"], 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                record = json.loads(line)
                                # Add metadata for training
                                record["data_type"] = data_type
                                record["source"] = file_info["source"]
                                combined_records.append(record)
                            except json.JSONDecodeError:
                                continue
            
            # Create training dataset file
            if combined_records:
                training_dataset_path = self.training_data_path / f"{data_type}_training.jsonl"
                
                with open(training_dataset_path, 'w', encoding='utf-8') as f:
                    for record in combined_records:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                dataset_info = {
                    "data_type": data_type,
                    "path": str(training_dataset_path),
                    "record_count": len(combined_records),
                    "size_mb": training_dataset_path.stat().st_size / (1024 * 1024),
                    "sources": list(set(f["source"] for f in files))
                }
                training_datasets.append(dataset_info)
                total_training_records += len(combined_records)
                
                logger.info(f"    {data_type}: {len(combined_records)} records")
        
        # Create unified training dataset
        unified_dataset_path = self.training_data_path / "unified_training.jsonl"
        unified_records = []
        
        for dataset_info in training_datasets:
            with open(dataset_info["path"], 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            unified_records.append(record)
                        except json.JSONDecodeError:
                            continue
        
        # Shuffle and save unified dataset
        import random
        random.shuffle(unified_records)
        
        with open(unified_dataset_path, 'w', encoding='utf-8') as f:
            for record in unified_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # Create train/validation split
        split_result = await self._create_train_val_split(unified_records)
        
        return {
            "datasets": training_datasets,
            "unified_dataset": {
                "path": str(unified_dataset_path),
                "record_count": len(unified_records),
                "size_mb": unified_dataset_path.stat().st_size / (1024 * 1024)
            },
            "train_val_split": split_result,
            "total_training_records": total_training_records,
            "data_types": list(datasets_by_type.keys())
        }
    
    async def _create_train_val_split(self, records: List[Dict[str, Any]], val_ratio: float = 0.1) -> Dict[str, Any]:
        """Create train/validation split."""
        import random
        
        # Shuffle records
        shuffled_records = records.copy()
        random.shuffle(shuffled_records)
        
        # Calculate split
        val_count = int(len(shuffled_records) * val_ratio)
        train_count = len(shuffled_records) - val_count
        
        train_records = shuffled_records[:train_count]
        val_records = shuffled_records[train_count:]
        
        # Save splits
        train_path = self.training_data_path / "train.jsonl"
        val_path = self.training_data_path / "validation.jsonl"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            for record in train_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        with open(val_path, 'w', encoding='utf-8') as f:
            for record in val_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_count": train_count,
            "val_count": val_count,
            "val_ratio": val_ratio,
            "train_size_mb": train_path.stat().st_size / (1024 * 1024),
            "val_size_mb": val_path.stat().st_size / (1024 * 1024)
        }
    
    async def _integrate_with_training(self) -> Dict[str, Any]:
        """Integrate prepared datasets with training module."""
        logger.info(" Integrating with training module...")
        
        # Create training configuration
        training_config = {
            "data_paths": {
                "train": str(self.training_data_path / "train.jsonl"),
                "validation": str(self.training_data_path / "validation.jsonl"),
                "unified": str(self.training_data_path / "unified_training.jsonl")
            },
            "dataset_info": {
                "total_records": sum(
                    len(list(open(f, 'r'))) 
                    for f in self.training_data_path.glob("*.jsonl") 
                    if f.exists()
                ),
                "data_types": [
                    d.stem.replace("_training", "") 
                    for d in self.training_data_path.glob("*_training.jsonl")
                ],
                "pipeline_id": self.pipeline_id,
                "created_at": datetime.now().isoformat()
            },
            "model_config": {
                "model_type": "capibara",
                "language": "spanish",
                "domain_specific": True,
                "multimodal": False  # Can be updated based on data types
            },
            "training_params": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "max_epochs": 3,
                "save_checkpoints": True,
                "eval_steps": 500
            }
        }
        
        # Save training configuration
        config_path = self.training_data_path / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        
        # Create training script integration
        integration_script = self._generate_training_integration_script()
        script_path = self.training_data_path / "run_training.py"
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(integration_script)
        
        return {
            "training_config_path": str(config_path),
            "integration_script_path": str(script_path),
            "ready_for_training": True,
            "next_steps": [
                f"cd {self.training_data_path}",
                "python run_training.py",
                "# Or integrate with capibara/training/ module"
            ],
            "integration_completed_at": datetime.now().isoformat()
        }
    
    def _generate_training_integration_script(self) -> str:
        """Generate training integration script."""
        script = f'''#!/usr/bin/env python3
"""
Training Integration Script
Generated by CapibaraGPT v3 Pipeline: {self.pipeline_id}
Generated at: {datetime.now().isoformat()}
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run training with pipeline-generated data."""
    
    # Load training configuration
    config_path = Path(__file__).parent / "training_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    logger.info(" Starting CapibaraGPT v3 Training")
    logger.info(f" Pipeline ID: {{config['dataset_info']['pipeline_id']}}")
    logger.info(f" Total records: {{config['dataset_info']['total_records']}}")
    logger.info(f"️  Data types: {{', '.join(config['dataset_info']['data_types'])}}")
    
    # Import training module
    try:
        from capibara.training import TrainingOrchestrator
        
        # Initialize training orchestrator
        trainer = TrainingOrchestrator(config)
        
        # Start training
        result = trainer.run_training()
        
        logger.info(" Training completed successfully!")
        return result
        
    except ImportError as e:
        logger.warning(f" Training module not available: {{e}}")
        logger.info(" Alternative: Use the generated datasets with your preferred training framework")
        logger.info(f"   Train data: {{config['data_paths']['train']}}")
        logger.info(f"   Validation data: {{config['data_paths']['validation']}}")
        return None

if __name__ == "__main__":
    main()
'''
        return script
    
    def _calculate_data_reduction_ratio(self, processing_result: Dict[str, Any]) -> float:
        """Calculate data reduction ratio after processing."""
        summary = processing_result.get("processing_summary", {})
        input_size = summary.get("total_input_size_gb", 1)
        output_size = summary.get("total_output_size_gb", 0)
        return output_size / input_size if input_size > 0 else 0.0
    
    async def _calculate_final_dataset_stats(self) -> Dict[str, Any]:
        """Calculate final dataset statistics."""
        stats = {
            "total_files": 0,
            "total_size_gb": 0.0,
            "total_records": 0,
            "data_types": [],
            "file_breakdown": {}
        }
        
        if self.training_data_path.exists():
            for file_path in self.training_data_path.glob("*.jsonl"):
                file_size_gb = file_path.stat().st_size / (1024 * 1024 * 1024)
                record_count = sum(1 for line in open(file_path, 'r') if line.strip())
                
                stats["total_files"] += 1
                stats["total_size_gb"] += file_size_gb
                stats["total_records"] += record_count
                
                file_info = {
                    "size_gb": file_size_gb,
                    "record_count": record_count
                }
                stats["file_breakdown"][file_path.name] = file_info
                
                # Extract data type
                if "_training.jsonl" in file_path.name:
                    data_type = file_path.name.replace("_training.jsonl", "")
                    if data_type not in stats["data_types"]:
                        stats["data_types"].append(data_type)
        
        return stats

# Quick execution function
async def run_complete_pipeline(config: Dict[str, Any] = None) -> PipelineResult:
    """Run the complete pipeline with default or provided configuration."""
    
    if config is None:
        config = {
            "storage": {
                "raw_data_path": "data/raw",
                "processed_data_path": "data/processed",
                "training_data_path": "data/training",
                "cache_path": "data/cache"
            },
            "processing": {
                "max_workers": 4,
                "batch_size": 1000,
                "min_quality_score": 0.6,
                "enable_deduplication": True
            },
            "monitoring": {
                "enabled": True,
                "log_level": "INFO"
            }
        }
    
    pipeline = CompletePipeline(config)
    result = await pipeline.execute_complete_pipeline()
    
    return result

# Demo execution
if __name__ == "__main__":
    asyncio.run(run_complete_pipeline())