#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cascade Training Integration - CapibaraGPT-v2

Complete integration between cascade dataset management and training strategies.
Connects the data pipeline with expert training strategies for each stage.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
import time

# Import cascade dataset management
from capibara.data.datasets.cascade_dataset_manager import (
    create_cascade_dataset_manager,
    CascadeDatasetManager
)

# Import training strategies
from capibara.training.expanded_expert_cores_strategy import (
    ExpandedExpertCoresStrategy,
    ExpertCoreType
)
from capibara.training.huggingface_consensus_strategy import (
    HuggingFaceConsensusStrategy
)

logger = logging.getLogger(__name__)

@dataclass
class StageTrainingConfig:
    """Configurestion for training a specific cascade stage."""
    stage_name: str
    model_size: str
    target_size_gb: float
    specialization: str
    expert_cores: List[str]
    datasets_required: List[str]
    quality_threshold: float
    training_strategy: str
    tpu_config: Dict[str, Any]

@dataclass
class TrainingIntegrationResult:
    """Result of training integration process."""
    stage_name: str
    datasets_loaded: int
    datasets_processed: int
    expert_cores_configured: int
    training_ready: bool
    estimated_training_time: float
    quality_score: float
    errors: List[str]

class CascadeTrainingIntegration:
    """
    Complete integration between cascade dataset management and training strategies.
    
    This class connects:
    1. Cascade dataset manager (data pipeline)
    2. Expert training strategies (expanded cores, consensus)
    3. Stage-specific configurations
    4. Quality validation and training preparation
    """
    
    def __init__(self, base_dir: str = "data/cascade_datasets"):
        """Initialize the cascade training integration."""
        self.base_dir = Path(base_dir)
        
        # Initialize components
        self.dataset_manager = create_cascade_dataset_manager(base_dir)
        self.training_configs = self._initialize_training_configs()
        
        # Training strategies
        self.expanded_strategy = None
        self.consensus_strategy = None
        
        logger.info(f"Cascade Training Integration initialized at {self.base_dir}")
    
    def _initialize_training_configs(self) -> Dict[str, StageTrainingConfig]:
        """Initialize training configurations for each cascade stage."""
        return {
            "stage_1_300m": StageTrainingConfig(
                stage_name="stage_1_300m",
                model_size="300M",
                target_size_gb=25.0,
                specialization="Linux Core Systems",
                expert_cores=[
                    "technical",
                    "systems",
                    "programming"
                ],
                datasets_required=[
                    "Linux Kernel Mailing List",
                    "Linux Documentation Project", 
                    "Stack Overflow Linux"
                ],
                quality_threshold=9.0,
                training_strategy="expanded_expert_cores",
                tpu_config={
                    "tpu_type": "v6-8",
                    "batch_size": 32,
                    "learning_rate": 1e-4
                }
            ),
            "stage_2_600m": StageTrainingConfig(
                stage_name="stage_2_600m",
                model_size="600M",
                target_size_gb=60.0,
                specialization="General Productivity Assistant",
                expert_cores=[
                    "general",
                    "academic",
                    "conversational",
                    "analytical"
                ],
                datasets_required=[
                    "FineWeb-Edu Subset",
                    "Wikipedia Multilingual",
                    "ArXiv Papers"
                ],
                quality_threshold=9.2,
                training_strategy="huggingface_consensus",
                tpu_config={
                    "tpu_type": "v6-16",
                    "batch_size": 64,
                    "learning_rate": 8e-5
                }
            )
        }
    
    def _map_datasets_to_expert_cores(self, stage_name: str) -> Dict[str, List[str]]:
        """Map datasets to expert cores for training."""
        stage_config = self.training_configs[stage_name]
        
        # Get available datasets for this stage
        available_datasets = self.dataset_manager.list_available_datasets().get(stage_name, [])
        
        # Map datasets to expert cores based on content and specialization
        dataset_to_core_mapping = {
            "Linux Kernel Mailing List": ["technical", "systems"],
            "Linux Documentation Project": ["technical", "systems"],
            "Stack Overflow Linux": ["programming", "technical"],
            "FineWeb-Edu Subset": ["academic", "general"],
            "Wikipedia Multilingual": ["general", "academic"],
            "ArXiv Papers": ["academic", "analytical"]
        }
        
        # Filter to only available datasets
        filtered_mapping = {
            dataset: cores 
            for dataset, cores in dataset_to_core_mapping.items()
            if dataset in available_datasets
        }
        
        return filtered_mapping
    
    def _validate_datasets_for_training(self, stage_name: str) -> Dict[str, Any]:
        """Validates that datasets are ready for training."""
        stage_config = self.training_configs[stage_name]
        
        validation_result = {
            "stage": stage_name,
            "datasets_required": stage_config.datasets_required,
            "datasets_available": [],
            "datasets_missing": [],
            "quality_scores": {},
            "ready_for_training": False
        }
        
        # Check each required dataset
        for dataset_name in stage_config.datasets_required:
            dataset_info = self.dataset_manager.get_dataset_info(stage_name, dataset_name)
            
            if dataset_info:
                validation_result["datasets_available"].append(dataset_name)
                
                # Check if processed data exists
                processed_path = self.dataset_manager.processed_dir / stage_name / dataset_info["category"]
                if processed_path.exists():
                    validation_result["quality_scores"][dataset_name] = 9.0  # Assume good quality
                else:
                    validation_result["quality_scores"][dataset_name] = 0.0
            else:
                validation_result["datasets_missing"].append(dataset_name)
        
        # Determine if ready for training
        validation_result["ready_for_training"] = (
            len(validation_result["datasets_available"]) >= len(stage_config.datasets_required) * 0.8 and
            all(score >= stage_config.quality_threshold for score in validation_result["quality_scores"].values())
        )
        
        return validation_result
    
    def _configure_expert_cores_for_stage(self, stage_name: str) -> Dict[str, Any]:
        """Configure expert cores for a specific training stage."""
        stage_config = self.training_configs[stage_name]
        
        # Map datasets to expert cores
        dataset_mapping = self._map_datasets_to_expert_cores(stage_name)
        
        # Configure expert cores based on training strategy
        if stage_config.training_strategy == "expanded_expert_cores":
            config = {
                "strategy": "expanded_expert_cores",
                "expert_cores": stage_config.expert_cores,
                "dataset_mapping": dataset_mapping,
                "tpu_config": stage_config.tpu_config,
                "max_cores": len(stage_config.expert_cores),
                "max_models_per_core": 3
            }
        elif stage_config.training_strategy == "huggingface_consensus":
            config = {
                "strategy": "huggingface_consensus",
                "expert_models": stage_config.expert_cores,
                "dataset_mapping": dataset_mapping,
                "tpu_config": stage_config.tpu_config,
                "max_responses": 5
            }
        else:
            config = {
                "strategy": "default",
                "expert_cores": stage_config.expert_cores,
                "dataset_mapping": dataset_mapping,
                "tpu_config": stage_config.tpu_config
            }
        
        return config
    
    def _prepare_training_data(self, stage_name: str) -> Dict[str, Any]:
        """Prepare training data for a specific stage."""
        stage_config = self.training_configs[stage_name]
        
        # Get validation result
        validation = self._validate_datasets_for_training(stage_name)
        
        if not validation["ready_for_training"]:
            return {
                "stage": stage_name,
                "ready": False,
                "errors": [f"Datasets not ready: {validation['datasets_missing']}"],
                "datasets_available": validation["datasets_available"]
            }
        
        # Prepare training data structure
        training_data = {
            "stage": stage_name,
            "ready": True,
            "datasets": {},
            "expert_cores": {},
            "training_config": stage_config
        }
        
        # Load dataset information
        for dataset_name in validation["datasets_available"]:
            dataset_info = self.dataset_manager.get_dataset_info(stage_name, dataset_name)
            if dataset_info:
                training_data["datasets"][dataset_name] = {
                    "info": dataset_info,
                    "processed_path": str(self.dataset_manager.processed_dir / stage_name / dataset_info["category"]),
                    "quality_score": validation["quality_scores"].get(dataset_name, 0.0)
                }
        
        # Configure expert cores
        expert_config = self._configure_expert_cores_for_stage(stage_name)
        training_data["expert_cores"] = expert_config
        
        return training_data
    
    async def integrate_stage_for_training(self, stage_name: str) -> TrainingIntegrationResult:
        """Integrate a specific stage for training."""
        logger.info(f"🔗 Integrating stage {stage_name} for training...")
        
        start_time = time.time()
        errors = []
        
        try:
            # Validate datasets
            validation = self._validate_datasets_for_training(stage_name)
            
            if not validation["ready_for_training"]:
                errors.append(f"Datasets not ready: {validation['datasets_missing']}")
                return TrainingIntegrationResult(
                    stage_name=stage_name,
                    datasets_loaded=len(validation["datasets_available"]),
                    datasets_processed=0,
                    expert_cores_configured=0,
                    training_ready=False,
                    estimated_training_time=0.0,
                    quality_score=0.0,
                    errors=errors
                )
            
            # Prepare training data
            training_data = self._prepare_training_data(stage_name)
            
            # Configure training strategies
            stage_config = self.training_configs[stage_name]
            
            if stage_config.training_strategy == "expanded_expert_cores":
                # Initialize expanded expert cores strategy
                expanded_config = {
                    "tpu_config": stage_config.tpu_config,
                    "expert_cores": training_data["expert_cores"]["expert_cores"],
                    "dataset_mapping": training_data["expert_cores"]["dataset_mapping"]
                }
                self.expanded_strategy = ExpandedExpertCoresStrategy(expanded_config)
                
            elif stage_config.training_strategy == "huggingface_consensus":
                # Initialize HuggingFace consensus strategy
                consensus_config = {
                    "expert_models": training_data["expert_cores"]["expert_models"],
                    "dataset_mapping": training_data["expert_cores"]["dataset_mapping"],
                    "tpu_config": stage_config.tpu_config
                }
                self.consensus_strategy = HuggingFaceConsensusStrategy(consensus_config)
            
            # Calculate quality score
            quality_scores = list(validation["quality_scores"].values())
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            # Estimate training time based on model size and data
            estimated_time = self._estimate_training_time(stage_config, len(validation["datasets_available"]))
            
            integration_time = time.time() - start_time
            
            logger.info(f"✅ Stage {stage_name} integrated successfully in {integration_time:.2f}s")
            
            return TrainingIntegrationResult(
                stage_name=stage_name,
                datasets_loaded=len(validation["datasets_available"]),
                datasets_processed=len(validation["datasets_available"]),
                expert_cores_configured=len(stage_config.expert_cores),
                training_ready=True,
                estimated_training_time=estimated_time,
                quality_score=avg_quality,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Integration error: {str(e)}")
            logger.error(f"❌ Error integrating stage {stage_name}: {e}")
            
            return TrainingIntegrationResult(
                stage_name=stage_name,
                datasets_loaded=0,
                datasets_processed=0,
                expert_cores_configured=0,
                training_ready=False,
                estimated_training_time=0.0,
                quality_score=0.0,
                errors=errors
            )
    
    def _estimate_training_time(self, stage_config: StageTrainingConfig, dataset_count: int) -> float:
        """Estimate training time based on model size and dataset count."""
        # Base training time in hours
        base_times = {
            "300M": 24,   # 24 hours for 300M model
            "600M": 48,   # 48 hours for 600M model
            "1.5B": 96,   # 96 hours for 1.5B model
            "3B": 168,    # 168 hours for 3B model
            "7B": 336,    # 336 hours for 7B model
            "15B": 672    # 672 hours for 15B model
        }
        
        base_time = base_times.get(stage_config.model_size, 24)
        
        # Adjust for dataset count and quality
        dataset_factor = min(dataset_count / 3, 2.0)  # Normalize to 3 datasets
        quality_factor = stage_config.quality_threshold / 9.0
        
        estimated_time = base_time * dataset_factor * quality_factor
        
        return estimated_time
    
    async def integrate_all_stages(self) -> Dict[str, TrainingIntegrationResult]:
        """Integrate all cascade stages for training."""
        logger.info("🚀 Integrating all cascade stages for training...")
        
        results = {}
        
        for stage_name in self.training_configs.keys():
            try:
                result = await self.integrate_stage_for_training(stage_name)
                results[stage_name] = result
                
                # Add delay between stages
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error integrating stage {stage_name}: {e}")
                results[stage_name] = TrainingIntegrationResult(
                    stage_name=stage_name,
                    datasets_loaded=0,
                    datasets_processed=0,
                    expert_cores_configured=0,
                    training_ready=False,
                    estimated_training_time=0.0,
                    quality_score=0.0,
                    errors=[str(e)]
                )
        
        return results
    
    def get_integration_summary(self, results: Dict[str, TrainingIntegrationResult]) -> Dict[str, Any]:
        """Get summary of integration results."""
        summary = {
            "total_stages": len(results),
            "ready_stages": sum(1 for r in results.values() if r.training_ready),
            "total_datasets": sum(r.datasets_loaded for r in results.values()),
            "total_expert_cores": sum(r.expert_cores_configured for r in results.values()),
            "total_training_time": sum(r.estimated_training_time for r in results.values()),
            "avg_quality_score": sum(r.quality_score for r in results.values()) / len(results) if results else 0.0,
            "stages": {}
        }
        
        for stage_name, result in results.items():
            summary["stages"][stage_name] = {
                "ready": result.training_ready,
                "datasets_loaded": result.datasets_loaded,
                "expert_cores": result.expert_cores_configured,
                "training_time": result.estimated_training_time,
                "quality_score": result.quality_score,
                "errors": result.errors
            }
        
        return summary
    
    def save_integration_report(self, results: Dict[str, TrainingIntegrationResult], 
                              output_file: Optional[Path] = None):
        """Save integration report to file."""
        if output_file is None:
            output_file = self.base_dir / "training_integration_report.json"
        
        report = {
            "timestamp": time.time(),
            "summary": self.get_integration_summary(results),
            "results": {
                stage_name: asdict(result) for stage_name, result in results.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Integration report saved to {output_file}")

def create_cascade_training_integration(base_dir: Optional[str] = None) -> CascadeTrainingIntegration:
    """Creates a cascade training integration instance."""
    if base_dir is None:
        base_dir = "data/cascade_datasets"
    
    return CascadeTrainingIntegration(base_dir)

async def integrate_cascade_training(base_dir: Optional[str] = None) -> Dict[str, TrainingIntegrationResult]:
    """Integrate cascade training system."""
    integration = create_cascade_training_integration(base_dir)
    return await integration.integrate_all_stages()

def main():
    """Main function for testing the cascade training integration."""
    logger.info("🦫 CapibaraGPT-v2 Cascade Training Integration")
    
    async def run_integration():
        # Create integration instance
        integration = create_cascade_training_integration()
        
        # Integrate all stages
        results = await integration.integrate_all_stages()
        
        # Save report
        integration.save_integration_report(results)
        
        # Show summary
        summary = integration.get_integration_summary(results)
        logger.info("📊 Integration Summary:")
        logger.info(f"  Total stages: {summary['total_stages']}")
        logger.info(f"  Ready stages: {summary['ready_stages']}")
        logger.info(f"  Total datasets: {summary['total_datasets']}")
        logger.info(f"  Total expert cores: {summary['total_expert_cores']}")
        logger.info(f"  Total training time: {summary['total_training_time']:.1f} hours")
        logger.info(f"  Average quality: {summary['avg_quality_score']:.3f}")
        
        return results
    
    return asyncio.run(run_integration())

if __name__ == "__main__":
    main()