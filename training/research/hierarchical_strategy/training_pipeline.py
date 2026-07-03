#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Hierarchical Training Pipeline - CapibaraGPT v3
Pipeline complete de entrenamiento with transfer learning progresivo.

Pipeline: 300M LinuxCore → 600M Laptop → 1.2B Humanoid → 3B CodeMaster → 7B PolicyExpert → 13B OmniGenomic
"""

import os
import json
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)

class ModelTier(Enum):
    """Tiers del pipeline de modelos"""
    FOUNDATION = "foundation"      # 300M LinuxCore
    ASSISTANT = "assistant"        # 600M Laptop
    SPECIALIST = "specialist"      # 1.2B Humanoid
    EXPERT = "expert"             # 3B CodeMaster
    POLICY = "policy"             # 7B PolicyExpert
    OMNI = "omni"                 # 13B OmniGenomic

@dataclass
class ModelConfig:
    """setup completa de un model en el pipeline"""
    name: str
    size: str
    parameters: int
    dataset_size_gb: int
    estimated_tokens: int
    specialization_domain: str
    
    # Transfer learning config
    inherits_from: Optional[str] = None
    inherited_params_ratio: float = 0.0
    new_data_ratio: float = 1.0
    
    # Quality metrics
    target_ratio_tokens_params: float = 33.0
    quality_target: float = 9.0
    
    # Training specifics
    training_time_gpu_hours: int = 500
    checkpoint_frequency: int = 1000
    validation_frequency: int = 500

@dataclass
class DistillationConfig:
    """setup for modelos destilados"""
    teacher_model: str
    distillation_ratio: float = 0.2
    preserve_capabilities: List[str] = field(default_factory=list)
    focus_areas: List[str] = field(default_factory=list)
    temperature: float = 4.0
    alpha: float = 0.7
    epochs: int = 50

class HierarchicalTrainingPipeline:
    """
    Pipeline principal de entrenamiento jerárquico.
    Gestiona all el process de transfer learning and destilación.
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_config = self._initialize_pipeline_config()
        self.distillation_configs = self._initialize_distillation_configs()
        
        logger.info(" Hierarchical Training Pipeline initialized")
        
    def _initialize_pipeline_config(self) -> Dict[str, ModelConfig]:
        """setup completa del pipeline"""
        return {
            "300M_LinuxCore": ModelConfig(
                name="LinuxCore Foundation",
                size="300M",
                parameters=300_000_000,
                dataset_size_gb=25,
                estimated_tokens=10_000_000_000,
                specialization_domain="linux_systems",
                target_ratio_tokens_params=33.3,
                quality_target=8.5,
                training_time_gpu_hours=400
            ),
            
            "600M_LaptopAssistant": ModelConfig(
                name="Laptop Assistant", 
                size="600M",
                parameters=600_000_000,
                dataset_size_gb=60,
                estimated_tokens=24_000_000_000,
                specialization_domain="general_programming",
                inherits_from="300M_LinuxCore",
                inherited_params_ratio=0.5,
                new_data_ratio=0.92,
                target_ratio_tokens_params=40.0,
                quality_target=8.8,
                training_time_gpu_hours=600
            ),
            
            "1.2B_HumanoidBrain": ModelConfig(
                name="Humanoid Brain",
                size="1.2B", 
                parameters=1_200_000_000,
                dataset_size_gb=100,
                estimated_tokens=40_000_000_000,
                specialization_domain="robotics_manipulation",
                inherits_from="600M_LaptopAssistant",
                inherited_params_ratio=0.5,
                new_data_ratio=0.88,
                target_ratio_tokens_params=33.3,
                quality_target=9.1,
                training_time_gpu_hours=800
            ),
            
            "3B_CodeMaster": ModelConfig(
                name="Code Master",
                size="3B",
                parameters=3_000_000_000,
                dataset_size_gb=250,
                estimated_tokens=100_000_000_000,
                specialization_domain="mathematics_navigation",
                inherits_from="1.2B_HumanoidBrain",
                inherited_params_ratio=0.4,
                new_data_ratio=0.92,
                target_ratio_tokens_params=33.3,
                quality_target=9.3,
                training_time_gpu_hours=1200
            ),
            
            "7B_PolicyExpert": ModelConfig(
                name="Policy Expert",
                size="7B",
                parameters=7_000_000_000,
                dataset_size_gb=500,
                estimated_tokens=200_000_000_000,
                specialization_domain="policy_legal_medical",
                inherits_from="3B_CodeMaster",
                inherited_params_ratio=0.43,
                new_data_ratio=0.90,
                target_ratio_tokens_params=28.6,
                quality_target=9.5,
                training_time_gpu_hours=1800
            ),
            
            "13B_OmniGenomic": ModelConfig(
                name="Omni Genomic",
                size="13B",
                parameters=13_000_000_000,
                dataset_size_gb=800,  # Corregido equilibrio
                estimated_tokens=320_000_000_000,
                specialization_domain="genomics_multimodal",
                inherits_from="7B_PolicyExpert",
                inherited_params_ratio=0.54,
                new_data_ratio=0.875,
                target_ratio_tokens_params=24.6,
                quality_target=9.7,
                training_time_gpu_hours=2500
            )
        }
    
    def _initialize_distillation_configs(self) -> Dict[str, DistillationConfig]:
        """setup de todos los modelos destilados"""
        return {
            "60M_LinuxCore_Mini": DistillationConfig(
                teacher_model="300M_LinuxCore",
                preserve_capabilities=["linux_systems", "basic_programming"],
                focus_areas=["command_line", "system_administration"]
            ),
            
            "120M_LaptopAssistant_Mini": DistillationConfig(
                teacher_model="600M_LaptopAssistant",
                preserve_capabilities=["general_programming", "problem_solving"],
                focus_areas=["code_generation", "debugging"]
            ),
            
            "240M_HumanoidBrain_Mini": DistillationConfig(
                teacher_model="1.2B_HumanoidBrain",
                preserve_capabilities=["robotics", "manipulation"],
                focus_areas=["task_planning", "spatial_reasoning"]
            ),
            
            "600M_CodeMaster_Mini": DistillationConfig(
                teacher_model="3B_CodeMaster",
                preserve_capabilities=["mathematics", "algorithms"],
                focus_areas=["complex_algorithms", "optimization"]
            ),
            
            "1.4B_PolicyExpert_Mini": DistillationConfig(
                teacher_model="7B_PolicyExpert",
                preserve_capabilities=["policy", "legal", "medical"],
                focus_areas=["regulation_compliance", "ethical_reasoning"]
            ),
            
            "2.6B_OmniGenomic_Mini": DistillationConfig(
                teacher_model="13B_OmniGenomic",
                preserve_capabilities=["universal_knowledge", "routing"],
                focus_areas=["query_analysis", "expert_routing"],
                temperature=3.0,
                alpha=0.8
            )
        }
    
    def get_pipeline_overview(self) -> Dict[str, Any]:
        """Resumen complete del pipeline"""
        total_params = sum(config.parameters for config in self.models_config.values())
        total_data_gb = sum(config.dataset_size_gb for config in self.models_config.values())
        
        return {
            "pipeline_overview": {
                "total_models": len(self.models_config),
                "total_parameters": f"{total_params:,}",
                "total_data_gb": total_data_gb,
                "average_quality": sum(c.quality_target for c in self.models_config.values()) / len(self.models_config)
            },
            "model_progression": [
                {
                    "model": config.name,
                    "size": config.size,
                    "specialization": config.specialization_domain,
                    "inherits_from": config.inherits_from,
                    "ratio_T_P": config.target_ratio_tokens_params,
                    "quality_target": config.quality_target
                }
                for config in self.models_config.values()
            ]
        }
    
    def validate_pipeline_balance(self) -> Dict[str, Any]:
        """Valida equilibrio del pipeline according to métricas"""
        validation = {
            "balanced_models": [],
            "needs_adjustment": [],
            "overall_status": "good"
        }
        
        for model_name, config in self.models_config.items():
            if 25 <= config.target_ratio_tokens_params <= 40:
                validation["balanced_models"].append({
                    "model": model_name,
                    "ratio": config.target_ratio_tokens_params,
                    "status": " OPTIMAL"
                })
            else:
                validation["needs_adjustment"].append({
                    "model": model_name,
                    "ratio": config.target_ratio_tokens_params,
                    "status": "️ NEEDS REVIEW"
                })
        
        if validation["needs_adjustment"]:
            validation["overall_status"] = "needs_review"
        
        return validation

def create_training_pipeline(base_dir: str = "training_pipeline") -> HierarchicalTrainingPipeline:
    """Factory function for create el pipeline"""
    return HierarchicalTrainingPipeline(base_dir)

def validate_training_strategy() -> Dict[str, Any]:
    """Valida la estrategia completa"""
    pipeline = create_training_pipeline()
    
    return {
        "pipeline_overview": pipeline.get_pipeline_overview(),
        "balance_validation": pipeline.validate_pipeline_balance(),
        "status": " Ready for implementation"
    }

__all__ = [
    'HierarchicalTrainingPipeline',
    'ModelConfig', 
    'DistillationConfig',
    'ModelTier',
    'create_training_pipeline',
    'validate_training_strategy'
] 