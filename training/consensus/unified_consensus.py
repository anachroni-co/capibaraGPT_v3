"""
Unified module for consensus distilling and strategies.
Merges the functionality of consensus_destiling.py and consensus_strategies.py
"""

import os
import sys
import time
import logging
import asyncio
import hashlib
import numpy as np
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Tuple, Union, Callable

# JAX imports with fallbacks
try:
    from flax import linen as nn
    from jax import numpy as jnp
    from flax.training import train_state
except ImportError:
    # Fallback implementations
    import numpy as jnp
    nn = None
    train_state = None

# Optax import with fallback
try:
    import optax
except ImportError:
    optax = None

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

# Training config imports with fallbacks
try:
    from .training_config import ModelScale, TrainingConfigFactory
except ImportError:
    # Fallback implementation
    from enum import Enum
    class ModelScale(Enum):
        MICRO_300M = "300M"
        SMALL_3B = "3B"
        MEDIUM_30B = "30B"
        LARGE_1T = "1T"
    
    class TrainingConfigFactory:
        @staticmethod
        def get_config(scale):
            return {}

logger = logging.getLogger(__name__)

# Integrated Capibara System
# Combines existing implementation with refined TPU training
#
# Architecture:
# 1. CapibaraConsensusDistiller (existing implementation)
# 2. CapibaraRefiner (existing implementation)
# 3. RefinedCapibaraTrainer (optimized for TPU)
# 4. Full integration with W&B, caching, and metrics

# Additional imports with fallbacks
try:
    from .voting import CapibaraVotingSystem
except ImportError:
    class CapibaraVotingSystem:
        """Fallback voting system for consensus learning."""
        def __init__(self, *args, **kwargs):
            self.consensus_threshold = kwargs.get('consensus_threshold', 0.8)
            self.voting_strategies = ['majority', 'weighted', 'confidence']
            logger.warning("Using CapibaraVotingSystem fallback")

        def vote(self, predictions, weights=None):
            """Basic voting implementation."""
            if not predictions:
                return None
            return predictions[0]  # Return first prediction as fallback

try:
    from .capibara_refinement import CapibaraRefiner, RefinerModel
except ImportError:
    class CapibaraRefiner:
        """Fallback refiner for improving predictions."""
        def __init__(self, *args, **kwargs):
            self.refinement_steps = kwargs.get('refinement_steps', 3)
            self.improvement_threshold = kwargs.get('improvement_threshold', 0.05)
            logger.warning("Using CapibaraRefiner fallback")

        def refine(self, prediction, context=None):
            """Basic refinement implementation."""
            return prediction  # No refinement in fallback
            
    class RefinerModel:
        """Fallback refiner model."""
        def __init__(self, *args, **kwargs):
            self.model_type = "fallback_refiner"
            logger.warning("Using RefinerModel fallback")

        def __call__(self, inputs):
            return inputs  # No processing in fallback

try:
    from .capibara_trainer import CapibaraConsensusDistiller, TeacherModel
except ImportError:
    class CapibaraConsensusDistiller:
        """Fallback consensus distiller."""
        def __init__(self, *args, **kwargs):
            self.distillation_temp = kwargs.get('distillation_temp', 3.0)
            self.consensus_weight = kwargs.get('consensus_weight', 0.7)
            logger.warning("Using CapibaraConsensusDistiller fallback")

        def distill(self, teacher_outputs, student_outputs):
            """Basic distillation implementation."""
            return student_outputs  # No distillation in fallback
            
    class TeacherModel:
        """Fallback teacher model."""
        def __init__(self, *args, **kwargs):
            self.knowledge_base = {}
            self.teaching_strategies = ['direct', 'guided', 'discovery']
            logger.warning("Using TeacherModel fallback")

        def teach(self, student_input):
            """Basic teaching fallback."""
            return {"guidance": "basic_feedback", "confidence": 0.5}

@dataclass
class IntegratedTrainingConfig:
    """Integrated configuration that combines all components."""
    
    # Model scale progression
    model_scales: List[str] = field(default_factory=lambda: [
        "300M", "600M", "1.2B", "3B", "7B", "13B", "30B", "65B", "130B", "1T"
    ])
    
    # Teacher models for each phase
    teacher_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "micro_phase": {
            "models": ["microsoft/CodeT5p-220m", "Salesforce/codet5-base", "microsoft/codebert-base"],
            "weights": [0.4, 0.3, 0.3],
            "parallel_calls": 3
        },
        "small_phase": {
            "models": ["microsoft/CodeT5p-770m", "Salesforce/codet5-large", "microsoft/unixcoder-base"],
            "weights": [0.5, 0.3, 0.2],
            "parallel_calls": 5
        },
        "medium_phase": {
            "models": ["bigcode/starcoder-1b", "microsoft/CodeT5p-2b", "Salesforce/codet5-large"],
            "weights": [0.6, 0.25, 0.15],
            "parallel_calls": 7
        },
        "large_phase": {
            "models": ["bigcode/starcoder-3b", "microsoft/CodeT5p-6b", "facebook/incoder-6B"],
            "weights": [0.7, 0.2, 0.1],
            "parallel_calls": 10
        }
    })
    
    # Refiner models for iterative improvement
    refiner_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "micro_phase": {
            "models": ["microsoft/CodeT5p-220m-bimodal", "Salesforce/codet5-base-multi-sum"],
            "max_iterations": 3
        },
        "small_phase": {
            "models": ["microsoft/CodeT5p-770m-py", "facebook/bart-large"],
            "max_iterations": 4
        },
        "medium_phase": {
            "models": ["bigcode/starcoder-1b", "microsoft/CodeT5p-2b"],
            "max_iterations": 5
        },
        "large_phase": {
            "models": ["bigcode/starcoder-3b", "microsoft/CodeT5p-6b"],
            "max_iterations": 6
        }
    })
    
    # Voting system configs
    voting_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "micro_phase": {"n_teachers": 5, "n_critics": 3, "alpha": 0.1},
        "small_phase": {"n_teachers": 7, "n_critics": 3, "alpha": 0.08},
        "medium_phase": {"n_teachers": 10, "n_critics": 5, "alpha": 0.06},
        "large_phase": {"n_teachers": 15, "n_critics": 7, "alpha": 0.05}
    })
    
    # Dataset generation
    dataset_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "micro_phase": {
            "size_range": (20_000_000, 80_000_000),
            "quality_threshold": 0.7,
            "batch_size": 64,
            "cache_enabled": True
        },
        "small_phase": {
            "size_range": (200_000_000, 500_000_000),
            "quality_threshold": 0.75,
            "batch_size": 128,
            "cache_enabled": True
        },
        "medium_phase": {
            "size_range": (2_000_000_000, 5_000_000_000),
            "quality_threshold": 0.8,
            "batch_size": 256,
            "cache_enabled": True
        },
        "large_phase": {
            "size_range": (10_000_000_000, 50_000_000_000),
            "quality_threshold": 0.85,
            "batch_size": 512,
            "cache_enabled": True
        }
    })
    
    # Training parameters
    training_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "micro_phase": {
            "learning_rate": 3e-4,
            "warmup_steps": 2000,
            "total_steps": 50000,
            "tpu_cores": 32
        },
        "small_phase": {
            "learning_rate": 2e-4,
            "warmup_steps": 5000,
            "total_steps": 100000,
            "tpu_cores": 128
        },
        "medium_phase": {
            "learning_rate": 1e-4,
            "warmup_steps": 10000,
            "total_steps": 200000,
            "tpu_cores": 512
        },
        "large_phase": {
            "learning_rate": 5e-5,
            "warmup_steps": 20000,
            "total_steps": 500000,
            "tpu_cores": 2048
        }
    })
    
    # Paths and caching
    cache_dir: str = "caches"
    output_dir: str = "output"
    wandb_project: str = "capibara-integrated-training"
    checkpoint_every: int = 5000

class EnhancedConsensusDistiller(CapibaraConsensusDistiller):
    """Enhanced version of CapibaraConsensusDistiller with TPU integration."""
    
    def __init__(
        self,
        teacher_models: List[TeacherModel],
        phase_name: str,
        config: IntegratedTrainingConfig,
        **kwargs
    ):
        # Phase-specific configuration
        phase_config = config.dataset_configs[phase_name]
        
        super().__init__(
            teacher_models=teacher_models,
            cache_dir=f"{config.cache_dir}/teacher_cache/{phase_name}",
            **kwargs
        )
        
        self.phase_name = phase_name
        self.config = config
        self.quality_threshold = phase_config["quality_threshold"]
        self.batch_size = phase_config["batch_size"]
        
        # Extended metrics
        self.phase_metrics = {
            "total_samples_generated": 0,
            "high_quality_samples": 0,
            "consensus_rate": 0.0,
            "average_teacher_latency": 0.0,
            "cache_hit_rate": 0.0
        }
    
    async def generate_enhanced_dataset(
        self,
        prompts: List[str],
        target_size: int
    ) -> Dict[str, Any]:
        """Generate enhanced dataset with metrics and quality filtering."""

        logger.info(f"Generating dataset {self.phase_name} - Target: {target_size:,} samples")
        
        high_quality_samples = []
        all_samples = []
        cache_hits = 0
        total_teacher_time = 0
        
        start_time = time.time()
        
        # Process in batches
        for i in range(0, min(len(prompts), target_size), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            
            batch_samples = []
            batch_start = time.time()
            
            for prompt in batch_prompts:
                # Check cache
                cache_key = self._get_cache_key(0, prompt)  # Use first teacher for cache key
                if self._load_from_cache(cache_key):
                    cache_hits += 1
                
                # Generate sample with consensus
                sample = self.create_training_pair(prompt)
                
                all_samples.append(sample)
                
                # Filter by quality
                if sample["quality"] >= self.quality_threshold:
                    high_quality_samples.append(sample)
                    
                    # Stop if we reached the target
                    if len(high_quality_samples) >= target_size:
                        break
            
            batch_time = time.time() - batch_start
            total_teacher_time += batch_time
            
            # Log progress
            if i % (self.batch_size * 10) == 0:
                progress = len(high_quality_samples) / target_size * 100
                logger.info(
                    f"Progress: {progress:.1f}% - "
                    f"Quality: {len(high_quality_samples)}/{len(all_samples)} "
                    f"({len(high_quality_samples)/max(len(all_samples), 1):.2%})"
                )
            
            if len(high_quality_samples) >= target_size:
                break
        
        total_time = time.time() - start_time
        
        # Update metrics
        self.phase_metrics.update({
            "total_samples_generated": len(all_samples),
            "high_quality_samples": len(high_quality_samples),
            "consensus_rate": len(high_quality_samples) / max(len(all_samples), 1),
            "average_teacher_latency": total_teacher_time / max(len(all_samples), 1),
            "cache_hit_rate": cache_hits / max(len(all_samples), 1),
            "generation_time_minutes": total_time / 60
        })
        
        # Tokenize final samples
        tokenized_dataset = self.tokenizer(
            [s["input"] for s in high_quality_samples],
            text_target=[s["output"] for s in high_quality_samples],
            return_tensors="np",
            padding="max_length",
            max_length=512,
            truncation=True,
        )
        
        tokenized_dataset["quality_scores"] = np.array([s["quality"] for s in high_quality_samples])
        
        logger.info(f"Dataset {self.phase_name} generated: {len(high_quality_samples):,} samples")
        
        return {
            "dataset": tokenized_dataset,
            "metrics": self.phase_metrics,
            "raw_samples": high_quality_samples[:1000]  # Save subset for analysis
        }

class EnhancedRefiner(CapibaraRefiner):
    """Enhanced version of CapibaraRefiner with metrics and caching."""
    
    def __init__(
        self,
        refiner_models: List[RefinerModel],
        phase_name: str,
        config: IntegratedTrainingConfig
    ):
        phase_config = config.refiner_configs[phase_name]
        
        super().__init__(
            refiner_models=refiners,
            max_iterations=phase_config["max_iterations"]
        )
        
        self.phase_name = phase_name
        self.config = config
        self.cache_dir = f"{config.cache_dir}/refinement_cache/{phase_name}"
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Refinement metrics
        self.refinement_metrics = {
            "total_refinements": 0,
            "successful_refinements": 0,
            "average_iterations": 0.0,
            "improvement_score": 0.0
        }
    
    def _get_refinement_cache_key(self, prompt: str, base_code: str) -> str:
        """Generate cache key for refinement."""
        combined = f"{prompt}|||{base_code}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _load_refinement_cache(self, cache_key: str) -> Optional[str]:
        """Load refinement from cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data['refined_code']
            except:
                pass
        return None
    
    def _save_refinement_cache(self, cache_key: str, refined_code: str, iterations: int):
        """Save refinement to cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_path, 'w') as f:
            json.dump({
                "refined_code": refined_code,
                "iterations": iterations,
                "timestamp": datetime.now().isoformat()
            }, f)
    
    def enhanced_refinement_loop(
        self, 
        prompt: str, 
        base_code: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Enhanced refinement loop with caching and metrics."""

        # Check cache
        cache_key = self._get_refinement_cache_key(prompt, base_code)
        cached_result = self._load_refinement_cache(cache_key)
        
        if cached_result:
            return cached_result, {"iterations": 0, "cached": True}
        
        # Execute refinement
        start_time = time.time()
        current = base_code
        iterations_used = 0
        improvement_scores = []
        
        for iteration in range(self.max_iterations):
            refinements = self.generate_refinements(prompt, current)
            
            # Filter refinements with substantial differences
            filtered = [r for r in refinements if self.is_substantially_different(current, r)]
            
            if not filtered:
                logger.debug(f"[Refiner] No further improvements at iteration {iteration}")
                break
            
            # Score refinements with heuristic metrics
            scored_refinements = []
            for refinement in filtered:
                score = self._score_refinement(prompt, current, refinement)
                scored_refinements.append((refinement, score))
            
            # Select best refinement
            if scored_refinements:
                best_refinement, best_score = max(scored_refinements, key=lambda x: x[1])
                
                if best_score > 0.5:  # Threshold for acceptance
                    current = best_refinement
                    iterations_used = iteration + 1
                    improvement_scores.append(best_score)
                    logger.debug(f"[Refiner] Iteration {iteration+1}: improvement score {best_score:.3f}")
                else:
                    logger.debug(f"[Refiner] Iteration {iteration+1}: no significant improvement")
                    break
            else:
                break
        
        refinement_time = time.time() - start_time
        
        # Save to cache
        self._save_refinement_cache(cache_key, current, iterations_used)
        
        # Update metrics
        self.refinement_metrics["total_refinements"] += 1
        if iterations_used > 0:
            self.refinement_metrics["successful_refinements"] += 1
        
        avg_iterations = (
            self.refinement_metrics["average_iterations"] * (self.refinement_metrics["total_refinements"] - 1) +
            iterations_used
        ) / self.refinement_metrics["total_refinements"]
        self.refinement_metrics["average_iterations"] = avg_iterations
        
        if improvement_scores:
            avg_improvement = np.mean(improvement_scores)
            self.refinement_metrics["improvement_score"] = (
                self.refinement_metrics["improvement_score"] * 0.9 + avg_improvement * 0.1
            )
        
        metadata = {
            "iterations": iterations_used,
            "refinement_time": refinement_time,
            "improvement_scores": improvement_scores,
            "cached": False
        }
        
        return current, metadata
    
    def _score_refinement(self, prompt: str, original: str, refinement: str) -> float:
        """Refinement quality scoring."""

        # Simple factors
        length_improvement = len(refinement) / max(len(original), 1)
        
        # Penalize excessive changes
        if length_improvement > 2.0 or length_improvement < 0.5:
            return 0.2
        
        # Reward moderate complexity
        complexity_score = min(1.0, len(refinement.split('\n')) / 10.0)
        
        # Combined score
        return (length_improvement * 0.3 + complexity_score * 0.7) * np.random.uniform(0.8, 1.0)

class IntegratedCapibaraTrainer:
    """Integrated trainer that combines all components."""
    
    def __init__(self, config: IntegratedTrainingConfig):
        self.config = config
        self.current_phase = None
        self.phase_results = {}
        
        # Initialize W&B
        if jax.process_index() == 0:
            wandb.init(
                project=config.wandb_project,
                config=dict(config.__dict__),
                name=f"capibara-integrated-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
    
    async def execute_integrated_training(
        self,
        target_scale: str = "1T",
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute complete integrated training."""

        logger.info(f"Starting Integrated Capibara Training towards {target_scale}")
        
        start_time = datetime.now()
        
        # Determine phases to execute
        phases = self._get_phases_for_target(target_scale)
        
        previous_models = None
        
        for phase_name in phases:
            logger.info(f"\nExecuting {phase_name}")

            # Execute integrated phase
            phase_result = await self._execute_integrated_phase(
                phase_name, previous_models
            )
            
            self.phase_results[phase_name] = phase_result
            previous_models = phase_result.get("trained_models")
            
            # Log to W&B
            if jax.process_index() == 0:
                self._log_phase_to_wandb(phase_name, phase_result)
        
        total_time = datetime.now() - start_time
        
        # Final analysis
        final_analysis = self._analyze_integrated_training()
        
        logger.info(f"Integrated Training completed in {total_time}")
        
        return {
            "phase_results": self.phase_results,
            "final_analysis": final_analysis,
            "total_time": total_time
        }
    
    async def _execute_integrated_phase(
        self, 
        phase_name: str, 
        previous_models: Optional[List] = None
    ) -> Dict[str, Any]:
        """Execute a complete integrated phase."""
        
        phase_start = datetime.now()
        
        # 1. Setup teachers and refiners
        teachers = await self._setup_teachers(phase_name)
        refiners = await self._setup_refiners(phase_name)
        
        # 2. Create distiller and refiner
        distiller = EnhancedConsensusDistiller(
            teacher_models=teachers,
            phase_name=phase_name,
            config=self.config
        )
        
        refiner = EnhancedRefiner(
            refiner_models=refiners,
            phase_name=phase_name,
            config=self.config
        )
        
        # 3. Create voting system
        voting_config = self.config.voting_configs[phase_name]
        voting_system = CapibaraVotingSystem(**voting_config)
        
        # 4. generate prompts base
        prompts = await self._generate_phase_prompts(phase_name)
        
        # 5. generate dataset with consenso
        dataset_config = self.config.dataset_configs[phase_name]
        target_size = dataset_config["size_range"][1]  # Use max size
        
        logger.info(f" Generando dataset consenso para {phase_name}")
        consensus_result = await distiller.generate_enhanced_dataset(prompts, target_size)
        
        # 6. apply refinamiento iterativo
        logger.info(f" Aplicando refinamiento iterativo")
        refined_samples = []
        
        for sample in consensus_result["raw_samples"]:
            refined_output, refinement_metadata = refiner.enhanced_refinement_loop(
                sample["input"], sample["output"]
            )
            
            refined_sample = sample.copy()
            refined_sample["output"] = refined_output
            refined_sample["refinement_metadata"] = refinement_metadata
            refined_samples.append(refined_sample)
        
        # 7. integrate with sistema de votaciÃÂ³n
        logger.info(f"Ã¯Â¸Â Integrando con sistema de votaciÃÂ³n")
        voted_samples = []
        
        for sample in refined_samples[:1000]:  # Subset for voting demo
            # Simular mÃÂºltiples outputs for votaciÃÂ³n
            candidate_outputs = [sample["output"]] + [
                f"variant_{i}_" + sample["output"] for i in range(voting_config["n_teachers"] - 1)
            ]
            
            # Simular latencias and scores
            latencies = [np.random.uniform(0.1, 0.5) for _ in candidate_outputs]
            scores = [np.random.uniform(0.7, 0.9) for _ in candidate_outputs]
            
            # Votar
            best_output = voting_system.peer_vote(
                sample["input"], candidate_outputs, latencies
            )
            
            voted_sample = sample.copy()
            voted_sample["output"] = best_output[0]
            voted_samples.append(voted_sample)
        
        # 8. train modelos de la fase
        training_config = self.config.training_configs[phase_name]
        
        trained_models = {}
        for model_scale in self._get_models_for_phase(phase_name):
            logger.info(f" Entrenando modelo {model_scale}")
            
            # Placeholder for entrenamiento real
            model_result = await self._train_model_with_integrated_data(
                model_scale, consensus_result["dataset"], training_config
            )
            
            trained_models[model_scale] = model_result
        
        phase_time = datetime.now() - phase_start
        
        # 9. Compilar resultados de fase
        phase_result = {
            "trained_models": trained_models,
            "consensus_metrics": consensus_result["metrics"],
            "refinement_metrics": refiner.refinement_metrics,
            "voting_metrics": voting_system.get_metrics(),
            "dataset_size": len(consensus_result["dataset"]["input_ids"]),
            "phase_time": phase_time,
            "samples_breakdown": {
                "consensus_samples": len(consensus_result["raw_samples"]),
                "refined_samples": len(refined_samples),
                "voted_samples": len(voted_samples)
            }
        }
        
        return phase_result
    
    async def _setup_teachers(self, phase_name: str) -> List[TeacherModel]:
        """Setup teachers for una fase."""
        
        teacher_config = self.config.teacher_configs[phase_name]
        teachers = []
        
        # Placeholder - en implementation real, carry modelos reales
        for i, model_name in enumerate(teacher_config["models"]):
            def teacher_fn(prompt: str, model=model_name) -> str:
                # Simular answer de teacher
                return f"# Generated by {model}\n{prompt.lower().replace('?', '').replace(' ', '_')}_function()"
            
            teachers.append(teacher_fn)
        
        logger.info(f" Setup {len(teachers)} teachers para {phase_name}")
        return teachers
    
    async def _setup_refiners(self, phase_name: str) -> List[RefinerModel]:
        """Setup refiners for una fase."""
        
        refiner_config = self.config.refiner_configs[phase_name]
        refiners = []
        
        # Placeholder - en implementation real, carry modelos reales
        for model_name in refiner_config["models"]:
            def refiner_fn(prompt: str, code: str, model=model_name) -> str:
                # Simular refinamiento
                if "# optimized" not in code:
                    return code.strip() + f"\n# optimized by {model}"
                return code
            
            refiners.append(refiner_fn)
        
        logger.info(f" Setup {len(refiners)} refiners para {phase_name}")
        return refiners
    
    async def _generate_phase_prompts(self, phase_name: str) -> List[str]:
        """Genera prompts especÃÂ­ficos for una fase."""
        
        base_prompts = [
            "Write a function that sorts a list",
            "Create a class for a binary tree",
            "Implement a REST API endpoint",
            "Generate a function for file processing",
            "Create a data validation function"
        ]
        
        # scale complejidad according to fase
        complexity_multiplier = {
            "micro_phase": 1,
            "small_phase": 2,
            "medium_phase": 5,
            "large_phase": 10
        }.get(phase_name, 1)
        
        expanded_prompts = base_prompts * complexity_multiplier * 1000  # Scale up
        
        logger.info(f" Generados {len(expanded_prompts):,} prompts para {phase_name}")
        return expanded_prompts
    
    def _get_phases_for_target(self, target_scale: str) -> List[str]:
        """Determina quÃÂ© fases execute for reach el target."""
        
        scale_to_phase = {
            "300M": ["micro_phase"],
            "1.2B": ["micro_phase"],
            "13B": ["micro_phase", "small_phase"],
            "65B": ["micro_phase", "small_phase", "medium_phase"],
            "1T": ["micro_phase", "small_phase", "medium_phase", "large_phase"]
        }
        
        return scale_to_phase.get(target_scale, ["micro_phase"])
    
    def _get_models_for_phase(self, phase_name: str) -> List[str]:
        """Obtiene modelos a train en una fase."""
        
        phase_to_models = {
            "micro_phase": ["300M", "600M", "1.2B"],
            "small_phase": ["3B", "7B", "13B"],
            "medium_phase": ["30B", "50B", "65B"],
            "large_phase": ["130B", "300B", "1T"]
        }
        
        return phase_to_models.get(phase_name, [])
    
    async def _train_model_with_integrated_data(
        self,
        model_scale: str,
        dataset: Dict[str, np.ndarray],
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Entrena model with data integrados."""
        
        # Placeholder for entrenamiento real
        start_time = time.time()
        
        # Simular entrenamiento
        await asyncio.sleep(0.1)  # Simular tiempo de entrenamiento
        
        training_time = time.time() - start_time
        
        # estimate parÃÂ¡metros
        param_count = int(model_scale.replace("M", "000000").replace("B", "000000000").replace("T", "000000000000"))
        
        return {
            "parameter_count": param_count,
            "training_time_hours": training_time / 3600,
            "final_loss": np.random.uniform(1.5, 2.5),
            "dataset_size": len(dataset["input_ids"]),
            "convergence_rate": np.random.uniform(0.85, 0.95)
        }
    
    def _log_phase_to_wandb(self, phase_name: str, phase_result: Dict[str, Any]):
        """Log resultados de fase a W&B."""
        
        wandb.log({
            f"{phase_name}/consensus_rate": phase_result["consensus_metrics"]["consensus_rate"],
            f"{phase_name}/refinement_success": phase_result["refinement_metrics"]["successful_refinements"],
            f"{phase_name}/dataset_size": phase_result["dataset_size"],
            f"{phase_name}/phase_time_minutes": phase_result["phase_time"].total_seconds() / 60,
            f"{phase_name}/average_quality": np.mean([
                result["final_loss"] for result in phase_result["trained_models"].values()
            ]),
            f"{phase_name}/total_parameters": sum([
                result["parameter_count"] for result in phase_result["trained_models"].values()
            ])
        })
    
    def _analyze_integrated_training(self) -> Dict[str, Any]:
        """analysis comprehensivo del entrenamiento integrado."""
        
        analysis = {
            "consensus_effectiveness": {},
            "refinement_impact": {},
            "voting_performance": {},
            "scaling_efficiency": {},
            "cost_analysis": {}
        }
        
        total_cost = 0
        total_samples = 0
        
        for phase_name, results in self.phase_results.items():
            # analysis de consenso
            consensus_metrics = results["consensus_metrics"]
            analysis["consensus_effectiveness"][phase_name] = {
                "consensus_rate": consensus_metrics["consensus_rate"],
                "cache_efficiency": consensus_metrics["cache_hit_rate"],
                "quality_threshold_met": consensus_metrics["high_quality_samples"] / max(consensus_metrics["total_samples_generated"], 1)
            }
            
            # analysis de refinamiento
            refinement_metrics = results["refinement_metrics"]
            analysis["refinement_impact"][phase_name] = {
                "success_rate": refinement_metrics["successful_refinements"] / max(refinement_metrics["total_refinements"], 1),
                "average_iterations": refinement_metrics["average_iterations"],
                "improvement_score": refinement_metrics["improvement_score"]
            }
            
            # analysis de votaciÃÂ³n
            voting_metrics = results["voting_metrics"]
            analysis["voting_performance"][phase_name] = {
                "teacher_accuracy": np.mean(list(voting_metrics["teacher_success"].values())) if voting_metrics["teacher_success"] else 0,
                "critic_accuracy": np.mean(list(voting_metrics["critic_success"].values())) if voting_metrics["critic_success"] else 0
            }
            
            # analysis de escalamiento
            for model_name, model_result in results["trained_models"].items():
                analysis["scaling_efficiency"][f"{phase_name}_{model_name}"] = {
                    "parameters": model_result["parameter_count"],
                    "training_hours": model_result["training_time_hours"],
                    "final_loss": model_result["final_loss"],
                    "params_per_hour": model_result["parameter_count"] / max(model_result["training_time_hours"], 1)
                }
            
            # analysis de costos (estimado)
            phase_cost = self._estimate_phase_cost(phase_name, results)
            analysis["cost_analysis"][phase_name] = phase_cost
            total_cost += phase_cost["total_cost"]
            total_samples += results["dataset_size"]
        
        analysis["summary"] = {
            "total_cost_usd": total_cost,
            "total_samples_generated": total_samples,
            "cost_per_sample": total_cost / max(total_samples, 1),
            "overall_consensus_rate": np.mean([
                metrics["consensus_rate"] 
                for metrics in [r["consensus_metrics"] for r in self.phase_results.values()]
            ]),
            "overall_refinement_success": np.mean([
                metrics["successful_refinements"] / max(metrics["total_refinements"], 1)
                for metrics in [r["refinement_metrics"] for r in self.phase_results.values()]
            ])
        }
        
        return analysis
    
    def _estimate_phase_cost(self, phase_name: str, results: Dict[str, Any]) -> Dict[str, float]:
        """Estima costos de una fase."""
        
        # Costos base by fase (estimados)
        base_costs = {
            "micro_phase": 5000,
            "small_phase": 25000,
            "medium_phase": 100000,
            "large_phase": 500000
        }
        
        # Factores de costo
        dataset_factor = results["dataset_size"] / 1_000_000  # Cost per million samples
        refinement_factor = results["refinement_metrics"]["total_refinements"] / 1000
        
        total_cost = base_costs.get(phase_name, 10000) * (1 + dataset_factor * 0.1 + refinement_factor * 0.05)
        
        return {
            "base_cost": base_costs.get(phase_name, 10000),
            "dataset_cost": dataset_factor * 1000,
            "refinement_cost": refinement_factor * 500,
            "total_cost": total_cost
        }

# Utilidades de integration
class DatasetMerger:
    """Combina datasets de diferentes fases for entrenamiento conjunto."""
    
    @staticmethod
    def merge_phase_datasets(
        phase_results: Dict[str, Dict[str, Any]],
        target_sizes: Optional[Dict[str, int]] = None
    ) -> Dict[str, np.ndarray]:
        """Combina datasets de mÃÂºltiples fases."""
        
        all_inputs = []
        all_outputs = []
        all_qualities = []
        phase_labels = []
        
        for phase_name, results in phase_results.items():
            dataset = results.get("dataset", {})
            if "input_ids" not in dataset:
                continue
            
            # determine size a use
            current_size = len(dataset["input_ids"])
            target_size = target_sizes.get(phase_name, current_size) if target_sizes else current_size
            actual_size = min(current_size, target_size)
            
            # add data
            all_inputs.extend(dataset["input_ids"][:actual_size])
            all_outputs.extend(dataset["labels"][:actual_size])
            all_qualities.extend(dataset["quality_scores"][:actual_size])
            phase_labels.extend([phase_name] * actual_size)
        
        return {
            "input_ids": np.array(all_inputs),
            "labels": np.array(all_outputs),
            "quality_scores": np.array(all_qualities),
            "phase_labels": np.array(phase_labels)
        }

class ModelEvaluator:
    """EvalÃÂºa modelos entrenados with mÃÂ©tricas especÃÂ­ficas."""
    
    def __init__(self, evaluation_tasks: List[str]):
        self.evaluation_tasks = evaluation_tasks
        self.eval_results = {}
    
    async def evaluate_phase_models(
        self,
        phase_name: str,
        trained_models: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """EvalÃÂºa todos los modelos de una fase."""
        
        phase_eval_results = {}
        
        for model_name, model_info in trained_models.items():
            model_results = {}
            
            for task in self.evaluation_tasks:
                # Placeholder for evaluaciÃÂ³n real
                score = await self._evaluate_model_on_task(model_info, task)
                model_results[task] = score
            
            # calculate score average
            model_results["average_score"] = np.mean(list(model_results.values()))
            phase_eval_results[model_name] = model_results
        
        self.eval_results[phase_name] = phase_eval_results
        return phase_eval_results
    
    async def _evaluate_model_on_task(self, model_info: Dict[str, Any], task: str) -> float:
        """EvalÃÂºa model en task especÃÂ­fica."""
        
        # evaluaci?n de evaluaciÃÂ³n basada en parÃÂ¡metros del model
        param_count = model_info["parameter_count"]
        final_loss = model_info["final_loss"]
        
        # Score base according to size del model
        base_score = min(0.9, np.log10(param_count / 1e6) / 4)  # Log scaling
        
        # setting by loss
        loss_adjustment = max(0, (3.0 - final_loss) / 3.0)
        
        # Score end with ruido
        final_score = (base_score * 0.7 + loss_adjustment * 0.3) + np.random.normal(0, 0.05)
        
        return max(0.0, min(1.0, final_score))

# Scripts de uso
async def run_integrated_training_example():
    """example complete de entrenamiento integrado."""
    
    logger.info(" Iniciando ejemplo de entrenamiento integrado")
    
    # setup
    config = IntegratedTrainingConfig()
    
    # adjust setup for example fast
    for phase_config in config.dataset_configs.values():
        phase_config["size_range"] = (1000, 5000)  # Smaller datasets for demo
    
    for phase_config in config.training_configs.values():
        phase_config["total_steps"] = 100  # Fewer steps for demo
    
    # execute entrenamiento
    trainer = IntegratedCapibaraTrainer(config)
    results = await trainer.execute_integrated_training(target_scale="13B")
    
    # evaluate resultados
    evaluator = ModelEvaluator([
        "code_generation", "code_completion", "bug_fixing", "code_explanation"
    ])
    
    evaluation_results = {}
    for phase_name, phase_result in results["phase_results"].items():
        eval_result = await evaluator.evaluate_phase_models(
            phase_name, phase_result["trained_models"]
        )
        evaluation_results[phase_name] = eval_result
    
    # combine datasets
    merger = DatasetMerger()
    combined_dataset = merger.merge_phase_datasets(
        results["phase_results"],
        target_sizes={"micro_phase": 2000, "small_phase": 3000}
    )
    
    # show resultados finales
    logger.info(" Entrenamiento integrado completado!")
    logger.info(f" AnÃÂ¡lisis final: {results['final_analysis']['summary']}")
    avg_eval = np.mean([
        np.mean([model['average_score'] for model in phase.values()])
        for phase in evaluation_results.values()
    ])
    logger.info(f" EvaluaciÃÂ³n promedio: {avg_eval:.3f}")
    logger.info(f" Dataset combinado: {len(combined_dataset['input_ids']):,} samples")
    
    return results, evaluation_results, combined_dataset

# setup de producciÃÂ³n optimizada
def get_production_integrated_config() -> IntegratedTrainingConfig:
    """setup optimizada for producciÃÂ³n."""
    
    config = IntegratedTrainingConfig()
    
    # Teachers more potentes for producciÃÂ³n
    config.teacher_configs = {
        "micro_phase": {
            "models": [
                "microsoft/CodeT5p-770m", 
                "Salesforce/codet5-large", 
                "microsoft/unixcoder-base",
                "bigcode/santacoder"
            ],
            "weights": [0.3, 0.25, 0.25, 0.2],
            "parallel_calls": 4
        },
        "small_phase": {
            "models": [
                "bigcode/starcoder-1b",
                "microsoft/CodeT5p-2b", 
                "Salesforce/codet5-large",
                "microsoft/unixcoder-base",
                "facebook/incoder-1B"
            ],
            "weights": [0.35, 0.25, 0.2, 0.1, 0.1],
            "parallel_calls": 8
        },
        "medium_phase": {
            "models": [
                "bigcode/starcoder-3b",
                "microsoft/CodeT5p-6b",
                "facebook/incoder-6B",
                "Salesforce/codet5-large-ntp-py",
                "microsoft/unixcoder-base-nine"
            ],
            "weights": [0.4, 0.25, 0.2, 0.1, 0.05],
            "parallel_calls": 12
        },
        "large_phase": {
            "models": [
                "bigcode/starcoder-7b",
                "microsoft/CodeT5p-16b",
                "facebook/incoder-6B",
                "bigcode/santacoder-fim",
                "huggingface/CodeBERTa-small-v1"
            ],
            "weights": [0.5, 0.25, 0.15, 0.07, 0.03],
            "parallel_calls": 16
        }
    }
    
    # Datasets more grandes for producciÃÂ³n
    config.dataset_configs = {
        "micro_phase": {
            "size_range": (50_000_000, 100_000_000),
            "quality_threshold": 0.75,
            "batch_size": 128,
            "cache_enabled": True
        },
        "small_phase": {
            "size_range": (500_000_000, 1_000_000_000),
            "quality_threshold": 0.8,
            "batch_size": 256,
            "cache_enabled": True
        },
        "medium_phase": {
            "size_range": (5_000_000_000, 10_000_000_000),
            "quality_threshold": 0.85,
            "batch_size": 512,
            "cache_enabled": True
        },
        "large_phase": {
            "size_range": (20_000_000_000, 100_000_000_000),
            "quality_threshold": 0.9,
            "batch_size": 1024,
            "cache_enabled": True
        }
    }
    
    # Training more intensivo
    config.training_configs = {
        "micro_phase": {
            "learning_rate": 2e-4,
            "warmup_steps": 5000,
            "total_steps": 100000,
            "tpu_cores": 64
        },
        "small_phase": {
            "learning_rate": 1e-4,
            "warmup_steps": 10000,
            "total_steps": 200000,
            "tpu_cores": 256
        },
        "medium_phase": {
            "learning_rate": 5e-5,
            "warmup_steps": 20000,
            "total_steps": 500000,
            "tpu_cores": 1024
        },
        "large_phase": {
            "learning_rate": 2e-5,
            "warmup_steps": 50000,
            "total_steps": 1000000,
            "tpu_cores": 4096
        }
    }
    
    return config

# function principal for execution
async def main():
    """function principal de execution."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated Capibara Training")
    parser.add_argument("--target-scale", default="13B", help="Target model scale")
    parser.add_argument("--production", action="store_true", help="Use production config")
    parser.add_argument("--resume-from", help="Resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    
    args = parser.parse_args()
    
    # setup
    if args.production:
        config = get_production_integrated_config()
        logger.info(" Usando configuraciÃÂ³n de producciÃÂ³n")
    else:
        config = IntegratedTrainingConfig()
        logger.info(" Usando configuraciÃÂ³n de desarrollo")
    
    if args.eval_only:
        # only evaluaciÃÂ³n
        logger.info(" Ejecutando solo evaluaciÃÂ³n")
        # implement evaluaciÃÂ³n standalone
        return
    
    # Entrenamiento complete
    trainer = IntegratedCapibaraTrainer(config)
    results = await trainer.execute_integrated_training(
        target_scale=args.target_scale,
        resume_from=args.resume_from
    )
    
    # show resumen
    summary = results["final_analysis"]["summary"]
    
    logger.info("\n" + "="*60)
    logger.info(" ENTRENAMIENTO CAPIBARA INTEGRADO COMPLETADO")
    logger.info("="*60)
    logger.info(f" Target Scale: {args.target_scale}")
    logger.info(f" Costo Total: ${summary['total_cost_usd']:,.0f}")
    logger.info(f" Samples Generados: {summary['total_samples_generated']:,}")
    logger.info(f" Tasa de Consenso: {summary['overall_consensus_rate']:.1%}")
    logger.info(f" ÃÂxito de Refinamiento: {summary['overall_refinement_success']:.1%}")
    logger.info(f" Costo por Sample: ${summary['cost_per_sample']:.4f}")
    logger.info("="*60)
    
    return results

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


# Estrategias de Consensus Distilling Integradas for CapibaraGPT v3
#
# Este mÃÂ³dulo integra las estrategias avanzadas de consensus distilling,
# incluyendo peer voting and critic arbitration, optimizado for modelos 3B+.
#
# ExtraÃÂ­do and optimizado de consensus_destiling.py and train_300M_scale.py

import os
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
try:
    from jax import numpy as jnp
except ImportError:
    import numpy as jnp  # type: ignore[assignment]
try:
    import optax
except ImportError:
    optax = None  # Fallback

# Create fallback imports
try:
    from .training_config import ModelScale, TrainingConfigFactory
except ImportError:
    # Fallback implementation
    from enum import Enum
    class ModelScale(Enum):
        MICRO_300M = "300M"
        SMALL_3B = "3B"
        MEDIUM_30B = "30B"
        LARGE_1T = "1T"

logger = logging.getLogger(__name__)

@dataclass
class VotingMetrics:
    """MÃÂ©tricas del sistema de votaciÃÂ³n."""
    teacher_success_rates: Dict[int, float] = field(default_factory=dict)
    critic_accuracy_rates: Dict[int, float] = field(default_factory=dict)
    average_quality_score: float = 0.0
    consensus_confidence: float = 0.0
    total_votes: int = 0
    tie_break_count: int = 0

class TeacherModel:
    """Wrapper for modelos teacher."""
    
    def __init__(self, model_id: int, model_name: str, weight: float = 1.0):
        self.model_id = model_id
        self.model_name = model_name
        self.weight = weight
        self.success_count = 0
        self.total_calls = 0
        self.latency_history = []
        self.quality_history = []
    
    def update_performance(self, success: bool, latency: float, quality: float):
        """update mÃÂ©tricas de rendimiento."""
        self.total_calls += 1
        if success:
            self.success_count += 1
        self.latency_history.append(latency)
        self.quality_history.append(quality)
        
        # maintain only las ÃÂºltimas 100 mediciones
        if len(self.latency_history) > 100:
            self.latency_history.pop(0)
            self.quality_history.pop(0)
    
    @property
    def success_rate(self) -> float:
        """Tasa de ÃÂ©xito del teacher."""
        return self.success_count / max(self.total_calls, 1)
    
    @property
    def average_latency(self) -> float:
        """Latencia average."""
        return np.mean(self.latency_history) if self.latency_history else 0.0
    
    @property
    def average_quality(self) -> float:
        """Calidad average."""
        return np.mean(self.quality_history) if self.quality_history else 0.0

class CriticModel:
    """Wrapper for modelos critic."""
    
    def __init__(self, critic_id: int, model_name: str, weight: float = 1.0):
        self.critic_id = critic_id
        self.model_name = model_name
        self.weight = weight
        self.correct_predictions = 0
        self.total_predictions = 0
        self.evaluation_history = []
    
    def update_accuracy(self, was_correct: bool, confidence: float):
        """update precision del critic."""
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1
        self.evaluation_history.append((was_correct, confidence))
        
        # maintain only las ÃÂºltimas 50 evaluaciones
        if len(self.evaluation_history) > 50:
            self.evaluation_history.pop(0)
    
    @property
    def accuracy(self) -> float:
        """precision del critic."""
        return self.correct_predictions / max(self.total_predictions, 1)

class AdvancedVotingSystem:
    """
    Sistema de votaciÃÂ³n advanced with peer voting and critic arbitration.
    
    Integra las funcionalidades de consensus distilling for modelos 3B+
    with optimizaciones de rendimiento and calidad.
    """
    
    def __init__(self, n_teachers: int = 7, n_critics: int = 5, alpha: float = 0.08):
        self.n_teachers = n_teachers
        self.n_critics = n_critics
        self.alpha = alpha
        self.step = 0
        self.teacher_weights = {i: 1.0 for i in range(n_teachers)}
        self.critic_weights = {i: 1.0 for i in range(n_critics)}
        
        # Inicializar teachers and critics
        self.teachers = self._initialize_teachers()
        self.critics = self._initialize_critics()
        
        # setup de reset
        self.teacher_reset_every = 100
        self.critic_reset_every = 25
        self.confidence_threshold = 0.7
        
        # MÃÂ©tricas
        self.voting_metrics = VotingMetrics()
    
    def _initialize_teachers(self) -> List[TeacherModel]:
        """Inicializar modelos teacher."""
        teachers = []
        # En una implementation real, estos serÃÂ­an modelos reales
        # by now, simulamos with identificadores
        for i in range(self.n_teachers):
            teacher = TeacherModel(
                model_id=i,
                model_name=f"teacher_model_{i}",
                weight=1.0
            )
            teachers.append(teacher)
        return teachers
    
    def _initialize_critics(self) -> List[CriticModel]:
        """Inicializar modelos critic."""
        critics = []
        for i in range(self.n_critics):
            critic = CriticModel(
                critic_id=i,
                model_name=f"critic_model_{i}",
                weight=1.0
            )
            critics.append(critic)
        return critics
    
    async def enhanced_peer_vote(
        self, 
        prompt: str, 
        outputs: List[str], 
        model_scores: List[float],
        latencies: List[float]
    ) -> Tuple[str, int, Dict[str, Any]]:
        """
        VotaciÃÂ³n mejorada between peers with consideraciÃÂ³n de calidad and latencia.
        
        Args:
            prompt: Prompt de input
            outputs: Salidas de los modelos
            model_scores: Puntuaciones de calidad
            latencies: Latencias de generaciÃÂ³n
            
        Returns:
            Tupla with (mejor_output, ÃÂ­ndice_ganador, mÃÂ©tricas)
        """
        if not outputs:
            return "", -1, {}
        
        if len(outputs) == 1:
            return outputs[0], 0, {"single_output": True}
        
        # Real weighted voting using provided scores and latencies
        votes = defaultdict(float)
        if model_scores and latencies:
            max_latency = max(latencies) if max(latencies) > 0 else 1.0
            for i, score in enumerate(model_scores):
                latency_penalty = 1.0 - (latencies[i] / max_latency)
                weight = self.teacher_weights[i % len(self.teacher_weights)]
                votes[i] += weight * float(score) * max(latency_penalty, 0.0)
        elif model_scores:
            for i, score in enumerate(model_scores):
                weight = self.teacher_weights[i % len(self.teacher_weights)]
                votes[i] += weight * float(score)
        else:
            for i in range(len(outputs)):
                votes[i] += 1.0

        winner_idx = max(votes.keys(), key=lambda k: votes[k]) if votes else 0
        
        # calculate mÃÂ©tricas de consenso
        consensus_confidence = votes[winner_idx] / sum(votes.values()) if votes else 0.0
        
        # verify if necesitamos arbitraje de critics
        needs_arbitration = (
            len(votes) > 1 and 
            consensus_confidence < self.confidence_threshold
        )
        
        if needs_arbitration:
            winner_idx, arbitration_metrics = await self._critic_arbitration(
                prompt, outputs, model_scores, votes
            )
            consensus_confidence = arbitration_metrics.get("arbitration_confidence", consensus_confidence)
        
        # update mÃÂ©tricas de teachers
        await self._update_teacher_performance(
            votes, winner_idx, latencies, model_scores
        )
        
        metrics = {
            "consensus_confidence": consensus_confidence,
            "total_votes": len(votes),
            "needs_arbitration": needs_arbitration,
            "vote_distribution": dict(votes)
        }
        
        return outputs[winner_idx], winner_idx, metrics
    
    async def _critic_arbitration(
        self,
        prompt: str,
        outputs: List[str],
        scores: List[float],
        votes: Dict[int, float]
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Arbitraje by critics when hay empate or baja confianza.
        
        Args:
            prompt: Prompt original
            outputs: Outputs candidatos
            scores: Puntuaciones de calidad
            votes: Votos de teachers
            
        Returns:
            Tupla with (ÃÂ­ndice_ganador, mÃÂ©tricas_arbitraje)
        """
        critic_evaluations = []
        
        # obtain evaluaciones de critics
        for critic in self.critics:
            evaluation = await self._critic_evaluate(
                critic, prompt, outputs, scores
            )
            critic_evaluations.append(evaluation)
        
        # combine evaluaciones de critics
        final_scores = defaultdict(float)
        for eval_data in critic_evaluations:
            for idx, score in eval_data["scores"].items():
                final_scores[idx] += score * eval_data["weight"]
        
        # select ganador end
        if final_scores:
            winner_idx = max(final_scores.keys(), key=lambda k: final_scores[k])
            arbitration_confidence = final_scores[winner_idx] / sum(final_scores.values())
        else:
            # Fallback a votaciÃÂ³n original
            winner_idx = max(votes.keys(), key=lambda k: votes[k])
            arbitration_confidence = 0.5
        
        # update precision de critics
        for critic, evaluation in zip(self.critics, critic_evaluations):
            was_correct = evaluation["predicted_winner"] == winner_idx
            critic.update_accuracy(was_correct, evaluation["confidence"])
        
        metrics = {
            "arbitration_confidence": arbitration_confidence,
            "critic_evaluations": critic_evaluations,
            "final_scores": dict(final_scores)
        }
        
        return winner_idx, metrics
    
    async def _critic_evaluate(
        self,
        critic: CriticModel,
        prompt: str,
        outputs: List[str],
        base_scores: List[float]
    ) -> Dict[str, Any]:
        """evaluate outputs usando un critic."""
        # Simular evaluaciÃÂ³n del critic
        # En implementation real, usarÃÂ­a el model critic
        
        scores = {}
        for i, (output, base_score) in enumerate(zip(outputs, base_scores)):
            # Simular score del critic basado en el score base + ruido
            critic_score = base_score + np.random.normal(0, 0.1)
            critic_score = max(0.0, min(1.0, critic_score))  # Clamp [0,1]
            scores[i] = critic_score
        
        # predict ganador
        predicted_winner = max(scores.keys(), key=lambda k: scores[k])
        confidence = max(scores.values()) if scores else 0.5
        
        return {
            "critic_id": critic.critic_id,
            "scores": scores,
            "predicted_winner": predicted_winner,
            "confidence": confidence,
            "weight": critic.weight
        }
    
    async def _update_teacher_performance(
        self,
        votes: Dict[int, float],
        winner_idx: int,
        latencies: List[float],
        quality_scores: List[float]
    ):
        """update rendimiento de teachers basado en resultados."""
        for vote_idx, weight in votes.items():
            teacher = self.teachers[vote_idx]
            
            # determine if el teacher votÃÂ³ correctamente
            success = vote_idx == winner_idx
            
            # obtain latencia and calidad
            latency = latencies[vote_idx] if vote_idx < len(latencies) else 0.0
            quality = quality_scores[winner_idx] if winner_idx < len(quality_scores) else 0.0
            
            # update mÃÂ©tricas del teacher
            teacher.update_performance(success, latency, quality)
            
            # adjust peso basado en rendimiento
            self._adjust_teacher_weight(teacher)
    
    def _adjust_teacher_weight(self, teacher: TeacherModel):
        """adjust peso del teacher basado en rendimiento."""
        success_rate = teacher.success_rate
        avg_quality = teacher.average_quality
        avg_latency = teacher.average_latency
        
        # calculate new peso
        quality_factor = avg_quality if avg_quality > 0 else 0.5
        latency_factor = max(0.1, 1.0 - (avg_latency / 10.0))  # Penalizar alta latencia
        success_factor = success_rate
        
        new_weight = (quality_factor * 0.4 + success_factor * 0.4 + latency_factor * 0.2)
        
        # apply suavizado exponencial
        self.teacher_weights[teacher.model_id] = self.teacher_weights[teacher.model_id] * 0.8 + new_weight * 0.2
        
        # maintain peso en rank razonable
        self.teacher_weights[teacher.model_id] = max(0.1, min(2.0, self.teacher_weights[teacher.model_id]))
    
    def maybe_reset_weights(self):
        """Resetear pesos periÃÂ³dicamente for evitar convergencia prematura."""
        self.step += 1
        
        # Reset teachers
        if self.step % self.teacher_reset_every == 0:
            for teacher in self.teachers:
                teacher.weight = 1.0
                logger.info(f"Reset teacher {teacher.model_id} weight at step {self.step}")
        
        # Reset critics
        if self.step % self.critic_reset_every == 0:
            for critic in self.critics:
                critic.weight = 1.0
                logger.info(f"Reset critic {critic.critic_id} weight at step {self.step}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """obtain mÃÂ©tricas completas del sistema."""
        teacher_metrics = {}
        for teacher in self.teachers:
            teacher_metrics[teacher.model_id] = {
                "success_rate": teacher.success_rate,
                "weight": self.teacher_weights[teacher.model_id],
                "average_latency": teacher.average_latency,
                "average_quality": teacher.average_quality,
                "total_calls": teacher.total_calls
            }
        
        critic_metrics = {}
        for critic in self.critics:
            critic_metrics[critic.critic_id] = {
                "accuracy": critic.accuracy,
                "weight": self.critic_weights[critic.critic_id],
                "total_predictions": critic.total_predictions
            }
        
        return {
            "step": self.step,
            "teachers": teacher_metrics,
            "critics": critic_metrics,
            "voting_metrics": {
                "total_votes": self.voting_metrics.total_votes,
                "average_confidence": self.voting_metrics.consensus_confidence
            }
        }

class DistillationManager:
    """
    Manager for progressive distillation with consensus.
    
    Maneja la destilaciÃÂ³n progresiva usando outputs consensuados
    de teachers for train el model estudiante.
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3):
        self.temperature = temperature
        self.alpha = alpha  # Peso de la pÃÂ©rdida de distillation
        self.step = 0
    
    async def progressive_distillation(
        self,
        teacher_outputs: List[jnp.ndarray],
        student_output: jnp.ndarray,
        targets: jnp.ndarray,
        consensus_weights: Optional[List[float]] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        perform distillation progresiva with consensus.
        
        Args:
            teacher_outputs: Outputs de los teachers
            student_output: Output del estudiante
            targets: Targets verdaderos
            consensus_weights: Pesos de consensus (optional)
            
        Returns:
            Tupla with (pÃÂ©rdida_total, mÃÂ©tricas)
        """
        if not teacher_outputs:
            # only use pÃÂ©rdida cross-entropy if not hay teachers
            ce_loss = self._compute_ce_loss(student_output, targets)
            return ce_loss, {"ce_loss": float(ce_loss), "distill_loss": 0.0}
        
        # combine outputs de teachers usando consensus
        if consensus_weights is None:
            consensus_weights = [1.0] * len(teacher_outputs)
        
        # Normalizar pesos
        total_weight = sum(consensus_weights)
        normalized_weights = [w / total_weight for w in consensus_weights]
        
        # create ensemble de teachers
        ensemble_output = jnp.zeros_like(teacher_outputs[0])
        for output, weight in zip(teacher_outputs, normalized_weights):
            ensemble_output += weight * output
        
        # calculate pÃÂ©rdidas
        ce_loss = self._compute_ce_loss(student_output, targets)
        distill_loss = self._compute_distillation_loss(
            student_output, ensemble_output, self.temperature
        )
        
        # combine pÃÂ©rdidas
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * distill_loss
        
        # MÃÂ©tricas
        metrics = {
            "total_loss": float(total_loss),
            "ce_loss": float(ce_loss),
            "distill_loss": float(distill_loss),
            "alpha": self.alpha,
            "temperature": self.temperature,
            "n_teachers": len(teacher_outputs)
        }
        
        self.step += 1
        return total_loss, metrics
    
    def _compute_ce_loss(self, logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        """calculate pÃÂ©rdida cross-entropy."""
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    
    def _compute_distillation_loss(
        self, 
        student_logits: jnp.ndarray, 
        teacher_logits: jnp.ndarray, 
        temperature: float
    ) -> jnp.ndarray:
        """calculate pÃÂ©rdida de distillation with temperatura."""
        # apply temperatura a los logits
        student_soft = jax.nn.softmax(student_logits / temperature)
        teacher_soft = jax.nn.softmax(teacher_logits / temperature)
        
        # KL divergence between distribuciones soft
        kl_div = teacher_soft * jnp.log(teacher_soft / (student_soft + 1e-8))
        
        # scale by temperatura al cuadrado
        return (temperature ** 2) * kl_div.sum(axis=-1).mean()

def should_use_consensus_for_scale(model_scale_name: str) -> bool:
    """determine if use consensus distilling according to escala."""
    large_scales = ["3B", "7B", "13B", "30B", "50B", "65B", "130B", "300B", "650B", "1T"]
    return model_scale_name in large_scales

def create_consensus_system_for_scale(model_scale_name: str) -> Optional[AdvancedVotingSystem]:
    """create sistema de consensus for una escala especÃÂ­fica."""
    if not should_use_consensus_for_scale(model_scale_name):
        return None
    
    scale_configs = {
        "3B": {"n_teachers": 7, "n_critics": 5, "alpha": 0.08},
        "7B": {"n_teachers": 7, "n_critics": 5, "alpha": 0.08},
        "13B": {"n_teachers": 7, "n_critics": 5, "alpha": 0.08},
        "30B": {"n_teachers": 10, "n_critics": 7, "alpha": 0.06},
        "50B": {"n_teachers": 10, "n_critics": 7, "alpha": 0.06},
        "65B": {"n_teachers": 10, "n_critics": 7, "alpha": 0.06},
        "130B": {"n_teachers": 15, "n_critics": 10, "alpha": 0.05},
        "300B": {"n_teachers": 15, "n_critics": 10, "alpha": 0.05},
        "650B": {"n_teachers": 15, "n_critics": 10, "alpha": 0.05},
        "1T": {"n_teachers": 15, "n_critics": 10, "alpha": 0.05}
    }
    
    config = scale_configs.get(model_scale_name, {"n_teachers": 7, "n_critics": 5, "alpha": 0.08})
    
    return AdvancedVotingSystem(
        n_teachers=config["n_teachers"],
        n_critics=config["n_critics"],
        alpha=config["alpha"]
    )

def create_distillation_manager_for_scale(model_scale: ModelScale) -> Optional[DistillationManager]:
    """
    create manager de distillation apropiado for una escala.
    
    Args:
        model_scale: Escala del model
        
    Returns:
        Manager de distillation if debe usarse, None otherwise
    """
    should_use = TrainingConfigFactory.should_use_consensus_distilling(model_scale)
    
    if not should_use:
        return None
    
    # configure parÃÂ¡metros according to escala
    scale_configs = {
        ModelScale.SMALL_3B: {"temperature": 4.0, "alpha": 0.3},
        ModelScale.SMALL_7B: {"temperature": 4.0, "alpha": 0.3},
        ModelScale.SMALL_13B: {"temperature": 4.0, "alpha": 0.3},
        ModelScale.MEDIUM_30B: {"temperature": 3.5, "alpha": 0.5},
        ModelScale.MEDIUM_50B: {"temperature": 3.5, "alpha": 0.5},
        ModelScale.MEDIUM_65B: {"temperature": 3.5, "alpha": 0.5},
        ModelScale.LARGE_130B: {"temperature": 3.0, "alpha": 0.7},
        ModelScale.LARGE_300B: {"temperature": 3.0, "alpha": 0.7},
        ModelScale.LARGE_650B: {"temperature": 3.0, "alpha": 0.7},
        ModelScale.XLARGE_1T: {"temperature": 3.0, "alpha": 0.7}
    }
    
    config = scale_configs.get(model_scale, {"temperature": 4.0, "alpha": 0.3})
    
    return DistillationManager(
        temperature=config["temperature"],
        alpha=config["alpha"]
    ) 

# Fallback implementations for unified consensus classes
class UnifiedVotingConsensus:
    """Unified voting consensus implementation."""
    
    def __init__(self):
        self.vote_count = 0
        
    def get_consensus(self, outputs: List[str]) -> str:
        """Get consensus from multiple outputs through voting."""
        if not outputs:
            return "No outputs provided"
        
        # Simple majority voting fallback
        self.vote_count += 1
        return outputs[0] if outputs else "fallback_output"

class UnifiedRefinementConsensus:
    """Unified refinement consensus implementation."""
    
    def __init__(self):
        self.refinement_count = 0
        
    def refine_output(self, base_output: str, candidate_outputs: List[str]) -> str:
        """Refine output using consensus."""
        self.refinement_count += 1
        
        if not base_output:
            return candidate_outputs[0] if candidate_outputs else "refined_fallback"
        
        # Simple refinement: add improvement marker
        return base_output + " # Refined through consensus"

class UnifiedDistillationConsensus:
    """Unified distillation consensus implementation."""
    
    def __init__(self):
        self.distillation_count = 0
        
    def distill_knowledge(self, teacher_outputs: List[str]) -> str:
        """Distill knowledge from multiple teacher outputs."""
        self.distillation_count += 1
        
        if not teacher_outputs:
            return "No knowledge to distill"
        
        # Simple distillation: combine outputs
        return f"Distilled from {len(teacher_outputs)} teachers: {teacher_outputs[0]}"

class UnifiedCrossTeachingConsensus:
    """Unified cross-teaching consensus implementation."""
    
    def __init__(self):
        self.teaching_count = 0
        
    def cross_teach(self, outputs: List[str]) -> str:
        """Cross-teach between different models."""
        self.teaching_count += 1
        
        if not outputs:
            return "No outputs for cross-teaching"
        
        # Simple cross-teaching: select best output
        return f"Cross-taught result: {outputs[0]}" if outputs else "cross_teaching_fallback"


@dataclass
class ConsensusConfig:
    """Configuration for UnifiedConsensusStrategy."""
    consensus_threshold: float = 0.7
    validation_patience: int = 5
    consensus_window: int = 10
    min_agreement_ratio: float = 0.7


class UnifiedConsensusStrategy:
    """Unified consensus strategy (fallback implementation used by MetaConsensusSystem)."""

    def __init__(self, config: ConsensusConfig):
        self.config = config
        self.history: list = []

    def update(self, responses: list, context: dict | None = None) -> dict:
        """Record responses and return a simple majority-vote result."""
        self.history.append(responses)
        if not responses:
            return {"consensus": "", "agreement": 0.0, "success": False}
        # Simple majority: pick the most common response
        counts: dict = {}
        for r in responses:
            key = str(r)
            counts[key] = counts.get(key, 0) + 1
        best = max(counts, key=lambda k: counts[k])
        agreement = counts[best] / len(responses)
        return {
            "consensus": best,
            "agreement": agreement,
            "success": agreement >= self.config.min_agreement_ratio,
        }

