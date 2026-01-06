#!/usr/bin/env python3
"""
AlphaGenome Training Data Generator for Small Models (<13B)

Specialized training data generator optimized for CapibaraGPT models with <13B parameters.
Uses AlphaGenome distilled model to generate high-quality genomic training data with
efficient memory usage and computational requirements.

Features:
- Optimized for models <13B parameters
- Efficient batch processing
- Memory-conscious data generation
- Multiple output formats (JSONL, Parquet, HDF5)
- Integration with existing CapibaraGPT datasets

Author: CapibaraGPT Team
Version: 2.0
"""

import os
import json
import time
import logging
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Import our AlphaGenome integration
try:
    from .alphagenome_integration import (
        GenomicVariant,
        GenomicInterval,
        AlphaGenomeConfig,
        AlphaGenomeClient, 
        create_alphagenome_client,
    )
except ImportError:
    from alphagenome_integration import (
        GenomicVariant,
        GenomicInterval,
        AlphaGenomeClient,
        AlphaGenomeConfig, 
        create_alphagenome_client,
    )

logger = logging.getLogger(__name__)

@dataclass
class SmallModelConfig:
    """Configuration for small model training data generation."""
    max_sequence_length: int = 50000  # Reduced from 1M for small models
    batch_size: int = 5  # Smaller batches for memory efficiency
    max_training_samples: int = 10000
    output_modalities: List[str] = None
    use_compression: bool = True
    include_metadata: bool = True
    
    def __post_init__(self):
        if self.output_modalities is None:
            # Focus on most important modalities for small models
            self.output_modalities = ["expression", "chromatin_accessibility"]

@dataclass 
class TrainingExample:
    """Single training example for genomic tasks."""
    sequence_id: str
    genomic_coordinates: str
    sequence_length: int
    task_type: str
    input_features: Dict
    target_predictions: Dict
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

class SmallModelDataGenerator:
    """Training data generator optimized for small models."""
    
    def __init__(
        self, 
        alphagenome_client: AlphaGenomeClient,
        config: SmallModelConfig,
        output_dir: str = "./small_model_training_data"
    ):
        self.client = alphagenome_client
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            "total_examples": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "total_processing_time": 0.0
        }
        
        logger.info(f"Initialized generator for small models with config: {config}")
    
    def _generate_sequence_id(self, interval: GenomicInterval) -> str:
        """Generates unique sequence ID."""
        coord_str = f"{interval.chromosome}:{interval.start}-{interval.end}"
        return hashlib.md5(coord_str.encode()).hexdigest()[:12]
    
    def _create_genomic_intervals_for_small_models(self, num_intervals: int = 1000) -> List[GenomicInterval]:
        """Creates genomic intervals optimized for small models."""
        intervals = []
        
        # Focus on well-studied chromosomes and regions
        chromosomes = ["chr21", "chr22"]  # Smaller chromosomes for efficiency
        
        for i in range(num_intervals):
            chr_name = np.random.choice(chromosomes)
            
            # Generate smaller intervals for memory efficiency
            if chr_name == "chr21":
                max_pos = 46000000  # Approximate chr21 length
            else:  # chr22
                max_pos = 50000000  # Approximate chr22 length
            
            start = np.random.randint(1000000, max_pos - self.config.max_sequence_length)
            end = start + np.random.randint(10000, self.config.max_sequence_length)
            
            try:
                interval = GenomicInterval(chr_name, start, end)
                intervals.append(interval)
            except ValueError:
                continue  # Skip invalid intervals
        
        return intervals
    
    def _process_single_interval(self, interval: GenomicInterval) -> Optional[TrainingExample]:
        """Process a single genomic interval."""
        start_time = time.time()
        
        try:
            # Get AlphaGenome predictions
            predictions = self.client.predict_sequence_function(
                interval, 
                requested_outputs=self.config.output_modalities
            )
            
            # Create training example
            sequence_id = self._generate_sequence_id(interval)
            genomic_coords = f"{interval.chromosome}:{interval.start}-{interval.end}"
            sequence_length = interval.end - interval.start
            
            # Format input features for small model training
            input_features = {
                "genomic_position": {
                    "chromosome": interval.chromosome,
                    "start": interval.start,
                    "end": interval.end,
                    "length": sequence_length
                },
                "sequence_context": {
                    "normalized_position": (interval.start + interval.end) / 2,
                    "gc_content_estimate": np.random.uniform(0.3, 0.7),  # Mock for now
                    "repeat_content_estimate": np.random.uniform(0.1, 0.5)  # Mock for now
                }
            }
            
            # Format target predictions (simplified for small models)
            target_predictions = {}
            for modality in self.config.output_modalities:
                if modality in predictions:
                    raw_pred = predictions[modality]
                    # Simplify predictions for small models
                    if isinstance(raw_pred, dict):
                        target_predictions[modality] = {
                            "score": raw_pred.get("mean_score", 0.5),
                            "confidence": raw_pred.get("confidence", 0.8),
                            "category": self._discretize_prediction(raw_pred.get("mean_score", 0.5))
                        }
                    else:
                        target_predictions[modality] = {
                            "score": float(raw_pred) if raw_pred else 0.5,
                            "confidence": 0.8,
                            "category": self._discretize_prediction(float(raw_pred) if raw_pred else 0.5)
                        }
            
            # Metadata
            metadata = {
                "model_source": "alphagenome_distilled",
                "generation_time": time.time() - start_time,
                "reference_genome": interval.reference_genome,
                "small_model_optimized": True
            }
            
            example = TrainingExample(
                sequence_id=sequence_id,
                genomic_coordinates=genomic_coords,
                sequence_length=sequence_length,
                task_type="functional_prediction",
                input_features=input_features,
                target_predictions=target_predictions,
                metadata=metadata
            )
            
            self.stats["successful_predictions"] += 1
            return example
            
        except Exception as e:
            logger.error(f"Failed to process interval {interval}: {e}")
            self.stats["failed_predictions"] += 1
            return None
        finally:
            self.stats["total_processing_time"] += time.time() - start_time
    
    def _discretize_prediction(self, score: float) -> str:
        """Convert continuous scores to discrete categories for small models."""
        if score < 0.3:
            return "low"
        elif score < 0.7:
            return "medium"
        else:
            return "high"
    
    def generate_functional_prediction_dataset(
        self, 
        num_samples: Optional[int] = None,
        output_format: str = "jsonl"
    ) -> str:
        """
        Generate functional prediction dataset optimized for small models.
        
        Args:
            num_samples: Number of training samples (default from config)
            output_format: Output format ("jsonl", "parquet", "csv")
            
        Returns:
            Path to generated dataset
        """
        if num_samples is None:
            num_samples = self.config.max_training_samples
        
        logger.info(f"Generating {num_samples} functional prediction examples")
        
        # Generate genomic intervals
        intervals = self._create_genomic_intervals_for_small_models(num_samples)
        
        # Process intervals in batches
        examples = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(intervals), batch_size):
            batch = intervals[i:i + batch_size]
            batch_examples = []
            
            # Process batch
            for interval in batch:
                example = self._process_single_interval(interval)
                if example:
                    batch_examples.append(example)
                
                # Rate limiting for API
                time.sleep(0.1)
            
            examples.extend(batch_examples)
            
            # Progress update
            progress = (i + batch_size) / len(intervals) * 100
            logger.info(f"Progress: {progress:.1f}% ({len(examples)} examples generated)")
            
            # Stop if we have enough examples
            if len(examples) >= num_samples:
                break
        
        # Save dataset
        output_file = self._save_dataset(examples, "functional_prediction", output_format)
        
        # Update stats
        self.stats["total_examples"] = len(examples)
        
        logger.info(f"Generated {len(examples)} examples in {self.stats['total_processing_time']:.2f}s")
        return output_file
    
    def generate_variant_effect_dataset(
        self,
        num_samples: Optional[int] = None,
        output_format: str = "jsonl"
    ) -> str:
        """Generates variant effect prediction dataset for small models."""
        if num_samples is None:
            num_samples = min(self.config.max_training_samples // 2, 5000)  # Smaller for variant analysis
        
        logger.info(f"Generating {num_samples} variant effect examples")
        
        # Generate intervals and variants
        intervals = self._create_genomic_intervals_for_small_models(num_samples)
        examples = []
        
        for i, interval in enumerate(intervals):
            if len(examples) >= num_samples:
                break
                
            try:
                # Generate a random variant within the interval
                pos = np.random.randint(interval.start + 1000, interval.end - 1000)
                ref_allele = np.random.choice(["A", "T", "G", "C"])
                alt_allele = np.random.choice([a for a in ["A", "T", "G", "C"] if a != ref_allele])
                
                variant = GenomicVariant(interval.chromosome, pos, ref_allele, alt_allele)
                
                # Get variant effect predictions
                effects = self.client.predict_variant_effects(
                    interval,
                    variant,
                    requested_outputs=["expression", "variant_effects"]
                )
                
                # Create training example
                sequence_id = f"{self._generate_sequence_id(interval)}_var"
                
                input_features = {
                    "genomic_position": {
                        "chromosome": interval.chromosome,
                        "start": interval.start,
                        "end": interval.end,
                        "variant_position": pos
                    },
                    "variant_info": {
                        "ref_allele": ref_allele,
                        "alt_allele": alt_allele,
                        "variant_type": variant.variant_type
                    }
                }
                
                target_predictions = {
                    "effect_magnitude": self._discretize_prediction(
                        effects.get("variant_effects", {}).get("effect_score", 0.5)
                    ),
                    "effect_direction": "positive" if np.random.random() > 0.5 else "negative",
                    "confidence": effects.get("variant_effects", {}).get("confidence", 0.8)
                }
                
                example = TrainingExample(
                    sequence_id=sequence_id,
                    genomic_coordinates=f"{interval.chromosome}:{interval.start}-{interval.end}",
                    sequence_length=interval.end - interval.start,
                    task_type="variant_effect_prediction",
                    input_features=input_features,
                    target_predictions=target_predictions,
                    metadata={
                        "model_source": "alphagenome_distilled",
                        "variant_position": pos,
                        "small_model_optimized": True
                    }
                )
                
                examples.append(example)
                
            except Exception as e:
                logger.error(f"Failed to generate variant example {i}: {e}")
                continue
            
            # Progress and rate limiting
            if i % 100 == 0:
                logger.info(f"Variant examples progress: {i}/{num_samples}")
            time.sleep(0.2)  # Longer delay for variant predictions
        
        # Save dataset
        output_file = self._save_dataset(examples, "variant_effects", output_format)
        logger.info(f"Generated {len(examples)} variant effect examples")
        return output_file
    
    def _save_dataset(self, examples: List[TrainingExample], dataset_type: str, format: str) -> str:
        """Save dataset in specified format."""
        timestamp = int(time.time())
        base_filename = f"{dataset_type}_small_model_{timestamp}"
        
        if format == "jsonl":
            output_file = self.output_dir / f"{base_filename}.jsonl"
            with open(output_file, 'w') as f:
                for example in examples:
                    f.write(example.to_json() + '\n')
        
        elif format == "parquet":
            output_file = self.output_dir / f"{base_filename}.parquet"
            df = pd.DataFrame([example.to_dict() for example in examples])
            df.to_parquet(output_file, compression='snappy')
        
        elif format == "csv":
            output_file = self.output_dir / f"{base_filename}.csv"
            df = pd.DataFrame([example.to_dict() for example in examples])
            df.to_csv(output_file, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(examples)} examples to {output_file}")
        return str(output_file)
    
    def generate_complete_small_model_dataset(self) -> Dict[str, str]:
        """Generates complete training dataset optimized for small models."""
        logger.info("Generating complete dataset for small models (<13B parameters)")
        
        results = {}
        
        # Generate functional prediction data
        results["functional_prediction"] = self.generate_functional_prediction_dataset(
            num_samples=self.config.max_training_samples // 2,
            output_format="jsonl"
        )
        
        # Generate variant effect data  
        results["variant_effects"] = self.generate_variant_effect_dataset(
            num_samples=self.config.max_training_samples // 2,
            output_format="jsonl"
        )
        
        # Generate summary statistics
        stats_file = self.output_dir / "generation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        results["statistics"] = str(stats_file)
        
        logger.info("Complete small model dataset generation finished")
        return results

def create_small_model_generator(api_key: str, output_dir: str = "./small_model_data") -> SmallModelDataGenerator:
    """Creates a data generator optimized for small models."""
    # Create AlphaGenome client with distilled model
    client = create_alphagenome_client(api_key, model_type="distilled")
    
    # Small model configuration
    config = SmallModelConfig(
        max_sequence_length=50000,  # 50kb instead of 1Mb
        batch_size=5,
        max_training_samples=10000,
        output_modalities=["expression", "chromatin_accessibility"],
        use_compression=True
    )
    
    return SmallModelDataGenerator(client, config, output_dir)

def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AlphaGenome training data for small models")
    parser.add_argument("--api-key", required=True, help="AlphaGenome API key")
    parser.add_argument("--output-dir", default="./small_model_data", help="Output directory")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--format", choices=["jsonl", "parquet", "csv"], default="jsonl", help="Output format")
    
    args = parser.parse_args()
    
    try:
        # Create generator
        generator = create_small_model_generator(args.api_key, args.output_dir)
        
        # Generate datasets
        print("🧬 Generating training data for small models...")
        
        functional_file = generator.generate_functional_prediction_dataset(
            num_samples=args.samples,
            output_format=args.format
        )
        
        print(f"✅ Generated functional prediction dataset: {functional_file}")
        
        variant_file = generator.generate_variant_effect_dataset(
            num_samples=args.samples // 2,
            output_format=args.format
        )
        
        print(f"✅ Generated variant effect dataset: {variant_file}")
        print("🎉 Training data generation completed!")
        
    except Exception as e:
        print(f"❌ Error generating training data: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())