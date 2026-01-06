"""
Genomic Datasets for CapibaraGPT v2

This module provides acceso a datasets genómicos especializados para 
entrenamiento en bioinformática y análisis genético.

Características:
- Secuencias de ADN/ARN/proteínas
- Anotaciones genómicas
- Datos de expresión génica
- Variantes genéticas
- Análisis filogenético
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Genomic dataset configurations
GENOMIC_DATASETS = {
    'dna_sequences': {
        'description': 'DNA sequences from various organisms',
        'format': 'FASTA',
        'size': 'very_large',
        'organisms': ['human', 'mouse', 'drosophila', 'ecoli'],
        'sequence_types': ['coding', 'non_coding', 'regulatory']
    },
    'protein_sequences': {
        'description': 'Protein sequences and structures',
        'format': 'FASTA + PDB',
        'size': 'large',
        'includes': ['sequence', 'structure', 'function_annotation']
    },
    'gene_expression': {
        'description': 'Gene expression data from RNA-seq',
        'format': 'matrix',
        'size': 'large',
        'conditions': ['normal', 'disease', 'treatment', 'development']
    },
    'genetic_variants': {
        'description': 'SNPs and other genetic variants',
        'format': 'VCF',
        'size': 'very_large',
        'populations': ['1000genomes', 'gnomad', 'ukbiobank']
    },
    'phylogenetic': {
        'description': 'Phylogenetic trees and evolutionary data',
        'format': 'newick + metadata',
        'size': 'medium',
        'scope': ['species_level', 'gene_families', 'viral_evolution']
    }
}

class GenomicSequenceProcessor:
    """Genomic sequence processor."""
    
    def __init__(self):
        self.nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.amino_acid_map = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15,
            'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20  # X for unknown
        }
    
    def encode_dna_sequence(self, sequence: str) -> List[int]:
        """Encodes DNA sequence to numbers."""
        return [self.nucleotide_map.get(base.upper(), 4) for base in sequence]
    
    def encode_protein_sequence(self, sequence: str) -> List[int]:
        """Encodes protein sequence to numbers."""
        return [self.amino_acid_map.get(aa.upper(), 20) for aa in sequence]
    
    def one_hot_encode_dna(self, sequence: str) -> np.ndarray:
        """Codifica ADN en formato one-hot."""
        encoded = self.encode_dna_sequence(sequence)
        one_hot = np.zeros((len(encoded), 5))  # 4 bases + N
        for i, base_idx in enumerate(encoded):
            one_hot[i, base_idx] = 1
        return one_hot
    
    def reverse_complement(self, sequence: str) -> str:
        """Gets the reverse complement of a DNA sequence."""
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        complement = ''.join(complement_map.get(base.upper(), 'N') for base in sequence)
        return complement[::-1]

class GenomicDatasetLoader:
    """Main loader for genomic datasets."""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.processor = GenomicSequenceProcessor()
        self.available_datasets = GENOMIC_DATASETS.copy()
        self.loaded_datasets = {}
        
        logger.info("🧬 Genomic dataset loader initialized")
        logger.info(f"   📁 Base path: {self.base_path}")
        logger.info(f"   📊 Available datasets: {len(self.available_datasets)}")
    
    def list_datasets(self) -> List[str]:
        """Lists available genomic datasets."""
        return list(self.available_datasets.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Gets detailed information about a dataset."""
        return self.available_datasets.get(dataset_name)
    
    def load_dna_sequences(self, organism: str = 'human', 
                          sequence_type: str = 'coding',
                          max_sequences: int = 1000) -> Dict[str, Any]:
        """Loads DNA sequences."""
        logger.info(f"📥 Loading DNA sequences: {organism} ({sequence_type})")
        
        # En implementación real, cargaría desde archivos FASTA
        # Por ahora, generar datos simulados
        sequences = []
        for i in range(min(max_sequences, 100)):  # Limitar for demo
            # Generar secuencia aleatoria
            length = np.random.randint(100, 2000)
            bases = ['A', 'T', 'G', 'C']
            sequence = ''.join(np.random.choice(bases, length))
            
            sequences.append({
                'id': f"{organism}_{sequence_type}_{i:04d}",
                'sequence': sequence,
                'length': length,
                'organism': organism,
                'type': sequence_type,
                'encoded': self.processor.encode_dna_sequence(sequence)
            })
        
        return {
            'dataset': 'dna_sequences',
            'organism': organism,
            'sequence_type': sequence_type,
            'count': len(sequences),
            'sequences': sequences,
            'metadata': {
                'format': 'processed',
                'encoding': 'nucleotide_integers',
                'processor': 'GenomicSequenceProcessor'
            }
        }
    
    def load_protein_sequences(self, max_sequences: int = 500) -> Dict[str, Any]:
        """Loads protein sequences."""
        logger.info(f"📥 Loading protein sequences (max: {max_sequences})")
        
        sequences = []
        amino_acids = list(self.processor.amino_acid_map.keys())[:-1]  # Excluir X
        
        for i in range(min(max_sequences, 50)):  # Limitar for demo
            length = np.random.randint(50, 500)
            sequence = ''.join(np.random.choice(amino_acids, length))
            
            sequences.append({
                'id': f"protein_{i:04d}",
                'sequence': sequence,
                'length': length,
                'encoded': self.processor.encode_protein_sequence(sequence),
                'predicted_function': f"function_{np.random.randint(1, 10)}"
            })
        
        return {
            'dataset': 'protein_sequences',
            'count': len(sequences),
            'sequences': sequences,
            'metadata': {
                'format': 'processed',
                'encoding': 'amino_acid_integers'
            }
        }
    
    def load_gene_expression_data(self, condition: str = 'normal',
                                 num_genes: int = 1000,
                                 num_samples: int = 100) -> Dict[str, Any]:
        """Loads gene expression data."""
        logger.info(f"📥 Loading gene expression data: {condition}")
        
        # Generar matriz de expresión simulada
        expression_matrix = np.random.lognormal(mean=0, sigma=1, size=(num_genes, num_samples))
        
        gene_names = [f"GENE_{i:04d}" for i in range(num_genes)]
        sample_names = [f"{condition}_sample_{i:03d}" for i in range(num_samples)]
        
        return {
            'dataset': 'gene_expression',
            'condition': condition,
            'expression_matrix': expression_matrix,
            'gene_names': gene_names,
            'sample_names': sample_names,
            'shape': expression_matrix.shape,
            'metadata': {
                'format': 'matrix',
                'normalization': 'log_normal',
                'condition': condition
            }
        }
    
    def load_genetic_variants(self, population: str = '1000genomes',
                             chromosome: str = 'chr1',
                             max_variants: int = 10000) -> Dict[str, Any]:
        """Loads variantes genéticas."""
        logger.info(f"📥 Loading genetic variants: {population} ({chromosome})")
        
        variants = []
        for i in range(min(max_variants, 1000)):  # Limitar for demo
            position = np.random.randint(1000000, 249000000)  # Posiciones en chr1
            ref_allele = np.random.choice(['A', 'T', 'G', 'C'])
            alt_allele = np.random.choice([a for a in ['A', 'T', 'G', 'C'] if a != ref_allele])
            
            variants.append({
                'id': f"rs{1000000 + i}",
                'chromosome': chromosome,
                'position': position,
                'ref_allele': ref_allele,
                'alt_allele': alt_allele,
                'frequency': np.random.random(),
                'population': population
            })
        
        return {
            'dataset': 'genetic_variants',
            'population': population,
            'chromosome': chromosome,
            'count': len(variants),
            'variants': variants,
            'metadata': {
                'format': 'processed_vcf',
                'population': population
            }
        }
    
    def create_training_batch(self, dataset_type: str, batch_size: int = 32) -> Dict[str, Any]:
        """Creates a training batch for a dataset type."""
        if dataset_type == 'dna_sequences':
            data = self.load_dna_sequences(max_sequences=batch_size)
            sequences = [seq['encoded'] for seq in data['sequences']]
            
            # Pad sequences to same length
            max_length = max(len(seq) for seq in sequences)
            padded_sequences = []
            for seq in sequences:
                padded = seq + [4] * (max_length - len(seq))  # Pad with N
                padded_sequences.append(padded)
            
            return {
                'type': 'dna_sequences',
                'batch_size': len(padded_sequences),
                'sequences': np.array(padded_sequences),
                'max_length': max_length
            }
        
        elif dataset_type == 'gene_expression':
            data = self.load_gene_expression_data(num_samples=batch_size)
            return {
                'type': 'gene_expression',
                'batch_size': batch_size,
                'expression_data': data['expression_matrix'],
                'gene_count': data['shape'][0]
            }
        
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

def print_dataset_summary():
    """Prints summary of available genomic datasets."""
    print("🧬 GENOMIC DATASETS SUMMARY")
    print("=" * 50)
    
    for name, info in GENOMIC_DATASETS.items():
        print(f"\n📊 {name.upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Format: {info['format']}")
        print(f"   Size: {info['size']}")
        
        if 'organisms' in info:
            print(f"   Organisms: {', '.join(info['organisms'])}")
        if 'conditions' in info:
            print(f"   Conditions: {', '.join(info['conditions'])}")

def get_genomic_loader() -> GenomicDatasetLoader:
    """Factory function to get the genomic loader."""
    return GenomicDatasetLoader()

def list_available_genomic_datasets() -> List[str]:
    """Quick list of available genomic datasets."""
    return list(GENOMIC_DATASETS.keys())

# Export main components
__all__ = [
    'GenomicDatasetLoader',
    'GenomicSequenceProcessor',
    'get_genomic_loader',
    'list_available_genomic_datasets',
    'print_dataset_summary',
    'GENOMIC_DATASETS'
]

# Initialize logging
logger.info("🧬 Genomic datasets module loaded successfully")

if __name__ == "__main__":
    print_dataset_summary()