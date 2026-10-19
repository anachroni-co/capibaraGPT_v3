"""Module for managing pure mathematics datasets."""

from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

@dataclass
class MathDataManager:
    """Manager for pure mathematics datasets."""

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the mathematics datasets manager.

        Args:
            base_dir: Base directory for storing the datasets
        """
        self.base_dir = Path(base_dir) if base_dir else Path("data/math")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Registry of datasets
        self.datasets = {
            "math-competition": {
                "name": "MATH Competition Dataset",
                "description": "Premier mathematical competition dataset with 12,500+ problems",
                "quality": 9.9,
                "size_gb": 850,
                "size_human": "850GB",
                "categories": [
                    "Prealgebra", "Algebra", "Number Theory",
                    "Counting & Probability", "Geometry",
                    "Intermediate Algebra", "Precalculus"
                ],
                "authority": ["UC Berkeley", "Carnegie Mellon", "Stanford"],
                "features": {
                    "problems": 12500,
                    "solutions": "LaTeX + natural language",
                    "difficulty": "high school to undergraduate",
                    "auxiliary": "AMPS dataset included"
                },
                "access_info": {
                    "url": "https://github.com/hendrycks/math",
                    "mirror_urls": [
                        "https://huggingface.co/datasets/hendrycks/competition_math",
                        "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"
                    ],
                    "download_command": "git clone https://github.com/hendrycks/math.git",
                    "alternative_download": "wget https://people.eecs.berkeley.edu/~hendrycks/MATH.tar",
                    "license": "MIT License",
                    "requires_auth": False,
                    "citation": "@article{hendrycks2021measuring, title={Measuring Mathematical Problem Solving with the MATH Dataset}, author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt}, journal={arXiv preprint arXiv:2103.03874}, year={2021}}",
                    "paper_url": "https://arxiv.org/abs/2103.03874"
                },
                "file_structure": {
                    "train": "12,500 training problems",
                    "test": "5,000 test problems",
                    "format": "JSON files with problem statement, solution, and answer",
                    "encoding": "UTF-8"
                }
            },
            "natural-proofs": {
                "name": "NaturalProofs Dataset",
                "description": "Large-scale mathematical theorem proving dataset",
                "quality": 9.8,
                "size_gb": 1200,
                "size_human": "1.2TB",
                "content": {
                    "theorems": 20000,
                    "definitions": 12500,
                    "additional_pages": 1000
                },
                "sources": ["ProofWiki", "Stacks Project"],
                "authority": ["University of Washington", "NYU", "Allen Institute"],
                "features": {
                    "language": "symbolic + natural",
                    "tasks": ["retrieval", "generation"],
                    "evaluation": "zero-shot generalization"
                },
                "access_info": {
                    "url": "https://github.com/wellecks/naturalproofs",
                    "download_url": "https://drive.google.com/file/d/1j8wZKV3GwZF-KV3HZJ8GpX3g9Z9gKG9K/view",
                    "download_command": "gdown 1j8wZKV3GwZF-KV3HZJ8GpX3g9Z9gKG9K",
                    "huggingface_url": "https://huggingface.co/datasets/wellecks/naturalproofs",
                    "license": "Apache 2.0",
                    "requires_auth": False,
                    "citation": "@inproceedings{welleck2021naturalproofs, title={NaturalProofs: Mathematical Theorem Proving in Natural Language}, author={Sean Welleck and Jiacheng Liu and Ronan Le Bras and Hannaneh Hajishirzi and Yejin Choi and Kyunghyun Cho}, booktitle={Advances in Neural Information Processing Systems}, year={2021}}",
                    "paper_url": "https://arxiv.org/abs/2104.01112"
                },
                "file_structure": {
                    "theorems": "JSON files with theorem statements and proofs",
                    "definitions": "Structured mathematical definitions",
                    "format": "Natural language + symbolic notation",
                    "encoding": "UTF-8"
                }
            },
            "deepmath": {
                "name": "DeepMath Collection",
                "description": "Multi-source pure mathematics compilation",
                "quality": 9.7,
                "size_gb": 950,
                "size_human": "950GB",
                "components": {
                    "identities": {
                        "famous": 71,
                        "versions": 400000
                    },
                    "symbolic": ["formula retrieval", "conjecture generation"],
                    "proving": ["formal verification", "pure reasoning"]
                },
                "authority": ["Google DeepMind", "Academic institutions"],
                "access_info": {
                    "url": "https://github.com/google-deepmind/deepmath",
                    "download_urls": [
                        "https://storage.googleapis.com/deepmath-data/identities.tar.gz",
                        "https://storage.googleapis.com/deepmath-data/symbolic-math.tar.gz"
                    ],
                    "download_commands": [
                        "wget https://storage.googleapis.com/deepmath-data/identities.tar.gz",
                        "wget https://storage.googleapis.com/deepmath-data/symbolic-math.tar.gz"
                    ],
                    "kaggle_url": "https://www.kaggle.com/datasets/google/deepmath",
                    "license": "Apache 2.0",
                    "requires_auth": False,
                    "citation": "@article{lample2019deep, title={Deep Learning for Symbolic Mathematics}, author={Guillaume Lample and Francois Charton}, journal={arXiv preprint arXiv:1912.01412}, year={2019}}",
                    "paper_url": "https://arxiv.org/abs/1912.01412"
                },
                "file_structure": {
                    "identities": "Mathematical identities in symbolic form",
                    "symbolic": "Symbolic mathematics expressions",
                    "format": "Text files with mathematical expressions",
                    "encoding": "UTF-8"
                }
            }
        }

    def get_dataset_info(self, dataset_id: str) -> Dict:
        """
        Obtain information for a specific dataset.

        Args:
            dataset_id: Identifier of the dataset

        Returns:
            Dict with dataset information
        """
        return self.datasets.get(dataset_id, {})

    def get_download_info(self, dataset_id: str) -> Dict:
        """
        Obtain specific download information for a dataset.

        Args:
            dataset_id: Identifier of the dataset

        Returns:
            Dict with access and download information
        """
        dataset = self.datasets.get(dataset_id, {})
        return dataset.get("access_info", {})

    def get_all_datasets(self) -> List[Dict]:
        """
        Obtain information for all datasets.

        Returns:
            List of dictionaries with information for each dataset
        """
        return list(self.datasets.values())

    def get_total_size_gb(self) -> float:
        """
        Calculate the total size of all datasets in GB.

        Returns:
            Total size in GB
        """
        return sum(
            dataset.get("size_gb", 0)
            for dataset in self.datasets.values()
        )

    def get_average_quality(self) -> float:
        """
        Calculate the average quality of the datasets.

        Returns:
            Average quality
        """
        qualities = [
            dataset.get("quality", 0)
            for dataset in self.datasets.values()
        ]
        return sum(qualities) / len(qualities) if qualities else 0.0

    def generate_readme(self, dataset_id: str) -> str:
        """
        Generate a detailed README file for a specific dataset.

        Args:
            dataset_id: Identifier of the dataset

        Returns:
            README content in markdown format
        """
        dataset = self.datasets.get(dataset_id, {})
        if not dataset:
            return "Dataset not found"

        access = dataset.get("access_info", {})
        structure = dataset.get("file_structure", {})

        readme_content = f"""# {dataset['name']}

## General Description
{dataset['description']}

## Dataset Information
- **Quality**: {dataset['quality']}/10
- **Size**: {dataset.get('size_human', 'N/A')}
- **Authorities**: {', '.join(dataset.get('authority', []))}

## Access and Download

### Main URLs
- **Main URL**: {access.get('url', 'N/A')}
- **Paper**: {access.get('paper_url', 'N/A')}

### Download Commands
```bash
{access.get('download_command', 'Not available')}
```

### Alternative URLs
{chr(10).join(f"- {url}" for url in access.get('mirror_urls', []))}

## License
{access.get('license', 'Not specified')}

## File Structure
{chr(10).join(f"- **{k}**: {v}" for k, v in structure.items())}

## Citation
```bibtex
{access.get('citation', 'Not available')}
```

## Usage Notes
- Authentication required: {'Yes' if access.get('requires_auth', False) else 'No'}
- Encoding format: {structure.get('encoding', 'UTF-8')}
"""
        return readme_content
