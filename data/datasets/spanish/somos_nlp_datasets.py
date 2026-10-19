"""
SomosNLP Datasets Manager for CapibaraGPT v3

Specialized manager for SomosNLP community datasets including:
- Open-source Spanish NLP datasets
- Hackathon 2022, 2023, and 2024 projects
- Clean Alpaca ES for instruction tuning
- #Somos600M project datasets
- Spanish cultural evaluation resources
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SomosNLPDatasets:
    """Manager for SomosNLP community datasets."""

    def __init__(self):
        """Initialize the SomosNLP datasets manager."""
        self.dataset_info = {
            # Core SomosNLP datasets
            "somos_clean_alpaca_es": {
                "name": "Somos Clean Alpaca ES",
                "description": "Curated Spanish instruction-tuning dataset (51.9k examples)",
                "size": "51.9k samples",
                "provider": "SomosNLP Community",
                "hf_dataset_name": "somosnlp/somos-clean-alpaca-es",
                "type": "instruction_tuning",
                "license": "MIT",
                "url": "https://huggingface.co/datasets/somosnlp/somos-clean-alpaca-es",
                "features": ["spanish_instructions", "quality_curated", "community_validated"],
                "languages": ["español"],
                "domain": "general"
            },

            # Traditional Spanish NLP datasets from the community
            "catalonia_independence": {
                "name": "Catalonia Independence Corpus",
                "description": "Sentiment classification dataset for Catalonia independence topic",
                "size": "~10k samples",
                "provider": "IXA-EHU",
                "hf_dataset_name": "catalonia_independence",
                "hf_contributor": "lewtan",
                "type": "sentiment_classification",
                "languages": ["catalán", "español"],
                "domain": "social_media",
                "paper": "https://www.aclweb.org/anthology/2020.lrec-1.171/"
            },

            "head_qa": {
                "name": "HEAD-QA",
                "description": "Spanish multiple choice medical questions",
                "size": "~2.7k questions",
                "provider": "University of Santiago de Compostela",
                "hf_dataset_name": "head_qa",
                "hf_contributor": "mariagrandury",
                "type": "question_answering",
                "languages": ["español"],
                "domain": "medical",
                "paper": "https://www.aclweb.org/anthology/P19-1092/"
            },

            "large_spanish_corpus": {
                "name": "Large Spanish Corpus",
                "description": "Large corpus for Spanish language modeling",
                "size": "Multi-GB",
                "provider": "José Cañete",
                "hf_dataset_name": "large_spanish_corpus",
                "hf_contributor": "lewtan",
                "type": "language_modeling",
                "languages": ["español"],
                "domain": "general"
            },

            "muchocine": {
                "name": "MuchoCine",
                "description": "Spanish movie reviews sentiment analysis",
                "size": "~3.9k reviews",
                "provider": "Universidad de Sevilla",
                "hf_dataset_name": "muchocine",
                "hf_contributor": "mapmeld",
                "type": "sentiment_classification",
                "languages": ["español"],
                "domain": "entertainment"
            },

            "spanish_billion_words": {
                "name": "Spanish Billion Words",
                "description": "Large Spanish corpus for pre-training",
                "size": "1B+ words",
                "provider": "SBWCE",
                "hf_dataset_name": "spanish_billion_words",
                "hf_contributor": "mariagrandury",
                "type": "language_modeling",
                "languages": ["español"],
                "domain": "general",
                "url": "https://crscardellino.github.io/SBWCE/"
            },

            "wikicorpus": {
                "name": "WikiCorpus",
                "description": "Wikipedia corpus for multiple languages including Spanish and Catalan",
                "size": "Multi-GB",
                "provider": "UPC",
                "hf_dataset_name": "wikicorpus",
                "hf_contributor": "albertvillanova",
                "type": "language_modeling",
                "languages": ["catalán", "español", "inglés"],
                "domain": "encyclopedia",
                "url": "https://www.cs.upc.edu/~nlp/wikicorpus/"
            },

            "ehealth_kd": {
                "name": "eHealth-KD",
                "description": "Spanish clinical named entity recognition",
                "size": "~1k documents",
                "provider": "Knowledge Learning",
                "hf_dataset_name": "ehealth_kd",
                "hf_contributor": "mariagrandury",
                "type": "named_entity_recognition",
                "languages": ["español"],
                "domain": "clinical",
                "url": "https://knowledge-learning.github.io/ehealthkd-2020/"
            }
        }

        # Hackathon projects and community initiatives
        self.hackathon_projects = {
            "hackathon_2022": {
                "theme": "NLP for Social Good",
                "participants": "500+",
                "projects": "15+",
                "focus": ["social_impact", "sustainability", "accessibility"],
                "outputs": ["models", "datasets", "demos"],
                "languages": ["español", "catalán", "gallego", "euskera"]
            },

            "hackathon_2023": {
                "theme": "Advancing Spanish NLP",
                "participants": "700+",
                "projects": "20+",
                "focus": ["multimodal", "reasoning", "cultural_awareness"],
                "outputs": ["instruction_datasets", "evaluation_benchmarks", "fine_tuned_models"]
            },

            "hackathon_2024": {
                "theme": "#Somos600M - Cultural Alignment",
                "participants": "800+",
                "projects": "25+",
                "focus": ["cultural_evaluation", "regional_varieties", "llm_alignment"],
                "outputs": ["cultural_benchmarks", "regional_datasets", "aligned_models"],
                "countries_represented": 29
            }
        }

        # Specialized Spanish corpora referenced by the community
        self.specialized_corpora = {
            "bascrawl": {
                "name": "BasCrawl",
                "description": "Basque language corpus for language modeling",
                "languages": ["euskera"],
                "domain": "general",
                "country": "España",
                "url": "https://doi.org/10.5281/zenodo.7313092"
            },

            "biomedical_spanish_embeddings": {
                "name": "Biomedical Spanish CBOW Word Embeddings",
                "description": "Spanish medical domain word embeddings",
                "languages": ["español"],
                "domain": "clinical",
                "country": "España",
                "url": "https://doi.org/10.5281/zenodo.7314041"
            },

            "csic_spanish_corpus": {
                "name": "CSIC Spanish Corpus",
                "description": "Academic Spanish corpus",
                "languages": ["español"],
                "domain": "academic",
                "country": "España",
                "url": "https://doi.org/10.5281/zenodo.7313126"
            },

            "infolibros_corpus": {
                "name": "InfoLibros Corpus",
                "description": "Literature corpus in Spanish",
                "languages": ["español"],
                "domain": "literature",
                "countries": ["Multiple"],
                "url": "https://doi.org/10.5281/zenodo.7313105"
            },

            "spanish_biomedical_corpus": {
                "name": "Spanish Biomedical Crawled Corpus",
                "description": "Large biomedical corpus in Spanish",
                "languages": ["español"],
                "domain": "clinical",
                "country": "España",
                "url": "https://doi.org/10.5281/zenodo.5513237"
            },

            "spanish_legal_corpus": {
                "name": "Spanish Legal Domain Corpora",
                "description": "Legal domain Spanish corpus",
                "languages": ["español"],
                "domain": "legal",
                "country": "España",
                "url": "https://doi.org/10.5281/zenodo.5495529",
                "github": "https://github.com/PlanTL-GOB-ES/lm-legal-es"
            },

            "tdx_thesis_corpus": {
                "name": "TDX Thesis Spanish Corpus",
                "description": "Academic thesis corpus",
                "languages": ["catalán", "español"],
                "domain": "academic",
                "country": "España",
                "url": "https://doi.org/10.5281/zenodo.7313149"
            }
        }

        # #Somos600M project specifics
        self.somos600m_project = {
            "name": "#Somos600M Project",
            "description": "Representing 600M Spanish speakers in AI systems",
            "mission": "Cultural alignment of LLMs for LATAM, Caribbean, and Spain",
            "paper": "https://arxiv.org/abs/2407.17479",
            "languages_represented": ["español", "português", "catalán", "gallego", "euskera"],
            "countries_covered": 29,
            "population_represented": "600M+ Spanish speakers, 265M+ Portuguese speakers",

            "initiatives": {
                "instruction_datasets": "Community-created instruction tuning datasets",
                "evaluation_leaderboard": "Open leaderboard for Spanish LLM evaluation",
                "corpus_collection": "Diverse regional Spanish varieties collection",
                "cultural_benchmarks": "Culture-specific evaluation benchmarks",
                "research_collaboration": "LATAM research group partnerships"
            },

            "datasets_created": {
                "somos_clean_alpaca_es": "51.9k curated Spanish instructions",
                "cultural_evaluation_sets": "Country-specific evaluation datasets",
                "regional_corpora": "Regional Spanish variety collections",
                "discrimination_analysis": "Social bias and discrimination datasets"
            }
        }

        # Technical specifications for integration
        self.technical_specs = {
            "data_formats": ["parquet", "json", "csv", "text"],
            "hf_integration": True,
            "argilla_annotation": True,
            "quality_validation": "community_curated",
            "evaluation_framework": "custom_spanish_benchmarks",
            "supported_tasks": [
                "instruction_tuning",
                "sentiment_analysis",
                "named_entity_recognition",
                "question_answering",
                "language_modeling",
                "cultural_evaluation",
                "bias_detection"
            ]
        }

    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get all available SomosNLP datasets."""
        return self.dataset_info

    def get_hackathon_projects(self) -> Dict[str, Dict[str, Any]]:
        """Get information about SomosNLP hackathon projects."""
        return self.hackathon_projects

    def get_specialized_corpora(self) -> Dict[str, Dict[str, Any]]:
        """Get specialized Spanish language corpora."""
        return self.specialized_corpora

    def get_somos600m_info(self) -> Dict[str, Any]:
        """Get information about the #Somos600M project."""
        return self.somos600m_project

    def search_datasets_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Search datasets by domain.

        Args:
            domain: Domain to search for (medical, legal, general, etc.)

        Returns:
            List of matching datasets
        """
        matches = []

        # Search in main datasets
        for dataset_id, info in self.dataset_info.items():
            if info.get("domain", "").lower() == domain.lower():
                matches.append({
                    "id": dataset_id,
                    "name": info["name"],
                    "description": info["description"],
                    "type": "main_dataset",
                    **info
                })

        # Search in specialized corpora
        for corpus_id, info in self.specialized_corpora.items():
            if info.get("domain", "").lower() == domain.lower():
                matches.append({
                    "id": corpus_id,
                    "name": info["name"],
                    "description": info["description"],
                    "type": "specialized_corpus",
                    **info
                })

        return matches

    def search_datasets_by_language(self, language: str) -> List[Dict[str, Any]]:
        """
        Search datasets by language.

        Args:
            language: Language to search for

        Returns:
            List of matching datasets
        """
        matches = []

        # Search in main datasets
        for dataset_id, info in self.dataset_info.items():
            if language.lower() in [lang.lower() for lang in info.get("languages", [])]:
                matches.append({
                    "id": dataset_id,
                    "type": "main_dataset",
                    **info
                })

        # Search in specialized corpora
        for corpus_id, info in self.specialized_corpora.items():
            if language.lower() in [lang.lower() for lang in info.get("languages", [])]:
                matches.append({
                    "id": corpus_id,
                    "type": "specialized_corpus",
                    **info
                })

        return matches

    def get_instruction_tuning_datasets(self) -> List[Dict[str, Any]]:
        """Get datasets suitable for instruction tuning."""
        instruction_datasets = []

        for dataset_id, info in self.dataset_info.items():
            if info.get("type") == "instruction_tuning" or "instruction" in info.get("features", []):
                instruction_datasets.append({
                    "id": dataset_id,
                    **info
                })

        return instruction_datasets

    def get_evaluation_datasets(self) -> List[Dict[str, Any]]:
        """Get datasets suitable for model evaluation."""
        eval_datasets = []

        evaluation_types = [
            "question_answering",
            "sentiment_classification",
            "named_entity_recognition",
            "cultural_evaluation"
        ]

        for dataset_id, info in self.dataset_info.items():
            if info.get("type") in evaluation_types:
                eval_datasets.append({
                    "id": dataset_id,
                    "evaluation_type": info.get("type"),
                    **info
                })

        return eval_datasets

    def get_cultural_alignment_resources(self) -> Dict[str, Any]:
        """Get resources for cultural alignment of LLMs."""
        return {
            "somos600m_project": self.somos600m_project,
            "cultural_datasets": [
                dataset for dataset in self.dataset_info.values()
                if "cultural" in dataset.get("features", []) or
                   "regional" in dataset.get("description", "").lower()
            ],
            "hackathon_contributions": {
                year: project for year, project in self.hackathon_projects.items()
                if "cultural" in project.get("focus", [])
            },
            "evaluation_framework": {
                "cultural_benchmarks": "Country-specific evaluation sets",
                "bias_detection": "Discrimination and bias analysis tools",
                "regional_evaluation": "Spanish variety-specific tests"
            }
        }

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about domains represented in SomosNLP datasets."""
        domain_counts = {}
        language_counts = {}

        # Count domains in main datasets
        for info in self.dataset_info.values():
            domain = info.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

            for lang in info.get("languages", []):
                language_counts[lang] = language_counts.get(lang, 0) + 1

        # Count domains in specialized corpora
        for info in self.specialized_corpora.values():
            domain = info.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

            for lang in info.get("languages", []):
                language_counts[lang] = language_counts.get(lang, 0) + 1

        return {
            "domain_distribution": domain_counts,
            "language_distribution": language_counts,
            "total_datasets": len(self.dataset_info),
            "total_corpora": len(self.specialized_corpora),
            "hackathon_editions": len(self.hackathon_projects),
            "languages_supported": list(language_counts.keys()),
            "domains_covered": list(domain_counts.keys())
        }

    def get_huggingface_integration_info(self) -> Dict[str, Any]:
        """Get information about Hugging Face integration."""
        hf_datasets = []

        for dataset_id, info in self.dataset_info.items():
            if "hf_dataset_name" in info:
                hf_datasets.append({
                    "id": dataset_id,
                    "hf_name": info["hf_dataset_name"],
                    "contributor": info.get("hf_contributor", "unknown"),
                    "type": info.get("type"),
                    "languages": info.get("languages", [])
                })

        return {
            "total_hf_datasets": len(hf_datasets),
            "datasets": hf_datasets,
            "integration_features": [
                "Direct HF dataset loading",
                "Community validation with Argilla",
                "Quality curation pipelines",
                "Multi-format support"
            ],
            "usage_example": {
                "load_dataset": "datasets.load_dataset('somosnlp/somos-clean-alpaca-es')",
                "validation": "Argilla annotation interface",
                "contribution": "Community-driven improvements"
            }
        }

    def get_community_impact(self) -> Dict[str, Any]:
        """Get information about SomosNLP community impact."""
        total_participants = sum(
            int(project.get("participants", "0").replace("+", ""))
            for project in self.hackathon_projects.values()
        )

        total_projects = sum(
            int(project.get("projects", "0").replace("+", ""))
            for project in self.hackathon_projects.values()
        )

        return {
            "community_size": "2000+ members",
            "hackathon_participants": f"{total_participants}+",
            "total_projects_created": f"{total_projects}+",
            "countries_represented": 30,
            "languages_supported": ["español", "portugués", "catalán", "gallego", "euskera"],
            "population_impact": "600M+ Spanish speakers, 265M+ Portuguese speakers",
            "research_outputs": [
                "51.9k instruction-tuned examples",
                "Multiple evaluation benchmarks",
                "Cultural alignment frameworks",
                "Bias detection tools"
            ],
            "academic_impact": {
                "papers_published": "Multiple research papers",
                "conferences": ["LXAI", "SEPLN", "NAACL"],
                "collaborations": "International research partnerships"
            }
        }

    def generate_usage_examples(self) -> Dict[str, str]:
        """Generate code examples for using SomosNLP datasets."""
        return {
            "load_alpaca_dataset": """
# Load SomosNLP Clean Alpaca ES dataset
from datasets import load_dataset

dataset = load_dataset("somosnlp/somos-clean-alpaca-es")
logger.info(f"Dataset size: {len(dataset['train'])}")
logger.info(f"Example: {dataset['train'][0]}")
            """,

            "filter_by_quality": """
# Filter by quality validation
filtered = dataset.filter(lambda x: x['prediction'][0]['label'] == 'ALL GOOD')
logger.info(f"High quality samples: {len(filtered)}")
            """,

            "load_medical_dataset": """
# Load HEAD-QA medical dataset
head_qa = load_dataset("head_qa", "es")
logger.info(f"Medical questions: {len(head_qa['train'])}")
            """,

            "cultural_evaluation": """
# Example cultural evaluation
from somosnlp import CulturalEvaluator

evaluator = CulturalEvaluator(
    countries=["España", "México", "Argentina", "Colombia"],
    domains=["social", "legal", "educational"]
)

results = evaluator.evaluate_model(model, cultural_test_set)
            """
        }


# Factory function
def get_somos_nlp_datasets() -> SomosNLPDatasets:
    """Get SomosNLP datasets manager."""
    return SomosNLPDatasets()


def main():
    """Main function for module execution."""
    logger.info("Module somos_nlp_datasets.py starting")
    return True

if __name__ == "__main__":
    main()
