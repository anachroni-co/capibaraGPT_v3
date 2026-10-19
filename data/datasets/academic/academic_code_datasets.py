"""
Module for handling academic and code datasets.
"""

import os
import git
import json
import arxiv
import logging
import requests
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from datasets import load_dataset
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

@dataclass
class AcademicPaper:
    """Metadata for an academic paper."""
    id: str
    title: str
    authors: List[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    categories: List[str] = None
    published: Optional[str] = None
    citations: Optional[int] = None

class AcademicCodeDatasetManager:
    """Manager for academic and code datasets."""

    # Base URLs
    ARXIV_BASE_URL = "https://arxiv.org/abs/"
    PWC_BASE_URL = "https://paperswithcode.com/api/v1/"
    OPENALEX_BASE_URL = "https://api.openalex.org/"
    CONNECTED_PAPERS_BASE_URL = "https://api.connectedpapers.com/v1/"

    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the academic and code datasets manager.

        Args:
            base_dir: Base directory for storing the datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.arxiv_dir = self.base_dir / "arxiv"
        self.pwc_dir = self.base_dir / "papers_with_code"
        self.openalex_dir = self.base_dir / "openalex"
        self.connected_papers_dir = self.base_dir / "connected_papers"
        self.code_dir = self.base_dir / "code"

        for dir_path in [self.arxiv_dir, self.pwc_dir, self.openalex_dir,
                        self.connected_papers_dir, self.code_dir]:
            dir_path.mkdir(exist_ok=True)

    def download_arxiv_papers(self, query: str, max_results: int = 100) -> List[str]:
        """
        Download papers from arXiv based on a query.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of paths to downloaded papers
        """
        try:
            # Search papers
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            downloaded_paths = []
            for paper in tqdm(search.results(), desc="Downloading papers from arXiv"):
                # Create directory for the paper
                paper_dir = self.arxiv_dir / paper.get_short_id()
                paper_dir.mkdir(exist_ok=True)

                # Save metadata
                metadata = {
                    "id": paper.get_short_id(),
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "categories": paper.categories,
                    "published": paper.published.isoformat(),
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url
                }

                metadata_path = paper_dir / "metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                # Download PDF
                pdf_path = paper_dir / "paper.pdf"
                if not pdf_path.exists():
                    paper.download_pdf(str(pdf_path))

                downloaded_paths.append(str(paper_dir))

            return downloaded_paths

        except Exception as e:
            logger.error(f"Error downloading papers from arXiv: {e}")
            raise

    def download_code_datasets(self, dataset_configs: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Download code datasets.

        Args:
            dataset_configs: List of dataset configurations
                           Example: [
                               {
                                   "name": "bigcode/the-stack",
                                   "subset": "data",
                                   "split": "train"
                               },
                               {
                                   "name": "openai/human-eval",
                                   "repo": "openai/human-eval"
                               }
                           ]

        Returns:
            Dictionary with paths to downloaded datasets
        """
        try:
            downloaded_paths = {}

            for config in dataset_configs:
                dataset_name = config["name"]
                dataset_dir = self.code_dir / dataset_name.replace("/", "_")
                dataset_dir.mkdir(exist_ok=True)

                if "repo" in config:
                    # Clone Git repository
                    repo_url = f"https://github.com/{config['repo']}.git"
                    git.Repo.clone_from(repo_url, dataset_dir)
                    downloaded_paths[dataset_name] = str(dataset_dir)
                else:
                    # Download dataset from Hugging Face
                    dataset = load_dataset(
                        dataset_name,
                        config.get("subset"),
                        split=config.get("split"),
                        cache_dir=str(dataset_dir)
                    )

                    # Save metadata
                    metadata = {
                        "name": dataset_name,
                        "subset": config.get("subset"),
                        "split": config.get("split"),
                        "features": list(dataset.features.keys()),
                        "num_rows": len(dataset),
                        "description": dataset.description,
                        "citation": dataset.citation,
                        "homepage": dataset.homepage
                    }

                    metadata_path = dataset_dir / "metadata.json"
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                    downloaded_paths[dataset_name] = str(dataset_dir)

            return downloaded_paths

        except Exception as e:
            logger.error(f"Error downloading code datasets: {e}")
            raise

    def download_papers_with_code(self, query: str) -> List[str]:
        """
        Download papers and code from Papers with Code.

        Args:
            query: Search query

        Returns:
            List of paths to downloaded papers/code
        """
        try:
            # Search papers
            url = f"{self.PWC_BASE_URL}papers/search"
            params = {"q": query}
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()

            downloaded_paths = []
            for paper in results["results"]:
                # Create directory for the paper
                paper_dir = self.pwc_dir / paper["id"]
                paper_dir.mkdir(exist_ok=True)

                # Save metadata
                metadata_path = paper_dir / "metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(paper, f, indent=2, ensure_ascii=False)

                # Download code if available
                if paper.get("github_url"):
                    code_dir = paper_dir / "code"
                    git.Repo.clone_from(paper["github_url"], code_dir)

                downloaded_paths.append(str(paper_dir))

            return downloaded_paths

        except Exception as e:
            logger.error(f"Error downloading from Papers with Code: {e}")
            raise

    def download_openalex_papers(self, query: str) -> List[str]:
        """
        Download papers from OpenAlex.

        Args:
            query: Search query

        Returns:
            List of paths to downloaded papers
        """
        try:
            # Search papers
            url = f"{self.OPENALEX_BASE_URL}works"
            params = {"filter": query}
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()

            downloaded_paths = []
            for paper in results["results"]:
                # Create directory for the paper
                paper_dir = self.openalex_dir / paper["id"].split("/")[-1]
                paper_dir.mkdir(exist_ok=True)

                # Save metadata
                metadata_path = paper_dir / "metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(paper, f, indent=2, ensure_ascii=False)

                downloaded_paths.append(str(paper_dir))

            return downloaded_paths

        except Exception as e:
            logger.error(f"Error downloading from OpenAlex: {e}")
            raise

    def download_connected_papers(self, seed_paper_id: str) -> List[str]:
        """
        Download related papers from Connected Papers.

        Args:
            seed_paper_id: ID of the seed paper

        Returns:
            List of paths to downloaded papers
        """
        try:
            # Get related papers
            url = f"{self.CONNECTED_PAPERS_BASE_URL}graph"
            params = {"seed": seed_paper_id}
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()

            downloaded_paths = []
            for paper in results["papers"]:
                # Create directory for the paper
                paper_dir = self.connected_papers_dir / paper["id"]
                paper_dir.mkdir(exist_ok=True)

                # Save metadata
                metadata_path = paper_dir / "metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(paper, f, indent=2, ensure_ascii=False)

                downloaded_paths.append(str(paper_dir))

            # Save relationship graph
            graph_path = self.connected_papers_dir / f"{seed_paper_id}_graph.json"
            with open(graph_path, "w", encoding="utf-8") as f:
                json.dump(results["graph"], f, indent=2, ensure_ascii=False)

            return downloaded_paths

        except Exception as e:
            logger.error(f"Error downloading from Connected Papers: {e}")
            raise
