"""
Module for handling Wikipedia and related resources datasets.
"""

import os
import logging
import json
import requests
from urllib.parse import urlparse
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Allowed hosts for dataset downloads (SSRF protection)
_ALLOWED_DOWNLOAD_HOSTS = frozenset({
    "dumps.wikimedia.org",
    "wikipedia2vec.s3.amazonaws.com",
    "commons.wikimedia.org",
})


def _validated_get(url: str, **kwargs) -> requests.Response:
    """HTTP GET with URL validation to prevent SSRF."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Disallowed URL scheme: {parsed.scheme}")
    if parsed.hostname and parsed.hostname not in _ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError(
            f"Disallowed host: {parsed.hostname}. "
            f"Allowed: {_ALLOWED_DOWNLOAD_HOSTS}"
        )
    response = requests.get(url, timeout=60, **kwargs)
    response.raise_for_status()
    return response

class WikiDataManager:
    """Manager for Wikipedia and related resources datasets."""

    DUMPS_BASE_URL = "https://dumps.wikimedia.org/"
    WIKIPEDIA2VEC_BASE_URL = "https://wikipedia2vec.s3.amazonaws.com/model/"
    DBPEDIA_BASE_URL = "https://downloads.dbpedia.org/current/"
    WIKIDATA_BASE_URL = "https://dumps.wikimedia.org/wikidatawiki/entities/"

    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the Wikipedia datasets manager.

        Args:
            base_dir: Base directory for storing the datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each dataset type
        self.dumps_dir = self.base_dir / "dumps"
        self.embeddings_dir = self.base_dir / "wikipedia2vec"
        self.dbpedia_dir = self.base_dir / "dbpedia"
        self.wikidata_dir = self.base_dir / "wikidata"

        for dir_path in [self.dumps_dir, self.embeddings_dir,
                        self.dbpedia_dir, self.wikidata_dir]:
            dir_path.mkdir(exist_ok=True)

    def download_wiki_dump(self, language: str = "en", date: Optional[str] = None) -> str:
        """
        Download a Wikipedia dump for a specific language.

        Args:
            language: Language code (e.g., "en", "es")
            date: Specific dump date (YYYYMMDD). If None, uses the latest.

        Returns:
            Path to the downloaded file
        """
        dump_url = f"{self.DUMPS_BASE_URL}/{language}wiki/latest/"
        dump_path = self.dumps_dir / f"{language}wiki-latest-pages-articles.xml.bz2"

        try:
            if not dump_path.exists():
                logger.info(f"Downloading Wikipedia dump for {language}...")
                response = _validated_get(dump_url, stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 KB

                with open(dump_path, 'wb') as f, tqdm(
                    desc=f"Downloading {language}wiki",
                    total=total_size,
                    unit='iB',
                    unit_scale=True
                ) as pbar:
                    for data in response.iter_content(block_size):
                        f.write(data)
                        pbar.update(len(data))

            return str(dump_path)

        except Exception as e:
            logger.error(f"Error downloading Wikipedia dump: {e}")
            raise

    def download_wikipedia2vec(self, language: str = "en", dim: int = 300) -> str:
        """
        Download pretrained Wikipedia2Vec embeddings.

        Args:
            language: Language code
            dim: Embedding dimensionality (100, 300, or 500)

        Returns:
            Path to the downloaded file
        """
        model_name = f"{language}wiki_20180420_{dim}d.txt.bz2"
        model_url = f"{self.WIKIPEDIA2VEC_BASE_URL}/{model_name}"
        model_path = self.embeddings_dir / model_name

        try:
            if not model_path.exists():
                logger.info(f"Downloading Wikipedia2Vec embeddings for {language}...")
                response = _validated_get(model_url, stream=True)
                response.raise_for_status()

                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            return str(model_path)

        except Exception as e:
            logger.error(f"Error downloading Wikipedia2Vec embeddings: {e}")
            raise

    def download_dbpedia(self, language: str = "en", datasets: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Download DBpedia datasets.

        Args:
            language: Language code
            datasets: List of specific datasets to download
                     (e.g., ["infobox-properties", "page-links"])

        Returns:
            Dictionary with names and paths of downloaded datasets
        """
        if datasets is None:
            datasets = ["infobox-properties", "page-links", "labels"]

        downloaded = {}

        try:
            for dataset in datasets:
                file_name = f"{dataset}_{language}.ttl.bz2"
                file_url = f"{self.DBPEDIA_BASE_URL}/core-i18n/{language}/{file_name}"
                file_path = self.dbpedia_dir / file_name

                if not file_path.exists():
                    logger.info(f"Downloading DBpedia dataset {dataset} for {language}...")
                    response = _validated_get(file_url, stream=True)
                    response.raise_for_status()

                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                downloaded[dataset] = str(file_path)

            return downloaded

        except Exception as e:
            logger.error(f"Error downloading DBpedia datasets: {e}")
            raise

    def download_wikidata(self, entity_type: str = "all") -> str:
        """
        Download Wikidata dumps.

        Args:
            entity_type: Type of entities to download ("all", "items", or "properties")

        Returns:
            Path to the downloaded file
        """
        file_name = f"wikidata-{entity_type}-latest.json.bz2"
        file_url = f"{self.WIKIDATA_BASE_URL}/{file_name}"
        file_path = self.wikidata_dir / file_name

        try:
            if not file_path.exists():
                logger.info(f"Downloading Wikidata dump ({entity_type})...")
                response = _validated_get(file_url, stream=True)
                response.raise_for_status()

                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            return str(file_path)

        except Exception as e:
            logger.error(f"Error downloading Wikidata dump: {e}")
            raise
