"""
Module for handling datasets from academic and governmental institutions.
"""

import os
import json
import logging
import requests
from urllib.parse import urlparse
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Allowed hosts for institutional dataset downloads (SSRF protection)
_ALLOWED_INSTITUTIONAL_HOSTS = frozenset({
    "archive.ics.uci.edu",
    "data.nasa.gov",
    "catalog.data.gov",
    "api.worldbank.org",
    "data.un.org",
    "data.uis.unesco.org",
    "raw.githubusercontent.com",
})


def _validated_get(url: str, **kwargs) -> requests.Response:
    """HTTP GET with URL validation to prevent SSRF."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Disallowed URL scheme: {parsed.scheme}")
    if parsed.hostname and parsed.hostname not in _ALLOWED_INSTITUTIONAL_HOSTS:
        raise ValueError(
            f"Disallowed host: {parsed.hostname}. "
            f"Allowed: {_ALLOWED_INSTITUTIONAL_HOSTS}"
        )
    return requests.get(url, timeout=60, **kwargs)

@dataclass
class DatasetMetadata:
    """Metadata for an institutional dataset."""
    id: str
    name: str
    description: str
    source: str
    format: str
    size: Optional[int] = None
    url: Optional[str] = None
    license: Optional[str] = None
    language: Optional[str] = None
    tags: List[str] = None
    features: Optional[int] = None
    instances: Optional[int] = None

class InstitutionalDatasetManager:
    """Manager for institutional datasets."""

    # Base URLs for APIs
    UCI_BASE_URL = "https://archive.ics.uci.edu/api/datasets/"
    NASA_BASE_URL = "https://data.nasa.gov/api/views/"
    DATAGOV_BASE_URL = "https://catalog.data.gov/api/3/action/package_search"
    WORLDBANK_BASE_URL = "https://api.worldbank.org/v2/"
    UNESCO_BASE_URL = "https://api.uis.unesco.org/sdmx/data/"
    UN_BASE_URL = "https://www.un-ilibrary.org/api/"
    FIVETHIRTYEIGHT_BASE_URL = "https://data.fivethirtyeight.com/"

    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the institutional datasets manager.

        Args:
            base_dir: Base directory for storing datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each source
        self.uci_dir = self.base_dir / "uci"
        self.nasa_dir = self.base_dir / "nasa"
        self.datagov_dir = self.base_dir / "datagov"
        self.worldbank_dir = self.base_dir / "worldbank"
        self.unesco_dir = self.base_dir / "unesco"
        self.un_dir = self.base_dir / "un"
        self.fivethirtyeight_dir = self.base_dir / "538"
        self.huggingface_dir = self.base_dir / "huggingface"

        for dir_path in [self.uci_dir, self.nasa_dir, self.datagov_dir,
                        self.worldbank_dir, self.unesco_dir, self.un_dir,
                        self.fivethirtyeight_dir, self.huggingface_dir]:
            dir_path.mkdir(exist_ok=True)

    def download_uci_dataset(self, dataset_id: str) -> str:
        """
        Download a dataset from UCI Machine Learning Repository.

        Args:
            dataset_id: Identifier of the dataset

        Returns:
            Path to downloaded file
        """
        try:
            # Obtain metadata of the dataset
            metadata_url = f"{self.UCI_BASE_URL}{dataset_id}"
            response = _validated_get(metadata_url)
            response.raise_for_status()
            metadata = response.json()

            # Create directory for the dataset
            dataset_dir = self.uci_dir / dataset_id
            dataset_dir.mkdir(exist_ok=True)

            # Download files of the dataset
            for file_info in metadata["files"]:
                file_url = file_info["url"]
                file_name = file_info["name"]
                file_path = dataset_dir / file_name

                if not file_path.exists():
                    logger.info(f"Downloading {file_name} from UCI...")
                    response = _validated_get(file_url, stream=True)
                    response.raise_for_status()

                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

            return str(dataset_dir)

        except Exception as e:
            logger.error(f"Error downloading UCI dataset {dataset_id}: {e}")
            raise

    def download_nasa_dataset(self, dataset_id: str) -> str:
        """
        Download a dataset from NASA Open Data.

        Args:
            dataset_id: Identifier of the dataset

        Returns:
            Path to downloaded file
        """
        try:
            # Obtain metadata of the dataset
            metadata_url = f"{self.NASA_BASE_URL}{dataset_id}"
            response = _validated_get(metadata_url)
            response.raise_for_status()
            metadata = response.json()

            # Create directory for the dataset
            dataset_dir = self.nasa_dir / dataset_id
            dataset_dir.mkdir(exist_ok=True)

            # Download files of the dataset
            for file_info in metadata["files"]:
                file_url = file_info["downloadUrl"]
                file_name = file_info["name"]
                file_path = dataset_dir / file_name

                if not file_path.exists():
                    logger.info(f"Downloading {file_name} from NASA...")
                    response = _validated_get(file_url, stream=True)
                    response.raise_for_status()

                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

            return str(dataset_dir)

        except Exception as e:
            logger.error(f"Error downloading NASA dataset {dataset_id}: {e}")
            raise

    def download_datagov_dataset(self, dataset_id: str) -> str:
        """
        Download a dataset from Data.gov.

        Args:
            dataset_id: Identifier of the dataset

        Returns:
            Path to downloaded file
        """
        try:
            # Obtain metadata of the dataset
            params = {"q": dataset_id}
            response = _validated_get(self.DATAGOV_BASE_URL, params=params)
            response.raise_for_status()
            metadata = response.json()["result"]["results"][0]

            # Create directory for the dataset
            dataset_dir = self.datagov_dir / dataset_id
            dataset_dir.mkdir(exist_ok=True)

            # Download resources of the dataset
            for resource in metadata["resources"]:
                file_url = resource["url"]
                file_name = resource["name"]
                file_path = dataset_dir / file_name

                if not file_path.exists():
                    logger.info(f"Downloading {file_name} from Data.gov...")
                    response = _validated_get(file_url, stream=True)
                    response.raise_for_status()

                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

            return str(dataset_dir)

        except Exception as e:
            logger.error(f"Error downloading Data.gov dataset {dataset_id}: {e}")
            raise

    def download_worldbank_dataset(self, indicator: str, country: str = "all",
                                 start_year: int = None, end_year: int = None) -> str:
        """
        Download data from World Bank.

        Args:
            indicator: Code of the indicator
            country: Country code or "all"
            start_year: Start year
            end_year: End year

        Returns:
            Path to downloaded file
        """
        try:
            # Build URL for the API
            url = f"{self.WORLDBANK_BASE_URL}countries/{country}/indicators/{indicator}"
            params = {
                "format": "json",
                "per_page": 1000
            }
            if start_year:
                params["date"] = f"{start_year}:{end_year or 'latest'}"

            # Perform request
            response = _validated_get(url, params=params)
            response.raise_for_status()
            data = response.json()[1]  # First element is metadata

            # Save data
            file_name = f"{indicator}_{country}_{start_year}-{end_year}.json"
            file_path = self.worldbank_dir / file_name

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return str(file_path)

        except Exception as e:
            logger.error(f"Error downloading World Bank data: {e}")
            raise

    def download_unesco_dataset(self, dataset_id: str) -> str:
        """
        Download a dataset from UNESCO.

        Args:
            dataset_id: Identifier of the dataset

        Returns:
            Path to downloaded file
        """
        try:
            # Build URL for the API
            url = f"{self.UNESCO_BASE_URL}{dataset_id}"
            params = {"format": "json"}

            # Perform request
            response = _validated_get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Save data
            file_path = self.unesco_dir / f"{dataset_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return str(file_path)

        except Exception as e:
            logger.error(f"Error downloading UNESCO dataset {dataset_id}: {e}")
            raise

    def download_un_dataset(self, dataset_id: str) -> str:
        """
        Download a dataset from UN iLibrary.

        Args:
            dataset_id: Identifier of the dataset

        Returns:
            Path to downloaded file
        """
        try:
            # Build URL for the API
            url = f"{self.UN_BASE_URL}datasets/{dataset_id}"
            params = {"format": "json"}

            # Perform request
            response = _validated_get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Save data
            file_path = self.un_dir / f"{dataset_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return str(file_path)

        except Exception as e:
            logger.error(f"Error downloading UN dataset {dataset_id}: {e}")
            raise

    def download_538_dataset(self, dataset_id: str) -> str:
        """
        Download a dataset from FiveThirtyEight.

        Args:
            dataset_id: Identifier of the dataset

        Returns:
            Path to downloaded file
        """
        try:
            # Build URL for the dataset
            url = f"{self.FIVETHIRTYEIGHT_BASE_URL}data/{dataset_id}/MANIFEST.json"
            response = _validated_get(url)
            response.raise_for_status()
            manifest = response.json()

            # Create directory for the dataset
            dataset_dir = self.fivethirtyeight_dir / dataset_id
            dataset_dir.mkdir(exist_ok=True)

            # Download files of the dataset
            for file_info in manifest["files"]:
                file_url = file_info["url"]
                file_name = file_info["name"]
                file_path = dataset_dir / file_name

                if not file_path.exists():
                    logger.info(f"Downloading {file_name} from FiveThirtyEight...")
                    response = _validated_get(file_url, stream=True)
                    response.raise_for_status()

                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                # Download README if exists
                readme_url = f"{self.FIVETHIRTYEIGHT_BASE_URL}data/{dataset_id}/README.md"
                try:
                    response = _validated_get(readme_url)
                    response.raise_for_status()
                    with open(dataset_dir / "README.md", "w", encoding="utf-8") as f:
                        f.write(response.text)
                except Exception:
                    logger.warning(f"No README found for {dataset_id}")

            return str(dataset_dir)

        except Exception as e:
            logger.error(f"Error downloading FiveThirtyEight dataset {dataset_id}: {e}")
            raise

    def download_huggingface_dataset(self, dataset_name: str, subset: Optional[str] = None,
                                   split: Optional[str] = None, cache_dir: Optional[str] = None) -> str:
        """
        Download a dataset from Hugging Face.

        Args:
            dataset_name: Name of the dataset on Hugging Face
            subset: Name of the subset (optional)
            split: Specific split to download (optional)
            cache_dir: Cache directory (optional)

        Returns:
            Path to the dataset directory
        """
        try:
            # Configure cache directory
            if cache_dir is None:
                cache_dir = str(self.huggingface_dir / dataset_name.replace("/", "_"))

            # Load dataset
            logger.info(f"Downloading dataset {dataset_name} from Hugging Face...")
            from datasets import load_dataset
            dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
                cache_dir=cache_dir
            )

            # Save metadata
            metadata = {
                "name": dataset_name,
                "subset": subset,
                "split": split,
                "features": list(dataset.features.keys()),
                "num_rows": len(dataset),
                "description": dataset.description,
                "citation": dataset.citation,
                "homepage": dataset.homepage
            }

            metadata_path = Path(cache_dir) / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            return cache_dir

        except Exception as e:
            logger.error(f"Error downloading Hugging Face dataset {dataset_name}: {e}")
            raise


def main():
    """Main function for module execution."""
    logger.info("Module institutional_datasets.py starting")
    return True

if __name__ == "__main__":
    main()
