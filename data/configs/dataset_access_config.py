"""Access configuration for premium datasets."""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum
import os

class AccessType(str, Enum):
    """Dataset access types."""
    DIRECT = "direct"              # Direct load
    API = "api"                    # Access via API
    INSTITUTIONAL = "institutional" # Requires institutional credentials
    MEDICAL = "medical"            # Requires medical credentials
    SUBSCRIPTION = "subscription"   # Requires subscription

@dataclass
class DatasetAccess:
    """Access configuration for a dataset."""
    name: str
    access_type: AccessType
    url: str
    requires_auth: bool
    api_key_env: Optional[str] = None
    preprocessing_required: bool = False
    preprocessing_steps: List[str] = None
    format: str = "json"
    rate_limits: Optional[Dict] = None

PSYCHOLOGY_DATASETS = {
    "smhd": DatasetAccess(
        name="Shared-Relevance Mental Health Diagnosis",
        access_type=AccessType.INSTITUTIONAL,
        url="https://georgetown.edu/smhd-dataset",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=[
            "anonymization",
            "text_normalization",
            "linguistic_features_extraction"
        ],
        format="json"
    ),
    "phq9": DatasetAccess(
        name="PHQ-9 Clinical Depression",
        access_type=AccessType.MEDICAL,
        url="https://nndc.org/phq9-dataset",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=[
            "clinical_standardization",
            "entity_classification",
            "temporal_alignment"
        ],
        format="csv"
    ),
    "mental_health_multimodal": DatasetAccess(
        name="Mental Health Multi-Modal",
        access_type=AccessType.DIRECT,
        url="https://huggingface.co/datasets/mental-health-multimodal",
        requires_auth=False,
        preprocessing_required=False,
        format="parquet"
    )
}

LEGAL_DATASETS = {
    "icj_pcij": DatasetAccess(
        name="ICJ-PCIJ Corpus Decisions",
        access_type=AccessType.SUBSCRIPTION,
        url="https://heinonline.org/HOL/Index",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=[
            "pdf_extraction",
            "text_normalization",
            "bilingual_alignment",
            "metadata_extraction"
        ],
        format="pdf+text"
    ),
    "wto_disputes": DatasetAccess(
        name="WTO Dispute Settlement",
        access_type=AccessType.API,
        url="https://www.worldtradeorganization.net/api/v1",
        requires_auth=True,
        api_key_env="WTO_API_KEY",
        preprocessing_required=False,
        rate_limits={
            "requests_per_second": 10,
            "requests_per_day": 10000
        },
        format="json"
    ),
    "icsid": DatasetAccess(
        name="ICSID Investment Disputes",
        access_type=AccessType.API,
        url="https://icsid.worldbank.org/api/v1",
        requires_auth=True,
        api_key_env="WORLD_BANK_API_KEY",
        preprocessing_required=True,
        preprocessing_steps=[
            "xml_structuring",
            "temporal_alignment",
            "metadata_enrichment"
        ],
        format="json"
    ),
    "itlos_cases": DatasetAccess(
        name="ITLOS + COSIS Climate",
        access_type=AccessType.INSTITUTIONAL,
        url="https://www.un.org/law/sea/data",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=[
            "xml_parsing",
            "climate_data_integration",
            "temporal_alignment"
        ],
        format="xml+json"
    )
}

def get_dataset_access_config(dataset_name: str) -> Optional[DatasetAccess]:
    """Obtain the access configuration for a dataset."""
    return {
        **PSYCHOLOGY_DATASETS,
        **LEGAL_DATASETS
    }.get(dataset_name)

def validate_access_credentials(dataset_name: str) -> bool:
    """Validate the access credentials for a dataset."""
    config = get_dataset_access_config(dataset_name)
    if not config:
        return False

    if config.access_type == AccessType.API:
        return bool(os.getenv(config.api_key_env))

    return True  # For other types, validation is done at access time
