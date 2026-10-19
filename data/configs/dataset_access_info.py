"""Access configuration for premium datasets."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

class AccessType(str, Enum):
    """Dataset access types."""
    DIRECT = "direct"              # Direct load without authentication
    API = "api"                    # Access via API with key
    INSTITUTIONAL = "institutional" # Requires academic credentials
    MEDICAL = "medical"            # Requires medical credentials
    LEGAL = "legal"               # Requires legal credentials
    SUBSCRIPTION = "subscription"   # Requires paid subscription

@dataclass
class DatasetAccess:
    """Access information for a dataset."""
    name: str
    access_type: AccessType
    url: str
    requires_auth: bool
    category: str = "unknown"
    api_key_env: Optional[str] = None
    preprocessing_required: bool = False
    preprocessing_steps: List[str] = None
    file_format: str = "mixed"
    size_gb: Optional[float] = None
    update_frequency: str = "static"
    rate_limits: Optional[Dict] = None
    metadata: Optional[Dict[str, Any]] = None

# Psychology Datasets
PSYCHOLOGY_DATASETS = {
    "SMHD": DatasetAccess(
        name="Shared-Relevance Mental Health Diagnosis",
        category="psychology",
        access_type=AccessType.INSTITUTIONAL,
        url="https://georgetown.edu/smhd-dataset",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=["anonymization", "text_normalization", "condition_labeling"],
        file_format="json",
        size_gb=2.5,
        update_frequency="yearly"
    ),
    "PHQ9_Clinical": DatasetAccess(
        name="PHQ-9 Clinical Depression",
        category="psychology",
        access_type=AccessType.MEDICAL,
        url="https://nndc.org/phq9-dataset",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=["patient_deidentification", "entity_scoring", "validation"],
        file_format="csv",
        size_gb=1.8,
        update_frequency="quarterly"
    ),
    "Mental_Health_Multimodal": DatasetAccess(
        name="Mental Health Multi-Modal Research",
        category="psychology",
        access_type=AccessType.DIRECT,
        url="https://huggingface.co/datasets/mental-health-research",
        requires_auth=False,
        preprocessing_required=True,
        preprocessing_steps=["feature_extraction", "scale_normalization"],
        file_format="parquet",
        size_gb=3.2,
        update_frequency="monthly"
    )
}

# International Law Datasets
LEGAL_DATASETS = {
    "ICJ_PCIJ": DatasetAccess(
        name="ICJ-PCIJ Corpus Decisions",
        category="legal",
        access_type=AccessType.LEGAL,
        url="https://www.icj-cij.org/advanced-search",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=["text_extraction", "language_detection", "metadata_enrichment"],
        file_format="pdf+xml",
        size_gb=15.0,
        update_frequency="weekly",
        rate_limits={"requests_per_hour": 1000}
    ),
    "WTO_Disputes": DatasetAccess(
        name="WTO Dispute Settlement Database",
        category="legal",
        access_type=AccessType.API,
        url="https://api.worldbank.org/wto-disputes",
        requires_auth=True,
        api_key_env="WTO_API_KEY",
        preprocessing_required=True,
        preprocessing_steps=["json_normalization", "dispute_classification"],
        file_format="json",
        size_gb=8.5,
        update_frequency="daily",
        rate_limits={"requests_per_minute": 60}
    ),
    "ICSID_Investment": DatasetAccess(
        name="ICSID Investment Disputes",
        category="legal",
        access_type=AccessType.SUBSCRIPTION,
        url="https://icsid.worldbank.org/cases/database",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=["xml_extraction", "award_classification"],
        file_format="xml",
        size_gb=12.0,
        update_frequency="daily"
    ),
    "ITLOS_COSIS": DatasetAccess(
        name="ITLOS Law of the Sea + COSIS Climate",
        category="legal",
        access_type=AccessType.DIRECT,
        url="https://www.itlos.org/decisions",
        requires_auth=False,
        preprocessing_required=True,
        preprocessing_steps=["opinion_extraction", "climate_tagging"],
        file_format="pdf",
        size_gb=5.5,
        update_frequency="monthly"
    )
}

# Theoretical Physics Datasets
PHYSICS_DATASETS = {
    "ArXiv_Physics": DatasetAccess(
        name="ArXiv Physics Corpus",
        category="physics",
        access_type=AccessType.API,
        url="https://arxiv.org/hep/api",
        requires_auth=False,
        preprocessing_required=True,
        preprocessing_steps=["pdf_extraction", "latex_parsing", "metadata_enrichment"],
        file_format="pdf+tex",
        size_gb=250.0,
        update_frequency="daily",
        rate_limits={"requests_per_second": 1}
    ),
    "CERN_OpenData": DatasetAccess(
        name="CERN Open Data",
        category="physics",
        access_type=AccessType.DIRECT,
        url="http://opendata.cern.ch",
        requires_auth=False,
        preprocessing_required=True,
        preprocessing_steps=["event_reconstruction", "particle_identification"],
        file_format="root",
        size_gb=1000.0,
        update_frequency="yearly"
    ),
    "OpenReACT": DatasetAccess(
        name="OpenReACT-CHON-EFH",
        category="physics",
        access_type=AccessType.INSTITUTIONAL,
        url="https://quantum-chemistry-datasets.org/react",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=["structure_optimization", "hessian_calculation"],
        file_format="hdf5",
        size_gb=85.0,
        update_frequency="static"
    )
}

# Linux Datasets
LINUX_DATASETS = {
    "LKML_Archive": DatasetAccess(
        name="Linux Kernel Mailing List Archive",
        category="linux",
        access_type=AccessType.DIRECT,
        url="https://lkml.org/archive",
        requires_auth=False,
        preprocessing_required=True,
        preprocessing_steps=["email_parsing", "thread_reconstruction", "code_extraction"],
        file_format="mbox",
        size_gb=45.0,
        update_frequency="hourly"
    ),
    "LDP_Collection": DatasetAccess(
        name="Linux Documentation Project",
        category="linux",
        access_type=AccessType.DIRECT,
        url="https://tldp.org/docs.html",
        requires_auth=False,
        preprocessing_required=True,
        preprocessing_steps=["format_conversion", "section_extraction"],
        file_format="mixed",
        size_gb=15.0,
        update_frequency="weekly"
    )
}

def get_dataset_access_info(dataset_name: str) -> Optional[DatasetAccess]:
    """Obtain the access information for a specific dataset."""
    all_datasets = {
        **PSYCHOLOGY_DATASETS,
        **LEGAL_DATASETS,
        **PHYSICS_DATASETS,
        **LINUX_DATASETS
    }
    return all_datasets.get(dataset_name)

def get_datasets_by_category(category: str) -> List[DatasetAccess]:
    """Obtain all the datasets of a specific category."""
    all_datasets = {
        **PSYCHOLOGY_DATASETS,
        **LEGAL_DATASETS,
        **PHYSICS_DATASETS,
        **LINUX_DATASETS
    }
    return [ds for ds in all_datasets.values() if ds.category == category]

def get_preprocessing_pipeline(dataset_name: str) -> List[str]:
    """Obtain the preprocessing steps for a specific dataset."""
    dataset = get_dataset_access_info(dataset_name)
    if dataset and dataset.preprocessing_required:
        return dataset.preprocessing_steps
    return []
