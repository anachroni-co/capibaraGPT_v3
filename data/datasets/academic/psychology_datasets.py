"""Psychology datasets configuration."""

from ..dataset_access_info import DataAccess, AccessType

PSYCHOLOGY_DATASETS = {
    "SMHD": DataAccess(
        name="Shared-Relevance Mental Health Diagnosis",
        access_type=AccessType.INSTITUTIONAL,
        url="https://georgetown.edu/datasets/smhd",
        requires_auth=True,
        file_format="jsonl",
        preprocessing_required=True,
        preprocessing_steps=[
            "anonymization",
            "text_cleaning",
            "temporal_sorting",
            "condition_labeling"
        ],
        metadata={
            "conditions": [
                "ADHD", "Anxiety", "Autism", "Bipolar",
                "Depression", "Eating Disorder", "OCD",
                "PTSD", "Schizophrenia"
            ],
            "source": "Reddit posts",
            "validation": "High-precision patterns",
            "authority": "Georgetown University + UW"
        }
    ),

    "PHQ9": DataAccess(
        name="PHQ-9 Clinical Depression Ecosystem",
        access_type=AccessType.MEDICAL,
        url="https://nndc.org/datasets/phq9",
        requires_auth=True,
        file_format="csv",
        preprocessing_required=True,
        preprocessing_steps=[
            "entity_scoring",
            "patient_anonymization",
            "clinical_validation",
            "temporal_alignment"
        ],
        metadata={
            "patients": "37,000+",
            "instruments": ["PHQ-9", "PHQ-2"],
            "entity_scale": "0-27",
            "classifications": [
                "Minimal", "Mild", "Moderate",
                "Moderately Severe", "Severe"
            ],
            "authority": "National Network Depression Centers"
        }
    ),

    "MHMRC": DataAccess(
        name="Mental Health Multi-Modal Research Collection",
        access_type=AccessType.API,
        url="https://huggingface.co/datasets/mental-health-multimodal",
        requires_auth=False,
        api_key_env="HF_API_KEY",
        file_format="parquet",
        preprocessing_required=True,
        preprocessing_steps=[
            "demographic_encoding",
            "behavioral_metrics_extraction",
            "clinical_instrument_scoring",
            "ml_optimization"
        ],
        metadata={
            "period": "2020-2021",
            "location": "Mexico City",
            "variables": [
                "stress", "anxiety", "PTSD",
                "demographics", "social media usage"
            ],
            "instruments": [
                "GAD-7", "C-SSRS",
                "Multiple validated scales"
            ],
            "authority": "Academic medical centers"
        }
    )
}
