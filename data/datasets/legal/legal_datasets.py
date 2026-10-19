"""International law datasets configuration."""

from ..dataset_access_info import DataAccess, AccessType

LEGAL_DATASETS = {
    "ICJ_PCIJ": DataAccess(
        name="ICJ-PCIJ Corpus Decisions",
        access_type=AccessType.LEGAL,
        url="https://www.icj-cij.org/api/datasets/decisions",
        requires_auth=True,
        file_format="json",
        preprocessing_required=True,
        preprocessing_steps=[
            "text_extraction",
            "language_detection",
            "opinion_classification",
            "metadata_enrichment"
        ],
        metadata={
            "period": "1922-2021",
            "languages": ["English", "French"],
            "content_types": [
                "Majority opinions",
                "Minority opinions",
                "Declarations",
                "Separate opinions",
                "Dissenting opinions"
            ],
            "authority": "UN + League of Nations"
        }
    ),

    "WTO_DISPUTES": DataAccess(
        name="WTO Dispute Settlement Database",
        access_type=AccessType.INSTITUTIONAL,
        url="https://www.worldbank.org/api/wto-disputes",
        requires_auth=True,
        file_format="parquet",
        preprocessing_required=True,
        preprocessing_steps=[
            "dispute_classification",
            "variable_extraction",
            "document_linking",
            "timeline_reconstruction"
        ],
        metadata={
            "disputes": "351",
            "entries": "~28,000",
            "documents": "3,000+",
            "coverage": "1995-2006+",
            "authority": "World Bank + WTO"
        }
    ),

    "ICSID": DataAccess(
        name="ICSID Investment Disputes Database",
        access_type=AccessType.LEGAL,
        url="https://icsid.worldbank.org/api/cases",
        requires_auth=True,
        file_format="json",
        preprocessing_required=True,
        preprocessing_steps=[
            "xml_extraction",
            "award_classification",
            "arbitrator_analysis",
            "outcome_labeling"
        ],
        metadata={
            "period": "1972-present",
            "coverage": [
                "ICSID Convention",
                "Additional Facility",
                "UNCITRAL rules"
            ],
            "content": [
                "Awards",
                "Annulment",
                "Follow-on proceedings"
            ],
            "authority": "World Bank ICSID"
        }
    ),

    "ITLOS_COSIS": DataAccess(
        name="ITLOS Law of the Sea + COSIS Climate",
        access_type=AccessType.LEGAL,
        url="https://www.itlos.org/api/decisions",
        requires_auth=True,
        file_format="json",
        preprocessing_required=True,
        preprocessing_steps=[
            "decision_extraction",
            "climate_annotation",
            "judge_analysis",
            "jurisdiction_classification"
        ],
        metadata={
            "period": "1996-2024",
            "judges": "21 independent",
            "content_types": [
                "Vessel release",
                "Provisional measures",
                "Advisory opinions",
                "Climate obligations"
            ],
            "authority": "UN Convention Law of Sea"
        }
    )
}
