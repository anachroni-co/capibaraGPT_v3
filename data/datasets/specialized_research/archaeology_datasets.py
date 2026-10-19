"""
Archaeology Data Service (ADS) Datasets Manager for CapibaraGPT v3

Specialized manager for archaeological datasets from the UK's national digital archive including:
- 4,852+ archaeological records and datasets
- Excavation data from prehistoric to modern periods
- Digital archives from major archaeological projects
- Bioarchaeological data and scientific analysis
- Cultural heritage and historical documentation
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import datetime
import json

logger = logging.getLogger(__name__)

class ArchaeologyDatasets:
    """Manager for Archaeology Data Service datasets."""

    def __init__(self):
        """
        Initialize the archaeology datasets manager.
        """
        self.dataset_info = {
            # Major flagship projects
            "feedsax_project": {
                "name": "Feeding Anglo-Saxon England (FeedSax)",
                "description": "Bioarchaeology of an Agricultural Revolution, 2017-2022",
                "size": "Multi-GB scientific data",
                "provider": "University of Oxford, University of Leicester",
                "doi": "https://doi.org/10.5284/1057492",
                "type": "bioarchaeology",
                "period": "8th-13th centuries",
                "funding": "European Research Council (Advanced Grant 741751)",
                "techniques": [
                    "stable_isotope_analysis",
                    "functional_weed_ecology",
                    "animal_palaeopathology",
                    "radiocarbon_dating"
                ],
                "scientific_focus": "Agricultural transformation and demographic growth",
                "data_types": ["grains", "seeds", "animal_bones", "pollen"],
                "geographic_coverage": "England"
            },

            "hambledon_hill": {
                "name": "Hambledon Hill Project",
                "description": "Neolithic monument complex excavation 1974-2008",
                "size": "Large-scale excavation archive",
                "provider": "Cardiff University, Historic England",
                "doi": "https://doi.org/10.5284/1097703",
                "type": "prehistoric_archaeology",
                "period": "Neolithic to Iron Age",
                "features": [
                    "two_neolithic_long_barrows",
                    "two_neolithic_causewayed_enclosures",
                    "iron_age_hillfort",
                    "disarticulated_human_bone_deposits"
                ],
                "methods": ["excavation", "field_survey", "air_photograph_analysis"],
                "geographic_coverage": "Dorset, England"
            },

            "roman_amphorae": {
                "name": "Roman Amphorae Digital Resource",
                "description": "Comprehensive database of Roman amphorae types and distribution",
                "size": "Multi-format database",
                "provider": "University of Southampton",
                "doi": "https://doi.org/10.5284/1028192",
                "type": "material_culture",
                "period": "Roman",
                "features": [
                    "amphorae_typology",
                    "fabric_analysis",
                    "distribution_patterns",
                    "3d_models",
                    "petrological_data"
                ],
                "geographic_coverage": "Mediterranean, Europe",
                "research_focus": "Trade networks and ceramic technology"
            },

            "scpx_azerbaijan": {
                "name": "South Caucasus Pipeline Expansion Archaeological Excavations",
                "description": "Pipeline archaeology in Azerbaijan 2013-2018",
                "size": "Multi-site excavation data",
                "provider": "Landesker Archaeology Ltd, BP Exploration",
                "doi": "https://doi.org/10.5284/1101054",
                "type": "rescue_archaeology",
                "period": "Chalcolithic to Medieval",
                "sites_excavated": 48,
                "cultures_represented": [
                    "Chalcolithic",
                    "Kura_Araxes_early_Bronze_Age",
                    "Xocali_Gedebey_late_Bronze_early_Iron",
                    "Antique_jar_graves",
                    "Medieval_settlements"
                ],
                "major_discovery": "Medieval castle at Kerpicli",
                "geographic_coverage": "Northwest Azerbaijan"
            },

            "corpus_vitrearum": {
                "name": "Corpus Vitrearum Medii Aevi Digital Archive",
                "description": "Medieval stained glass documentation and analysis",
                "size": "Comprehensive visual archive",
                "provider": "Corpus Vitrearum Medii Aevi",
                "doi": "https://doi.org/10.5284/1132566",
                "type": "art_history",
                "period": "Medieval",
                "focus": "Stained glass windows and artistic techniques",
                "data_types": ["images", "documentation", "analysis"],
                "geographic_coverage": "Europe"
            }
        }

        # Temporal periods covered
        self.temporal_periods = {
            "prehistoric": {
                "name": "Prehistoric",
                "dataset_count": 1316,
                "subperiods": [
                    "Palaeolithic", "Mesolithic", "Neolithic",
                    "Bronze Age", "Iron Age"
                ]
            },
            "roman": {
                "name": "Roman",
                "dataset_count": 1194,
                "period_range": "43-410 CE",
                "geographic_focus": "Britain and Roman Empire"
            },
            "medieval": {
                "name": "Medieval",
                "dataset_count": 1503,
                "period_range": "410-1500 CE",
                "includes": ["Anglo-Saxon", "Norman", "Later Medieval"]
            },
            "post_medieval": {
                "name": "Post Medieval",
                "dataset_count": 2280,
                "period_range": "1500-1800 CE"
            },
            "modern": {
                "name": "Modern",
                "dataset_count": 450,
                "period_range": "1800-present"
            }
        }

        # Geographic coverage
        self.geographic_coverage = {
            "british_isles": {
                "name": "British Isles",
                "dataset_count": 4661,
                "countries": ["England", "Scotland", "Wales", "Ireland"]
            },
            "continental_europe": {
                "name": "Continental Europe",
                "dataset_count": 73,
                "includes": ["France", "Germany", "Italy", "Scandinavia"]
            },
            "middle_east": {
                "name": "Middle East",
                "dataset_count": 19,
                "includes": ["Turkey", "Syria", "Jordan", "Israel/Palestine"]
            },
            "africa": {
                "name": "Africa",
                "dataset_count": 25,
                "includes": ["Egypt", "Ethiopia", "Eritrea"]
            },
            "asia": {
                "name": "Asia",
                "dataset_count": 15,
                "includes": ["Central Asia", "South Asia"]
            },
            "south_america": {
                "name": "South America",
                "dataset_count": 6
            }
        }

        # Data types and categories
        self.data_categories = {
            "event": {
                "name": "Archaeological Events",
                "count": 4113,
                "description": "Excavations, surveys, and archaeological interventions"
            },
            "evidence": {
                "name": "Archaeological Evidence",
                "count": 183,
                "description": "Artifacts, ecofacts, and material remains"
            },
            "object": {
                "name": "Archaeological Objects",
                "count": 1747,
                "description": "Portable artifacts and finds"
            },
            "maritime_craft": {
                "name": "Maritime Craft",
                "count": 20,
                "description": "Ships, boats, and marine archaeology"
            },
            "monument": {
                "name": "Monuments",
                "count": 4086,
                "description": "Buildings, structures, and landscape features"
            }
        }

        # Research methodologies
        self.methodologies = {
            "excavation": {
                "description": "Stratigraphic excavation and recording",
                "data_outputs": ["context_sheets", "plans", "sections", "photographs"]
            },
            "survey": {
                "description": "Field walking, geophysical survey, aerial photography",
                "data_outputs": ["distribution_maps", "geophysical_plots", "photographs"]
            },
            "scientific_analysis": {
                "description": "Laboratory analysis of materials",
                "techniques": [
                    "radiocarbon_dating",
                    "stable_isotope_analysis",
                    "petrological_analysis",
                    "archaeobotany",
                    "zooarchaeology",
                    "micromorphology"
                ]
            },
            "digital_documentation": {
                "description": "3D recording, photogrammetry, GIS",
                "outputs": ["3d_models", "orthophotos", "gis_data"]
            }
        }

        # File formats and data standards
        self.technical_specs = {
            "data_formats": [
                "CSV", "XML", "PDF", "TIFF", "JPEG", "DWG", "SHP", "KML"
            ],
            "metadata_standards": [
                "Dublin Core",
                "MIDAS Heritage",
                "CIDOC-CRM"
            ],
            "doi_system": "Crossref DOI for persistent identification",
            "license": "Creative Commons Attribution 4.0 International",
            "preservation_standards": "OAIS compliant digital preservation"
        }

    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get all available archaeology datasets."""
        return self.dataset_info

    def get_temporal_coverage(self) -> Dict[str, Dict[str, Any]]:
        """Get temporal period coverage statistics."""
        return self.temporal_periods

    def get_geographic_coverage(self) -> Dict[str, Dict[str, Any]]:
        """Get geographic coverage statistics."""
        return self.geographic_coverage

    def get_data_categories(self) -> Dict[str, Dict[str, Any]]:
        """Get data type categories and counts."""
        return self.data_categories

    def search_by_period(self, period: str) -> List[Dict[str, Any]]:
        """
        Search datasets by temporal period.

        Args:
            period: Temporal period to search for

        Returns:
            List of matching datasets
        """
        matches = []

        for dataset_id, info in self.dataset_info.items():
            dataset_period = info.get("period", "").lower()
            if period.lower() in dataset_period or any(
                period.lower() in p.lower() for p in info.get("features", [])
            ):
                matches.append({
                    "id": dataset_id,
                    **info
                })

        return matches

    def search_by_geographic_region(self, region: str) -> List[Dict[str, Any]]:
        """
        Search datasets by geographic region.

        Args:
            region: Geographic region to search for

        Returns:
            List of matching datasets
        """
        matches = []

        for dataset_id, info in self.dataset_info.items():
            coverage = info.get("geographic_coverage", "").lower()
            if region.lower() in coverage:
                matches.append({
                    "id": dataset_id,
                    **info
                })

        return matches

    def search_by_research_type(self, research_type: str) -> List[Dict[str, Any]]:
        """
        Search datasets by research type or methodology.

        Args:
            research_type: Type of research to search for

        Returns:
            List of matching datasets
        """
        matches = []

        for dataset_id, info in self.dataset_info.items():
            dataset_type = info.get("type", "").lower()
            techniques = [t.lower() for t in info.get("techniques", [])]
            methods = [m.lower() for m in info.get("methods", [])]

            if (research_type.lower() in dataset_type or
                any(research_type.lower() in t for t in techniques) or
                any(research_type.lower() in m for m in methods)):
                matches.append({
                    "id": dataset_id,
                    **info
                })

        return matches

    def get_bioarchaeology_datasets(self) -> List[Dict[str, Any]]:
        """Get datasets related to bioarchaeology and scientific analysis."""
        bioarch_datasets = []

        for dataset_id, info in self.dataset_info.items():
            if (info.get("type") == "bioarchaeology" or
                "bioarch" in info.get("description", "").lower() or
                any("isotope" in t or "bone" in t or "pollen" in t
                    for t in info.get("techniques", []))):
                bioarch_datasets.append({
                    "id": dataset_id,
                    **info
                })

        return bioarch_datasets

    def get_digital_heritage_datasets(self) -> List[Dict[str, Any]]:
        """Get datasets focused on digital heritage and documentation."""
        digital_datasets = []

        for dataset_id, info in self.dataset_info.items():
            if ("digital" in info.get("name", "").lower() or
                "3d" in str(info.get("features", [])).lower() or
                info.get("type") == "art_history"):
                digital_datasets.append({
                    "id": dataset_id,
                    **info
                })

        return digital_datasets

    def get_rescue_archaeology_datasets(self) -> List[Dict[str, Any]]:
        """Get datasets from rescue/commercial archaeology projects."""
        rescue_datasets = []

        for dataset_id, info in self.dataset_info.items():
            if (info.get("type") == "rescue_archaeology" or
                "pipeline" in info.get("description", "").lower() or
                "commercial" in info.get("description", "").lower()):
                rescue_datasets.append({
                    "id": dataset_id,
                    **info
                })

        return rescue_datasets

    def get_methodological_approaches(self) -> Dict[str, Any]:
        """Get information about archaeological methodologies."""
        return self.methodologies

    def get_technical_specifications(self) -> Dict[str, Any]:
        """Get technical specifications and data standards."""
        return self.technical_specs

    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the ADS collection."""
        total_datasets = sum(period["dataset_count"] for period in self.temporal_periods.values())
        total_geographic = sum(region["dataset_count"] for region in self.geographic_coverage.values())
        total_categories = sum(cat["count"] for cat in self.data_categories.values())

        return {
            "total_records": 4852,
            "total_by_period": total_datasets,
            "total_by_geography": total_geographic,
            "total_by_category": total_categories,
            "temporal_span": "Prehistoric to Modern (500,000+ years)",
            "geographic_span": "Global coverage with UK focus",
            "update_frequency": "Daily additions",
            "data_preservation": "OAIS compliant long-term preservation",
            "access_model": "Open access with CC BY 4.0 license",
            "major_funders": [
                "Arts and Humanities Research Council (AHRC)",
                "European Research Council (ERC)",
                "Historic England",
                "British Academy"
            ]
        }

    def generate_search_examples(self) -> Dict[str, Any]:
        """Generate example searches and use cases."""
        return {
            "period_search": {
                "example": "search_by_period('Medieval')",
                "description": "Find all Medieval archaeological datasets",
                "expected_results": "Datasets from 410-1500 CE period"
            },

            "geographic_search": {
                "example": "search_by_geographic_region('Scotland')",
                "description": "Find datasets from Scotland",
                "expected_results": "Archaeological projects in Scottish sites"
            },

            "methodology_search": {
                "example": "search_by_research_type('isotope')",
                "description": "Find datasets using isotope analysis",
                "expected_results": "Bioarchaeological projects with scientific analysis"
            },

            "bioarchaeology_focus": {
                "example": "get_bioarchaeology_datasets()",
                "description": "Get all bioarchaeological datasets",
                "expected_results": "Scientific analysis of archaeological materials"
            },

            "digital_heritage": {
                "example": "get_digital_heritage_datasets()",
                "description": "Find digital documentation projects",
                "expected_results": "3D recording, photogrammetry, digital archives"
            }
        }

    def get_research_impact(self) -> Dict[str, Any]:
        """Get information about research impact and applications."""
        return {
            "academic_impact": {
                "journal_publications": "1000+ peer-reviewed papers",
                "monographs": "100+ archaeological monographs",
                "phd_theses": "500+ doctoral dissertations",
                "citation_network": "Highly cited archaeological literature"
            },

            "policy_impact": {
                "heritage_management": "Informs UK heritage policy",
                "planning_guidance": "Archaeological advice for development",
                "conservation_strategies": "Monument preservation planning",
                "education_resources": "Teaching materials for universities"
            },

            "technological_innovation": {
                "digital_preservation": "Pioneering digital archaeology methods",
                "data_standards": "MIDAS Heritage metadata standard",
                "3d_recording": "Advanced documentation techniques",
                "database_design": "Archaeological information systems"
            },

            "international_collaboration": {
                "european_projects": "Collaboration with European archaeologists",
                "global_partnerships": "International archaeological missions",
                "data_sharing": "Cross-border archaeological data exchange",
                "capacity_building": "Training international archaeologists"
            }
        }

# Factory function
def get_archaeology_datasets() -> ArchaeologyDatasets:
    """Get Archaeology Data Service datasets manager."""
    return ArchaeologyDatasets()
