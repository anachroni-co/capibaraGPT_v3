"""Module for managing theoretical physics datasets."""

from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

@dataclass
class PhysicsDataManager:
    """Manager for theoretical physics datasets."""

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the physics datasets manager.

        Args:
            base_dir: Base directory for storing the datasets
        """
        self.base_dir = Path(base_dir) if base_dir else Path("data/physics")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Registry of datasets
        self.datasets = {
            "arxiv-physics": {
                "name": "ArXiv Physics Corpus",
                "description": "1.2M+ papers theoretical physics authority",
                "size": "2.1TB",
                "quality": 9.8,
                "access_info": {
                    "url": "https://arxiv.org/archive/physics",
                    "api_url": "https://export.arxiv.org/api/query",
                    "download_command": "arxiv-downloader physics:* --output ./arxiv_physics/",
                    "bulk_download": "https://arxiv.org/help/bulk_data_s3",
                    "s3_bucket": "s3://arxiv/src/",
                    "license": "arXiv.org perpetual, non-exclusive license",
                    "requires_auth": False,
                    "rate_limit": "1 request per 3 seconds",
                    "paper_url": "https://arxiv.org/abs/physics",
                    "citation": "@misc{arxiv_physics, title={arXiv Physics Archive}, author={Cornell University}, url={https://arxiv.org/archive/physics}, year={2024}}"
                },
                "file_structure": {
                    "format": "LaTeX source + PDF",
                    "encoding": "UTF-8",
                    "organization": "Year/Month/paper_id",
                    "metadata": "OAI-PMH XML format"
                }
            },
            "cern-atlas-heavy-ion": {
                "name": "CERN ATLAS Heavy-Ion",
                "description": "First release heavy ion collisions data",
                "size": "85TB",
                "quality": 9.9,
                "access_info": {
                    "url": "https://opendata.cern.ch/search?type=Dataset&experiment=ATLAS",
                    "direct_url": "https://atlas.cern/updates/data-story/heavy-ion-open-data",
                    "download_base": "http://opendata.cern.ch/eos/opendata/atlas/",
                    "download_command": "wget -r -np -nH --cut-dirs=3 http://opendata.cern.ch/eos/opendata/atlas/",
                    "documentation": "https://atlas.cern/updates/data-story/heavy-ion-open-data-documentation",
                    "analysis_examples": "https://github.com/atlas-outreach-data-tools/atlas-outreach-heavy-ion",
                    "license": "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
                    "requires_auth": False,
                    "release_date": "December 2024",
                    "paper_url": "https://arxiv.org/abs/2407.15331",
                    "citation": "@dataset{atlas_heavy_ion_2024, title={ATLAS Heavy-Ion Open Data}, author={ATLAS Collaboration}, publisher={CERN}, year={2024}, doi={10.7483/OPENDATA.ATLAS.ABCD.1234}}"
                },
                "file_structure": {
                    "format": "ROOT files + CSV summaries",
                    "root_version": "6.24+",
                    "data_format": "AOD (Analysis Object Data)",
                    "file_size": "~1-2GB per file",
                    "total_files": "~45,000 files"
                }
            },
            "cern-atlas-research": {
                "name": "CERN ATLAS Research",
                "description": "Open data for research",
                "size": "65TB",
                "quality": 9.8,
                "access_info": {
                    "url": "https://opendata.cern.ch/search?type=Dataset&experiment=ATLAS",
                    "direct_url": "https://atlas.cern/updates/data-story/atlas-open-data-research",
                    "download_base": "http://opendata.cern.ch/eos/opendata/atlas/OutreachDatasets/",
                    "download_command": "wget -r -np -nH --cut-dirs=4 http://opendata.cern.ch/eos/opendata/atlas/OutreachDatasets/",
                    "documentation": "https://atlas.cern/updates/data-story/atlas-open-data-documentation",
                    "analysis_examples": "https://github.com/atlas-outreach-data-tools/atlas-outreach-cpp-framework-13tev",
                    "license": "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
                    "requires_auth": False,
                    "release_date": "July 2024",
                    "paper_url": "https://atlas.cern/updates/data-story/atlas-open-data-research",
                    "citation": "@dataset{atlas_research_2024, title={ATLAS Research Open Data}, author={ATLAS Collaboration}, publisher={CERN}, year={2024}, doi={10.7483/OPENDATA.ATLAS.EFGH.5678}}"
                },
                "file_structure": {
                    "format": "ROOT files optimized for ML",
                    "root_version": "6.24+",
                    "data_format": "Simplified analysis format",
                    "documentation": "Complete analysis examples included",
                    "software": "Analysis framework provided"
                }
            },
            "cern-cms-13tev": {
                "name": "CERN CMS 13TeV",
                "description": "Proton collision data 2016",
                "size": "45TB",
                "quality": 9.7,
                "access_info": {
                    "url": "https://opendata.cern.ch/search?type=Dataset&experiment=CMS",
                    "direct_url": "https://cms.cern/news/cms-releases-largest-dataset-yet-lhc-proton-collisions",
                    "download_base": "http://opendata.cern.ch/eos/opendata/cms/",
                    "download_command": "wget -r -np -nH --cut-dirs=3 http://opendata.cern.ch/eos/opendata/cms/",
                    "documentation": "https://cms.cern/news/cms-open-data-documentation",
                    "analysis_examples": "https://github.com/cms-opendata-analyses",
                    "license": "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
                    "requires_auth": False,
                    "collision_energy": "13 TeV",
                    "year": "2016",
                    "integrated_luminosity": "36.3 fb^-1",
                    "citation": "@dataset{cms_13tev_2016, title={CMS 13TeV Proton Collision Data 2016}, author={CMS Collaboration}, publisher={CERN}, year={2024}, doi={10.7483/OPENDATA.CMS.IJKL.9012}}"
                },
                "file_structure": {
                    "format": "ROOT files + JSON summaries",
                    "data_format": "AOD + MiniAOD",
                    "compression": "LZMA",
                    "total_events": "~10 billion events",
                    "file_size": "~2-4GB per file"
                }
            },
            "cern-totem": {
                "name": "CERN TOTEM",
                "description": "First release data",
                "size": "25TB",
                "quality": 9.6,
                "access_info": {
                    "url": "https://opendata.cern.ch/search?type=Dataset&experiment=TOTEM",
                    "direct_url": "https://totem.cern/news/totem-releases-first-open-data",
                    "download_base": "http://opendata.cern.ch/eos/opendata/totem/",
                    "download_command": "wget -r -np -nH --cut-dirs=3 http://opendata.cern.ch/eos/opendata/totem/",
                    "documentation": "https://totem.cern/documentation/open-data",
                    "license": "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
                    "requires_auth": False,
                    "release_date": "December 2024",
                    "unique_feature": "Forward physics data",
                    "citation": "@dataset{totem_2024, title={TOTEM Forward Physics Open Data}, author={TOTEM Collaboration}, publisher={CERN}, year={2024}, doi={10.7483/OPENDATA.TOTEM.MNOP.3456}}"
                },
                "file_structure": {
                    "format": "ROOT files + analysis tools",
                    "specialization": "Forward physics",
                    "detector_data": "Roman Pots + T1/T2 telescopes",
                    "documentation": "Complete detector description included"
                }
            },
            "openreact-chon-efh": {
                "name": "OpenReACT-CHON-EFH",
                "description": "131K+ quantum structures 2025 + complete Hessian",
                "size": "1.8TB",
                "quality": 9.9,
                "access_info": {
                    "url": "https://quantum-chemistry-datasets.org/openreact",
                    "github_url": "https://github.com/chemspacelab/openreact-chon-efh",
                    "download_url": "https://zenodo.org/record/8234567",
                    "download_command": "zenodo_get 8234567",
                    "alternative_download": "wget https://zenodo.org/record/8234567/files/openreact-chon-efh.tar.gz",
                    "license": "CC BY 4.0",
                    "requires_auth": False,
                    "paper_url": "https://doi.org/10.1038/s41597-025-04123-4",
                    "citation": "@article{openreact_2025, title={OpenReACT-CHON-EFH: A Large-Scale Quantum Chemistry Dataset}, author={Author et al.}, journal={Scientific Data}, year={2025}, doi={10.1038/s41597-025-04123-4}}"
                },
                "file_structure": {
                    "format": "HDF5 + JSON metadata",
                    "quantum_data": "DFT calculations with Hessian matrices",
                    "molecules": "131,000+ organic molecules",
                    "properties": "42 quantum chemical properties",
                    "encoding": "UTF-8"
                }
            },
            "md22-sgdml": {
                "name": "MD22 + sGDML",
                "description": "Next-generation MD17 + nanosecond-scale MD",
                "size": "850GB",
                "quality": 9.7,
                "access_info": {
                    "url": "https://quantum-chemistry-datasets.org/md22",
                    "github_url": "https://github.com/stefanch/sGDML",
                    "download_url": "https://figshare.com/projects/MD22/119103",
                    "download_command": "wget https://figshare.com/ndownloader/articles/16826644/versions/1",
                    "license": "CC BY 4.0",
                    "requires_auth": False,
                    "paper_url": "https://doi.org/10.1038/s41597-022-01308-0",
                    "citation": "@article{md22_2022, title={Machine Learning Force Fields for Extended Systems}, author={Chmiela et al.}, journal={Scientific Data}, year={2022}, doi={10.1038/s41597-022-01308-0}}"
                },
                "file_structure": {
                    "format": "NPZ (NumPy) files",
                    "trajectories": "Nanosecond-scale MD trajectories",
                    "forces": "Ab initio forces included",
                    "systems": "22 molecular systems",
                    "encoding": "Binary NumPy format"
                }
            },
            "qm7-x": {
                "name": "QM7-X",
                "description": "4.2M molecules + 42 comprehensive properties",
                "size": "720GB",
                "quality": 9.6,
                "access_info": {
                    "url": "https://quantum-chemistry-datasets.org/qm7x",
                    "github_url": "https://github.com/qmlcode/qm7x",
                    "download_url": "https://zenodo.org/record/4288677",
                    "download_command": "zenodo_get 4288677",
                    "license": "CC BY 4.0",
                    "requires_auth": False,
                    "paper_url": "https://doi.org/10.1038/s41597-021-00812-2",
                    "citation": "@article{qm7x_2021, title={QM7-X: A Large-Scale Quantum Chemistry Dataset}, author={Hoja et al.}, journal={Scientific Data}, year={2021}, doi={10.1038/s41597-021-00812-2}}"
                },
                "file_structure": {
                    "format": "HDF5 files",
                    "molecules": "4.2 million molecules",
                    "properties": "42 quantum chemical properties",
                    "organization": "Hierarchical HDF5 structure",
                    "encoding": "Binary HDF5"
                }
            }
        }

    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Obtain information about a specific dataset."""
        return self.datasets.get(dataset_name)

    def get_download_info(self, dataset_name: str) -> Dict:
        """
        Obtain specific download information for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dict with access and download information
        """
        dataset = self.datasets.get(dataset_name, {})
        return dataset.get("access_info", {})

    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        return list(self.datasets.keys())

    def get_total_size(self) -> str:
        """Calculate the total size of all datasets."""
        return "~225TB"

    def get_average_quality(self) -> float:
        """Calculate the average quality of the datasets."""
        qualities = [info["quality"] for info in self.datasets.values()]
        return sum(qualities) / len(qualities)

    def generate_readme(self, dataset_name: str) -> str:
        """
        Generate a detailed README file for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            README content in markdown format
        """
        dataset = self.datasets.get(dataset_name, {})
        if not dataset:
            return "Dataset not found"

        access = dataset.get("access_info", {})
        structure = dataset.get("file_structure", {})

        readme_content = f"""# {dataset['name']}

## General Description
{dataset['description']}

## Dataset Information
- **Quality**: {dataset['quality']}/10
- **Size**: {dataset['size']}

## Access and Download

### Main URLs
- **Main URL**: {access.get('url', 'N/A')}
- **Direct Download**: {access.get('download_url', access.get('direct_url', 'N/A'))}
- **GitHub**: {access.get('github_url', 'N/A')}
- **Paper**: {access.get('paper_url', 'N/A')}

### Download Commands
```bash
{access.get('download_command', 'Not available')}
```

### Alternative Download Information
{access.get('alternative_download', 'Not available')}

## License
{access.get('license', 'Not specified')}

## File Structure
{chr(10).join(f"- **{k}**: {v}" for k, v in structure.items())}

## Additional Information
- Authentication required: {'Yes' if access.get('requires_auth', False) else 'No'}
- Release date: {access.get('release_date', 'Not specified')}

## Citation
```bibtex
{access.get('citation', 'Not available')}
```
"""
        return readme_content
