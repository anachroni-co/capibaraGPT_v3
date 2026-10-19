#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Systems & Logs Datasets Manager - CapibaraGPT v3
Specialized datasets for computational systems, logs, security and performance.

This module manages world-class datasets from authoritative sources such as:
- Google, NASA, Intel, NIST
- Institutional research labs
- Industry standard benchmarks
"""

import os
import json
import zipfile
import tarfile
import logging
import requests
from pathlib import Path

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

@dataclass
class SystemDataConfig:
    """Configuration for system datasets."""
    name: str
    source: str
    url: str
    description: str
    category: str
    data_types: List[str]
    size_estimate: str
    quality_score: float
    use_cases: List[str]
    format: str
    license: str
    documentation_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    requires_auth: bool = False

class SystemsLogsDataManager:
    """
    Manager for specialized computational systems datasets.

    Features:
    - 10 world-class datasets from authoritative sources
    - Real production data from Google, NASA, Intel
    - Complete coverage: logs, security, performance, I/O
    - Academically rigorous methodology
    """

    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the systems logs data manager.

        Args:
            base_dir: Base directory for storing datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Specialized directories
        self.logs_dir = self.base_dir / "logs"
        self.security_dir = self.base_dir / "security"
        self.performance_dir = self.base_dir / "performance"
        self.network_dir = self.base_dir / "network"
        self.storage_dir = self.base_dir / "storage"
        self.cicd_dir = self.base_dir / "cicd"

        # Create directory structure
        for directory in [self.logs_dir, self.security_dir, self.performance_dir,
                         self.network_dir, self.storage_dir, self.cicd_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "raw").mkdir(exist_ok=True)
            (directory / "processed").mkdir(exist_ok=True)
            (directory / "metadata").mkdir(exist_ok=True)

        # World-class dataset configurations
        self.dataset_configs = self._initialize_world_class_datasets()

    def _initialize_world_class_datasets(self) -> Dict[str, SystemDataConfig]:
        """
        TOP 10 MOST CURATED DATASETS - Rigorous Selection Criteria

        - Documented and peer-reviewed methodology
        - Real production systems data
        - Community adoption and active maintenance
        - Massive scale but well documented
        - Direct enterprise applicability

        AVERAGE QUALITY: 9.1/10 (Exceptional)
        """

        configs = {
            # 1. LogHub (CUHK) - The Gold Standard - 10.0/10
            "loghub": SystemDataConfig(
                name="LogHub",
                source="CUHK (Chinese University of Hong Kong)",
                url="https://github.com/logpai/loghub",
                description="16+ real production system datasets. Pre-parsed and labeled logs from HDFS, Spark, Linux kernel, Apache, etc. Perfect documentation + benchmarks.",
                category="logs",
                data_types=["system_logs", "parsed_logs", "labeled_data"],
                size_estimate="Several GB",
                quality_score=10.0,
                use_cases=["log_analysis", "anomaly_detection", "system_monitoring"],
                format="CSV/JSON/Raw logs",
                license="MIT",
                documentation_url="https://github.com/logpai/loghub/blob/master/README.md"
            ),

            # 2. Google Cluster Data (Kubernetes/Borg) - 10.0/10
            "google_cluster": SystemDataConfig(
                name="Google Cluster Data",
                source="Google Research",
                url="https://github.com/google/cluster-data",
                description="Real production cluster data from Google. 29 days of complete traces with resource requests/usage from millions of jobs. Massive scale but well documented.",
                category="performance",
                data_types=["cluster_traces", "resource_usage", "job_scheduling"],
                size_estimate="Several TB",
                quality_score=10.0,
                use_cases=["container_orchestration", "resource_management", "scheduling_optimization"],
                format="CSV/Protocol Buffers",
                license="Creative Commons",
                documentation_url="https://github.com/google/cluster-data/blob/master/README.md"
            ),

            # 3. CICIDS2017/2018 (Canadian Institute) - 9.5/10
            "cicids": SystemDataConfig(
                name="CICIDS2017/2018",
                source="Canadian Institute for Cybersecurity",
                url="https://www.unb.ca/cic/datasets/ids-2017.html",
                description="Most modern intrusion detection dataset. Real attacks in controlled environment with network flows + packet captures. Impeccable methodology, well labeled.",
                category="security",
                data_types=["network_flows", "packet_captures", "intrusion_labels"],
                size_estimate="8+ GB",
                quality_score=9.5,
                use_cases=["network_security", "ids_training", "attack_detection"],
                format="CSV/PCAP",
                license="Academic Use",
                documentation_url="https://www.unb.ca/cic/datasets/ids-2017.html"
            ),

            # 4. LANL Cybersecurity Datasets - 9.5/10
            "lanl_cyber": SystemDataConfig(
                name="LANL Cybersecurity",
                source="Los Alamos National Laboratory",
                url="https://csr.lanl.gov/data/",
                description="Real supercomputer data. 90 days of complete logs with authentication, process, network data. Scale: billions of events. Unprecedented scale.",
                category="security",
                data_types=["authentication_logs", "process_data", "network_data"],
                size_estimate="Several TB",
                quality_score=9.5,
                use_cases=["enterprise_security", "behavioral_analysis", "threat_hunting"],
                format="CSV/JSON",
                license="Public Domain",
                documentation_url="https://csr.lanl.gov/data/"
            ),

            # 5. SNIA I/O Traces - 9.0/10
            "snia_io": SystemDataConfig(
                name="SNIA I/O Traces",
                source="Storage Networking Industry Association",
                url="http://iotta.snia.org/traces",
                description="Industry standard for storage. Real enterprise system workloads with multiple storage types and patterns. Standardized and well documented format.",
                category="storage",
                data_types=["io_traces", "storage_workloads", "performance_metrics"],
                size_estimate="10+ GB",
                quality_score=9.0,
                use_cases=["storage_optimization", "io_analysis", "performance_tuning"],
                format="Binary traces/CSV",
                license="SNIA License",
                documentation_url="http://iotta.snia.org/traces/about"
            ),

            # 6. TravisTorrent (CI/CD) - 9.0/10
            "travis_torrent": SystemDataConfig(
                name="TravisTorrent",
                source="TestRoots Research Group",
                url="https://travistorrent.testroots.org",
                description="35M+ builds from open source projects. Complete CI/CD pipeline data with build times, test results, dependencies. Perfect longitudinal analysis.",
                category="cicd",
                data_types=["build_logs", "test_results", "ci_metrics"],
                size_estimate="Several GB",
                quality_score=9.0,
                use_cases=["devops_optimization", "build_prediction", "ci_analysis"],
                format="CSV/JSON",
                license="MIT",
                documentation_url="https://travistorrent.testroots.org/page_dataformat/"
            ),

            # 7. NASA System Logs - 9.0/10
            "nasa_logs": SystemDataConfig(
                name="NASA System Logs",
                source="NASA",
                url="https://www.kaggle.com/datasets/nasa/nasa-system-logs",
                description="Mission-critical systems data. Extremely well curated and documented with multiple system types. High reliability requirements.",
                category="logs",
                data_types=["mission_critical_logs", "system_telemetry", "reliability_data"],
                size_estimate="100+ MB",
                quality_score=9.0,
                use_cases=["critical_systems_analysis", "reliability_engineering", "fault_tolerance"],
                format="Log files/CSV",
                license="Public Domain",
                documentation_url="https://www.kaggle.com/datasets/nasa/nasa-system-logs"
            ),

            # 8. Intel PCM Performance Data - 8.5/10
            "intel_pcm": SystemDataConfig(
                name="Intel PCM Performance Data",
                source="Intel Corporation",
                url="https://github.com/intel/pcm",
                description="Hardware-level performance metrics. CPU utilization, cache behavior, power consumption from real systems under load. Performance-energy correlation.",
                category="performance",
                data_types=["performance_counters", "cpu_metrics", "power_data"],
                size_estimate="Variable",
                quality_score=8.5,
                use_cases=["performance_tuning", "power_optimization", "hardware_analysis"],
                format="CSV/JSON",
                license="BSD-3-Clause",
                documentation_url="https://github.com/intel/pcm/blob/master/README.md"
            ),

            # 9. CAIDA Internet Traces - 8.5/10
            "caida_traces": SystemDataConfig(
                name="CAIDA Internet Traces",
                source="CAIDA (Center for Applied Internet Data Analysis)",
                url="https://www.caida.org/catalog/datasets/",
                description="Real internet backbone traffic. 20+ years of historical data with multiple vantage points. Rigorous, peer-reviewed methodology. Authoritative source.",
                category="network",
                data_types=["internet_traces", "backbone_traffic", "routing_data"],
                size_estimate="Several TB",
                quality_score=8.5,
                use_cases=["network_analysis", "traffic_modeling", "internet_research"],
                format="PCAP/Custom binary",
                license="CAIDA Data License",
                documentation_url="https://www.caida.org/catalog/datasets/"
            ),

            # 10. SPEC Benchmark Repository - 8.0/10
            "spec_benchmarks": SystemDataConfig(
                name="SPEC Benchmark Repository",
                source="Standard Performance Evaluation Corporation",
                url="https://www.spec.org/benchmarks.html",
                description="Industry standard benchmarks. Results from thousands of configurations with exhaustive hardware/OS combinations. Globally standardized methodology. Gold standard benchmarks.",
                category="performance",
                data_types=["benchmark_results", "system_configurations", "performance_data"],
                size_estimate="Several GB",
                quality_score=8.0,
                use_cases=["performance_comparison", "system_characterization", "benchmarking"],
                format="CSV/XML/Custom",
                license="SPEC License",
                documentation_url="https://www.spec.org/benchmarks.html"
            )
        }

        return configs

    def list_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """List all available datasets with their metadata."""
        datasets_info = {}

        for dataset_id, config in self.dataset_configs.items():
            datasets_info[dataset_id] = {
                "name": config.name,
                "source": config.source,
                "description": config.description,
                "category": config.category,
                "quality_score": config.quality_score,
                "size_estimate": config.size_estimate,
                "use_cases": config.use_cases,
                "format": config.format,
                "requires_auth": config.requires_auth
            }

        return datasets_info

    def get_datasets_by_category(self, category: str) -> Dict[str, SystemDataConfig]:
        """Obtain datasets filtered by category."""
        return {
            dataset_id: config
            for dataset_id, config in self.dataset_configs.items()
            if config.category == category
        }

    def get_top_quality_datasets(self, min_score: float = 9.0) -> Dict[str, SystemDataConfig]:
        """Obtain datasets with quality score above the minimum."""
        return {
            dataset_id: config
            for dataset_id, config in self.dataset_configs.items()
            if config.quality_score >= min_score
        }

    def download_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Download detailed information for a specific dataset."""
        if dataset_id not in self.dataset_configs:
            logger.error(f"Dataset {dataset_id} not found")
            return None

        config = self.dataset_configs[dataset_id]

        # Determine destination directory
        category_dirs = {
            "logs": self.logs_dir,
            "security": self.security_dir,
            "performance": self.performance_dir,
            "network": self.network_dir,
            "storage": self.storage_dir,
            "cicd": self.cicd_dir
        }

        target_dir = category_dirs.get(config.category, self.base_dir)
        dataset_dir = target_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        metadata = {
            "id": dataset_id,
            "name": config.name,
            "source": config.source,
            "url": config.url,
            "description": config.description,
            "category": config.category,
            "data_types": config.data_types,
            "size_estimate": config.size_estimate,
            "quality_score": config.quality_score,
            "use_cases": config.use_cases,
            "format": config.format,
            "license": config.license,
            "documentation_url": config.documentation_url,
            "local_path": str(dataset_dir),
            "status": "metadata_downloaded"
        }

        metadata_file = dataset_dir / "dataset_info.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset {dataset_id} information saved to {metadata_file}")
        return metadata

    def simulate_download_status(self, dataset_id: str) -> Dict[str, Any]:
        """Simulate the download status of a dataset."""
        if dataset_id not in self.dataset_configs:
            return {"error": f"Dataset {dataset_id} not found"}

        config = self.dataset_configs[dataset_id]

        return {
            "dataset_id": dataset_id,
            "name": config.name,
            "source": config.source,
            "status": "ready_for_download",
            "estimated_size": config.size_estimate,
            "quality_score": config.quality_score,
            "requirements": {
                "auth_required": config.requires_auth,
                "license_agreement": config.license,
                "documentation": config.documentation_url
            },
            "download_instructions": f"Visit {config.url} for download instructions"
        }

    def get_category_summary(self) -> Dict[str, Dict[str, Any]]:
        """Obtain summary by categories."""
        categories = {}

        for config in self.dataset_configs.values():
            category = config.category
            if category not in categories:
                categories[category] = {
                    "count": 0,
                    "datasets": [],
                    "avg_quality": 0.0,
                    "total_size_estimates": []
                }

            categories[category]["count"] += 1
            categories[category]["datasets"].append(config.name)
            categories[category]["total_size_estimates"].append(config.size_estimate)

        # Calculate average quality
        for category in categories:
            category_configs = [c for c in self.dataset_configs.values() if c.category == category]
            categories[category]["avg_quality"] = sum(c.quality_score for c in category_configs) / len(category_configs)

        return categories

# Utility functions for integration with CapibaraGPT v3

def create_systems_datasets_manager(base_dir: Optional[str] = None) -> SystemsLogsDataManager:
    """Factory function to create the systems datasets manager."""
    if base_dir is None:
        base_dir = "data/systems_logs"

    return SystemsLogsDataManager(base_dir)

def get_world_class_datasets_summary() -> Dict[str, Any]:
    """
    Summary of the TOP 10 MOST CURATED DATASETS with rigorous criteria.
    Implements the user's specific usage recommendations.
    """
    manager = create_systems_datasets_manager()

    return {
        "total_datasets": len(manager.dataset_configs),
        "average_quality": sum(config.quality_score for config in manager.dataset_configs.values()) / len(manager.dataset_configs),
        "categories": list(set(config.category for config in manager.dataset_configs.values())),
        "top_sources": [
            'Google Research', 'NASA', 'Intel Corporation',
            'CUHK', 'Los Alamos National Laboratory',
            'Canadian Institute for Cybersecurity',
            'Storage Networking Industry Association',
            'CAIDA', 'Standard Performance Evaluation Corporation'
        ],
        "selection_criteria": {
            "methodology": "Documented and peer-reviewed",
            "data_quality": "Real production systems data",
            "community": "Community adoption and active maintenance",
            "scale": "Massive scale but well documented",
            "applicability": "Direct enterprise applicability"
        },
        "recommendations": {
            "getting_started_top3": {
                "datasets": ["loghub", "nasa_logs", "cicids"],
                "reason": "Easiest, best documented, clean and well-structured"
            },
            "serious_research_top5": {
                "datasets": ["loghub", "nasa_logs", "cicids", "google_cluster", "lanl_cyber"],
                "reason": "Add Google Cluster Data and LANL for advanced analysis"
            },
            "specific_applications": {
                "storage": ["snia_io"],
                "cicd": ["travis_torrent"],
                "performance": ["intel_pcm", "spec_benchmarks"],
                "network": ["caida_traces"]
            }
        },
        "quality_breakdown": {
            "perfect_10": ["loghub", "google_cluster"],
            "premium_9plus": ["cicids", "lanl_cyber", "snia_io", "travis_torrent", "nasa_logs"],
            "specialized_8plus": ["intel_pcm", "caida_traces", "spec_benchmarks"]
        },
        "category_summary": manager.get_category_summary()
    }

def get_recommended_datasets_by_use_case(use_case: str) -> Dict[str, Any]:
    """
    Implements the user's specific recommendations for different use cases.

    Args:
        use_case: 'beginners', 'research', 'storage', 'cicd', 'performance', 'network'

    Returns:
        Dictionary with recommended datasets and justification
    """
    recommendations = {
        'beginners': {
            'datasets': ['loghub', 'nasa_logs', 'cicids'],
            'description': 'Getting Started (Top 3)',
            'reason': 'LogHub - Easiest, best documented; NASA Logs - Clean, well-structured; CICIDS2017 - Modern security focus',
            'why_these': 'Perfect methodology, excellent documentation, clear use cases'
        },
        'research': {
            'datasets': ['loghub', 'nasa_logs', 'cicids', 'google_cluster', 'lanl_cyber'],
            'description': 'Serious Research (Full Top 5)',
            'reason': 'Add Google Cluster Data and LANL for advanced analysis',
            'why_these': 'Unique Google production data + LANL supercomputers, unprecedented scale'
        },
        'storage': {
            'datasets': ['snia_io'],
            'description': 'Specific Applications - Storage',
            'reason': 'SNIA traces - Industry standard for storage optimization',
            'why_these': 'Real enterprise workloads, standardized format, I/O analysis'
        },
        'cicd': {
            'datasets': ['travis_torrent'],
            'description': 'Specific Applications - CI/CD',
            'reason': 'TravisTorrent - 35M+ builds from open source projects',
            'why_these': 'DevOps optimization, build prediction, perfect longitudinal analysis'
        },
        'performance': {
            'datasets': ['intel_pcm', 'spec_benchmarks'],
            'description': 'Specific Applications - Performance',
            'reason': 'Intel PCM + SPEC - Hardware-level metrics + industry benchmarks',
            'why_these': 'Performance-energy correlation, globally standardized methodology'
        },
        'network': {
            'datasets': ['caida_traces'],
            'description': 'Specific Applications - Network',
            'reason': 'CAIDA traces - 20+ years of historical internet backbone data',
            'why_these': 'Authoritative source, rigorous peer-reviewed methodology'
        }
    }

    if use_case not in recommendations:
        return {"error": f"Use case '{use_case}' not found. Options: {list(recommendations.keys())}"}

    return recommendations[use_case]

if __name__ == "__main__":
    # Functionality demo
    manager = create_systems_datasets_manager()

    logger.info("Systems & Logs Datasets Manager - CapibaraGPT v3")
    logger.info("=" * 60)

    # Show available datasets
    logger.info(f"Total world-class datasets: {len(manager.dataset_configs)}")

    # Show by category
    categories = manager.get_category_summary()
    for category, info in categories.items():
        logger.info(f"{category.upper()}: {info['count']} datasets (average quality: {info['avg_quality']:.1f}/10)")

    # Show top quality
    top_quality = manager.get_top_quality_datasets()
    logger.info(f"\nMaximum quality datasets (9.0+): {len(top_quality)}")
    for dataset_id, config in top_quality.items():
        logger.info(f"   {config.name} ({config.source}) - {config.quality_score}/10")
