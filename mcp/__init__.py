"""
MCP (Model Control Plane) Package - Model versioning and routing.

This package provides model control plane functionality including version
management, resource allocation, and request routing for CapibaraGPT models.

Key Components:
    - VersionManager: Manages model versions and deployments
    - ModelVersion: Data class for version information
    - ResourceManager: Handles resource allocation for models
    - ResourceAllocation: Data class for resource assignments
    - ModelRouter: Routes requests to appropriate model instances
    - ModelRequest: Data class for model requests

Author: Skydesk International Dev Team.
"""

from .version_manager import VersionManager, ModelVersion
from .resource_manager import ResourceManager, ResourceAllocation
from .model_router import ModelRouter, ModelRequest

__all__ = [
    'VersionManager',
    'ModelVersion',
    'ResourceManager',
    'ResourceAllocation',
    'ModelRouter',
    'ModelRequest'
] 