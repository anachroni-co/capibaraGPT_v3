"""
processors dataset_registry module.

# This module provides functionality for dataset_registry.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Set
from dataclasses import dataclass, field
import os
import wget
import gdown
import kaggle
import requests
import tarfile
import zipfile

from core.config import ModelConfig
from .dataset import DatasetConfig
from capibara.core.tokenizer import load_tokenizer_from_config
from .wiki_datasets import WikiDatasetManager
from .institutional_datasets import InstitutionalDatasetManager
from .academic_code_datasets import AcademicCodeDatasetManager
from .multimodal_conversation_datasets import MultimodalConversationManager
from .systems_logs_datasets import SystemsLogsDatasetManager
from .math_datasets import MathDatasetManager
from .physics_datasets import PhysicsDatasetManager
from .linux_datasets import LinuxDatasetManager

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module dataset_registry.py starting")
    return True

if __name__ == "__main__":
    main()
