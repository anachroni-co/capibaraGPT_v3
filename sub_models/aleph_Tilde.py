"""
sub_models aleph_Tilde module.

# This module provides functionality for aleph_Tilde.
"""

import os
import sys

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

if __name__ == "__main__":
    main()
