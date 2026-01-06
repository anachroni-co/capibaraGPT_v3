"""
tools dataset module.

# This module provides functionality for dataset.
"""

import json
import os
import sys
import logging
import hashlib
# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from capibara.jax import jax
from functools import partial
import numpy as np # type: ignore
from capibara.jax.numpy import jnp
from dataclasses import dataclass, field
from sklearn.model_selection import KFold # type: ignore
from typing import Dict, Any, List, Optional, Iterator, Tuple, Union, cast, Generator

# setup de logging
logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module dataset.py starting")
    return True

if __name__ == "__main__":
    main()
