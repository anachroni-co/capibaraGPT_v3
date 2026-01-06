"""
capibaras capibara2 module.

# This module provides functionality for capibara2.
"""

import os
import sys

import logging
# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module capibara2.py starting")
    return True

if __name__ == "__main__":
    main()
