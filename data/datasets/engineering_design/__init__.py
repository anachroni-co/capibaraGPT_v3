"""
CapibaraGPT v3 Engineering Design Datasets

Specialized datasets for engineering design including:
- CAD designs (houses, parts, mechanical components)
- Electronics circuit design
- FPGA and hardware programming
- PCB routing and design
"""

from .electronics_datasets import ElectronicsDatasets, get_electronics_datasets
from .fpga_datasets import FPGADatasets, get_fpga_datasets
