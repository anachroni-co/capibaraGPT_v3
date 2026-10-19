"""
ViM-VQ: Vector Quantization with Vision in Mind

Implementation of Vector Quantization optimized for vision applications
with complete TPU support and CapibaraGPT-specific optimizations.
"""

from .quantizer import (
    ViMVQConfig,
    VIM_VQ_BASE,
    VIM_VQ_SMALL,
    VIM_VQ_LARGE,
    ViMVQQuantizer,
    create_vim_vq_quantizer,
)

__all__ = [
    'ViMVQQuantizer',
    'ViMVQConfig',
    'create_vim_vq_quantizer',
    'VIM_VQ_SMALL',
    'VIM_VQ_BASE',
    'VIM_VQ_LARGE'
]

__version__ = '1.0.0'
