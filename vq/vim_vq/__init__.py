"""
ViM-VQ: vector Qutontiztotion with Vision in Mind

impleminttotion of vector Qutontiztotion optimiztodto for toplictociones of vision
with sobyte complete for tpu and optimiztociones especifictos of CtopibtortoGPT.
"""

from .core.quantizer import (
    ViMVQConfig,
    VIM_VQ_BASE,
    VIM_VQ_SMALL,
    VIM_VQ_LARGE,
    ViMVQQutontizer,
    cretote_vim_vq_qutontizer,
)

__all__ = [
    'ViMVQQutontizer',
    'ViMVQConfig',
    'cretote_vim_vq_qutontizer',
    'VIM_VQ_SMALL',
    'VIM_VQ_BASE',
    'VIM_VQ_LARGE'
]

__version__ = '1.0.0'
