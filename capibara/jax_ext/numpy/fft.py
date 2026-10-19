# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX FFT module - compatibility layer."""

import numpy as np
from typing import Any, Optional, Sequence, Union

# Try to import JAX FFT, fallback to numpy
try:
    import jax.numpy.fft as jax_fft
    HAS_JAX = True

    # Re-export JAX FFT functions
    fft = jax_fft.fft
    fft2 = jax_fft.fft2
    fftn = jax_fft.fftn
    fftfreq = jax_fft.fftfreq
    fftshift = jax_fft.fftshift
    hfft = jax_fft.hfft
    ifft = jax_fft.ifft
    ifft2 = jax_fft.ifft2
    ifftn = jax_fft.ifftn
    ifftshift = jax_fft.ifftshift
    ihfft = jax_fft.ihfft
    irfft = jax_fft.irfft
    irfft2 = jax_fft.irfft2
    irfftn = jax_fft.irfftn
    rfft = jax_fft.rfft
    rfft2 = jax_fft.rfft2
    rfftfreq = jax_fft.rfftfreq
    rfftn = jax_fft.rfftn

except ImportError:
    HAS_JAX = False

    # Fallback to numpy FFT
    fft = np.fft.fft
    fft2 = np.fft.fft2
    fftn = np.fft.fftn
    fftfreq = np.fft.fftfreq
    fftshift = np.fft.fftshift
    hfft = np.fft.hfft
    ifft = np.fft.ifft
    ifft2 = np.fft.ifft2
    ifftn = np.fft.ifftn
    ifftshift = np.fft.ifftshift
    ihfft = np.fft.ihfft
    irfft = np.fft.irfft
    irfft2 = np.fft.irfft2
    irfftn = np.fft.irfftn
    rfft = np.fft.rfft
    rfft2 = np.fft.rfft2
    rfftfreq = np.fft.rfftfreq
    rfftn = np.fft.rfftn

__all__ = [
    'fft', 'fft2', 'fftn', 'fftfreq', 'fftshift',
    'hfft', 'ifft', 'ifft2', 'ifftn', 'ifftshift',
    'ihfft', 'irfft', 'irfft2', 'irfftn',
    'rfft', 'rfft2', 'rfftfreq', 'rfftn',
]
