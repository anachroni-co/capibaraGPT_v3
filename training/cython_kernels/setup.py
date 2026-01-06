"""
Setup script for Cython kernels compilation
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Define extensions
extensions = [
    Extension(
        "consensus_kernels",
        ["consensus_kernels.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "routing_kernels", 
        ["routing_kernels.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "similarity_kernels",
        ["similarity_kernels.pyx"], 
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "aggregation_kernels",
        ["aggregation_kernels.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    name="capibara_cython_kernels",
    version="1.0.0",
    description="High-performance Cython kernels for Capibara meta-consensus",
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'profile': False,
        'linetrace': False,
    }),
    zip_safe=False,
)