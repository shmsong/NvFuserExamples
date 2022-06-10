from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='examples',
    ext_modules=[
        CUDAExtension(
            name='examples',
            pkg='examples',
            include_dirs=[ this_dir],
            sources=['example_uses/example0.cpp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
