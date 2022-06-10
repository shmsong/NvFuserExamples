from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='mha_manual',
    ext_modules=[
        CUDAExtension(
            name='mha_manual',
            pkg='mha_manual',
            include_dirs=[ this_dir],
            sources=['fmha/fmha_manual.cpp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
