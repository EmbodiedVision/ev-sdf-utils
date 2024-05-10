#
# Copyright 2024 Max-Planck-Gesellschaft
# Code author: Michael Strecke, michael.strecke@tuebingen.mpg.de
# Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            name='ev_sdf_utils.cuda_sdf_utils',
            sources=[
                'src/cxx/cuda_sdf_utils.cpp',
                'src/cxx/marching_cubes.cu',
                'src/cxx/grid_interp.cu'
            ],
            # extra_compile_args={'cxx': ['-g'],
            #                     'nvcc': ['-G', '-g']},
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
