# EV SDF Utils

This repo contains CUDA functions for fast trilinear interpolation and marching cubes on pytorch tensors.
If the tensors are on CPU, it will fall back to scipy and scikit-image functions for marching cubes and trilinear interpolation.
If the tensors are on CUDA however, custom cuda kernels will be executed to speed up the computations.

This code was initially developed for [DiffSDFSim](https://diffsdfsim.is.tue.mpg.de).
If you use this code for your research, please cite DiffSDFSim:
```
@inproceedings{strecke2021_diffsdfsim,
  title = {{DiffSDFSim}: Differentiable Rigid-Body Dynamics With Implicit Shapes},
  author = {Strecke, Michael and Stueckler, Joerg},
  booktitle = {International Conference on {3D} Vision ({3DV})},
  month = dec,
  year = {2021},
  doi = {10.1109/3DV53792.2021.00020},
  month_numeric = {12}
}
```

## Installation
0. Make sure you have your environment with `Python>=3.7` and CUDA set up for the PyTorch version you're aiming to use.
1. Run the following command from the root of this repo to install the module and its dependencies:
   ```bash
   pip install .
   ```
   Running this command will install all required dependencies (i.e. current versions of torch, scipy and scikit-image) and build the CUDA  extensions for the newest pytorch version.

   If you don't want to clone the repo, you can also directly install the package via `pip`'s `git`-interface:
   ```bash
   pip install git+https://github.com/EmbodiedVision/ev-sdf-utils.git
   ```

***IMPORTANT:*** If you already have a PyTorch version installed, the build might not be compatible with it as it uses the most recent PyTorch in an isolated build environment during the build process.
In this case, make sure you have torch, setuptools and wheel installed and run the build without the build-isolation (see the [pip Documentation](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/#disabling-build-isolation)). 
```bash
pip install -U pip setuptools wheel
pip install . --no-build-isolation
```
Setting the environment variable `PIP_NO_BUILD_ISOLATION=0` has the same effect as using the `--no-build-isolation` flag (see the [pip Documentation](https://pip.pypa.io/en/stable/topics/configuration/#environment-variables)).
This can be useful if pip is run from a different process, e.g. inside a conda environment setup.

## Usage

### TL;DR
After [installation](#installation), the module `ev_sdf_utils` is available and provides two functions:
```python
from ev_sdf_utils import marching_cubes, grid_interp
```
Please refer to the documentation strings in [utils.py](src/ev_sdf_utils/utils.py) on how to use them.


### Example Script
The file [test.py](test.py) contains an example on how to use the provided functions.
It will run marching cubes and grid interpolation examples with runtime comparisons between CPU and CUDA implementations and check for differences in the results (basically checking the custom cuda kernels against scipy/scikit-image).
To run the test file, you need to install additional modules as specified in [requirements.txt](requirements.txt).
You can install them by running
```bash
pip install -r requirements.txt
```

## Troubleshooting

### `no kernel image available for device` Error

By default, torch will only build for the CUDA architecture present on your system.
If you want to build the module for several CUDA architectures, set the environment variable `TORCH_CUDA_ARCH_LIST` accordingly, for example like below:
```bash
export TORCH_CUDA_ARCH_LIST="5.0 5.2 6.0 6.1 7.0 7.5 8.0 8.6"
pip install .
```
You can look up which GPU needs which CUDA architecture on the [CUDA Wikipedia page](https://en.wikipedia.org/wiki/CUDA).

## License
This code is licensed under the Apache License, Version 2.0, (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).

