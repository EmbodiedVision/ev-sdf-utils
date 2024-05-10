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
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from skimage import measure

import ev_sdf_utils.cuda_sdf_utils as cuda_sdf_utils

from typing import Tuple


def marching_cubes(vol: torch.tensor, isolevel: float) -> Tuple[torch.tensor, torch.tensor]:
    """
    Extracts a triangular mesh from the volume at the given isosurface.
    :param vol: the volumetric grid to extract the surface from
    :param isolevel: the isolevel representing the surface
    :return: vertices and faces representing the triangle mesh as torch tensors
    """
    if vol.is_cuda:
        verts, faces = cuda_sdf_utils.marching_cubes(vol, isolevel)

        faces = faces.long()

        # Make vertices unique and remove degenerate faces
        verts, inds = verts.unique(dim=0, return_inverse=True)
        faces = inds[faces]
        faces = faces[
            (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])]
        return verts, faces
    else:
        v, t, _, _ = measure.marching_cubes(vol.numpy(), isolevel)
        return torch.from_numpy(v.copy()).to(vol.dtype), torch.from_numpy(t.copy()).long()


def grid_interp(vol: torch.tensor, inds: torch.tensor, bounds_error: bool = True, fill_value: float = float('nan'))\
        -> torch.tensor:
    """
    Trilinear grid interpolation in the volume
    :param vol: a grid of volumetric values
    :param inds: a tensor of shape (n, 3) of (possibly non-integer) indices to query in the volume
    :param bounds_error: whether to throw and error when the query index is out of bounds
    :param fill_value: the value to fill in for query points outside the volume
    :return: a tensor of shape (n) with the interpolated values
    """
    if inds.ndim == 3:
        assert vol.ndim >= 4 and inds.shape[0] == vol.shape[0]
        batched = True
    else:
        # Add singleton batch dim
        inds = inds.unsqueeze(0)
        vol = vol.unsqueeze(0)
        batched = False

    if vol.ndim == 4:
        # Add dim for n-d vectors
        vol = vol.unsqueeze(1)
    if vol.is_cuda and inds.is_cuda:
        if not vol.is_contiguous():
            vol = vol.contiguous()
        if not inds.is_contiguous():
            inds = inds.contiguous()
        vals = cuda_sdf_utils.grid_interp(vol, inds, bounds_error=bounds_error, fill_value=fill_value).squeeze(2)
    elif vol.is_cuda or inds.is_cuda:
        raise ValueError("Both vol and inds need to be on the same device!")
    else:
        vals = []
        for v, i in zip(vol, inds):
            v = v.permute(1, 2, 3, 0)
            x = range(v.shape[0])
            y = range(v.shape[1])
            z = range(v.shape[2])
            interp = RegularGridInterpolator((x, y, z), v.numpy(), bounds_error=bounds_error,
                                             fill_value=fill_value)
            vals.append(interp(i.detach().numpy()))
        vals = torch.from_numpy(np.stack(vals)).squeeze(2)

    if not batched:
        vals = vals.squeeze(0)
    return vals
