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
import time

import numpy as np
import torch
import trimesh
from sklearn.neighbors import NearestNeighbors

import ev_sdf_utils


# From https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist


def grid_grads(vol):
    """
    Finite-difference gradients for a given volume
    The input volume can be batched.
    """

    grads = torch.stack([
        torch.cat([
            vol.new_zeros(vol.shape[:-3] + (1,) + vol.shape[-2:]),
            (vol[..., 2:, :, :] - vol[..., :-2, :, :]) / 2.,
            vol.new_zeros(vol.shape[:-3] + (1,) + vol.shape[-2:])
        ], dim=-3),
        torch.cat([
            vol.new_zeros(vol.shape[:-2] + (1,) + vol.shape[-1:]),
            (vol[..., :, 2:, :] - vol[..., :, :-2, :]) / 2.,
            vol.new_zeros(vol.shape[:-2] + (1,) + vol.shape[-1:])
        ], dim=-2),
        torch.cat([
            vol.new_zeros(vol.shape[:-1] + (1,)),
            (vol[..., :, :, 2:] - vol[..., :, :, :-2]) / 2.,
            vol.new_zeros(vol.shape[:-1] + (1,))
        ], dim=-1)
    ], dim=0 if vol.ndim == 3 else 1)

    return grads


def run_mc_interp(sdf, sdf_grads, inds):
    start_mc = time.time()
    vs, fs = [], []
    for s in sdf:
        v, f = ev_sdf_utils.marching_cubes(s, 0.)
        vs.append(v)
        fs.append(f)
    time_mc = time.time() - start_mc

    start_interp = time.time()
    vals = ev_sdf_utils.grid_interp(sdf_grads, inds, bounds_error=False)
    if sdf_grads.is_cuda:
        torch.cuda.synchronize()
    time_interp = time.time() - start_interp

    return vs, fs, vals, time_mc, time_interp


sdf_file1 = np.load('data/obj_000001.npz')
sdf_file2 = np.load('data/obj_000010.npz')
sdfs = torch.stack([
    torch.from_numpy(sdf_file1['sdf']),
    torch.from_numpy(sdf_file2['sdf']),
])

sdf_grads = grid_grads(sdfs)

resolution = sdfs.shape
n_interp_points = 3000000

inds = (torch.rand(sdfs.shape[0], n_interp_points, 3)
        * (torch.tensor(sdfs.shape[1:]) + 1.).unsqueeze(0).repeat(sdfs.shape[0], 1, 1)) - 1.

vs_cpu, fs_cpu, vals_cpu, time_mc_cpu, time_interp_cpu = run_mc_interp(sdfs, sdf_grads, inds)

sdf_cuda = sdfs.cuda()
sdf_grads_cuda = sdf_grads.cuda()
inds_cuda = inds.cuda()

vs_cuda, fs_cuda, vals_cuda, time_mc_cuda, time_interp_cuda = run_mc_interp(sdf_cuda, sdf_grads_cuda, inds_cuda)

print("=" * 80)
print(f"Timings for MC at resolution {list(resolution)}:")
print(f"Time MC CPU: {time_mc_cpu}s")
print(f"Time MC CUDA: {time_mc_cuda}s")
print(f"Speedup: {time_mc_cpu / time_mc_cuda}")
print("-" * 80)
print(f"Timings for grid interpolations with {n_interp_points} points:")
print(f"Time interp CPU: {time_interp_cpu}s")
print(f"Time interp CUDA: {time_interp_cuda}s")
print(f"Speedup: {time_interp_cpu / time_interp_cuda}")
print("=" * 80)
print(f"Chamfer distance between meshes: "
      f"""{sum([chamfer_distance(v_cpu.numpy(), v_cuda.cpu().numpy()) for v_cpu, v_cuda in zip(vs_cpu, vs_cuda)])
           / sdfs.shape[0]}""")
print(f"Average absolute interpolation difference: {(vals_cpu - vals_cuda.cpu()).abs().nanmean()}")
print("=" * 80)

ms_cpu = [trimesh.Trimesh(v_cpu, f_cpu, vertex_colors=[0, 255, 0]) for v_cpu, f_cpu in zip(vs_cpu, fs_cpu)]
ms_cuda = [trimesh.Trimesh(v_cuda.cpu(), f_cuda.cpu(), vertex_colors=[0, 0, 255])
           for v_cuda, f_cuda in zip(vs_cuda, fs_cuda)]

scenes = [trimesh.Scene([m_cpu, m_cuda]) for m_cpu, m_cuda in zip(ms_cpu, ms_cuda)]

print("Showing computed meshes (green: cpu, blue: cuda), exit with \"q\"")
print("=" * 80)
for scene in scenes:
    scene.show()
