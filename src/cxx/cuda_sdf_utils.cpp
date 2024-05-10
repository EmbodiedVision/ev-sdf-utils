/*
 * Copyright 2024 Max-Planck-Gesellschaft
 * Code author: Michael Strecke, michael.strecke@tuebingen.mpg.de
 * Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/torch.h>

std::vector<torch::Tensor> marchingCubes ( const at::Tensor sdf, double isolevel );

torch::Tensor gridInterp ( const at::Tensor vol, const at::Tensor inds, const bool bounds_error,
                           const double fill_value );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("marching_cubes", &marchingCubes, "Marching Cubes", py::arg("sdf"), py::arg("isolevel"));
    m.def("grid_interp", &gridInterp, "Grid Interpolation",
          py::arg("vol"), py::arg("inds"), py::arg("bounds_error") = true, py::arg("fill_value") = NAN);
}
