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
#include "common.cuh"

template<typename T, typename T3>
inline __device__
T interpolateTrilinear ( const torch::TensorAccessor<T, 3, at::DefaultPtrTraits, signed int> vol, const T3& idx,
                         const int3& volSize, const double fill_value ) {
    if ( idx.x < 0 || idx.x > volSize.x - 1 ||
         idx.y < 0 || idx.y > volSize.y - 1 ||
         idx.z < 0 || idx.z > volSize.z - 1 )
        return fill_value;

    const int3 lowIdx = make_int3 ( static_cast<int> ( idx.x ),
                                    static_cast<int> ( idx.y ),
                                    static_cast<int> ( idx.z ) );

    // By the test above, lowIdx.i + 1 is only greate than volSize.i - 1 if lowIdx.i == volSize.i
    // We cap the highIdx in this case as interpFac.i will be 0 and we will only be using values from lowIdx.i
    // in the interpolation.
    const int3 highIdx = make_int3 ( min(lowIdx.x + 1, volSize.x - 1),
                                     min(lowIdx.y + 1, volSize.y - 1),
                                     min(lowIdx.z + 1, volSize.z - 1) );

    const T3 interpFac = idx - lowIdx;

    T vs[] = { vol[lowIdx.x][lowIdx.y][lowIdx.z],
               vol[highIdx.x][lowIdx.y][lowIdx.z],
               vol[lowIdx.x][highIdx.y][lowIdx.z],
               vol[highIdx.x][highIdx.y][lowIdx.z],
               vol[lowIdx.x][lowIdx.y][highIdx.z],
               vol[highIdx.x][lowIdx.y][highIdx.z],
               vol[lowIdx.x][highIdx.y][highIdx.z],
               vol[highIdx.x][highIdx.y][highIdx.z]
             };

    for ( int i = 0; i < 4; ++i ) {
        vs[i] = ( 1 - interpFac.x ) * vs[ 2 * i ]
                + interpFac.x * vs[ 2 * i + 1 ];
    }

    for ( int i = 0; i < 2; ++i ) {
        vs[i] = ( 1 - interpFac.y ) * vs[ 2 * i ]
                + interpFac.y * vs[ 2 * i + 1 ];
    }

    return ( 1 - interpFac.z ) * vs[0] + interpFac.z * vs[1];
}

template<typename T>
__global__
void kernel_gridInterp ( const torch::PackedTensorAccessor32<T, 5> vol,
                         const T* inds_ptr,
                         const int3 volSize,
                         const int n_batch,
                         const int n_inds,
                         const int dim,
                         torch::PackedTensorAccessor32<T, 3> vals,
                         const double fill_value ) {
    const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = blockIdx.y * blockDim.y + threadIdx.y;

    if ( b_idx >= n_batch || idx >= n_inds )
        return;

    auto v = T3(T());
    auto inds = (decltype(v) *)inds_ptr;

    for ( int i = 0; i < dim; ++i )
        vals[b_idx][idx][i] = interpolateTrilinear ( vol[b_idx][i], inds[b_idx * n_inds + idx], volSize, fill_value );
}

torch::Tensor gridInterp ( const at::Tensor vol, const at::Tensor inds, const bool bounds_error,
                           const double fill_value ) {
    assert(("Volume must be floating point!", vol.is_floating_point()));
    assert(("Volume and inds must have the same scalar type!", vol.scalar_type() == inds.scalar_type()));
    assert(("Volume must be contiguous!", vol.is_contiguous()));
    assert(("Indices must be contiguous!", inds.is_contiguous()));

    dim3 threads ( 1, 1024 );
    dim3 blocks ( ( inds.size(0) + threads.x - 1 ) / threads.x, ( inds.size(1) + threads.y - 1 ) / threads.y );

    const int3 volSize = make_int3( vol.size(2), vol.size(3), vol.size(4) );

    auto max_inds = std::get<0>(inds.max(1));
    auto min_inds = std::get<0>(inds.min(1));

    auto vals = vol.new_zeros({inds.size(0), inds.size(1), vol.size(1)});
    AT_DISPATCH_FLOATING_TYPES(vol.scalar_type(), "gridInterp", ([&] {
        if (bounds_error) {
            bool index_within_bounds = true;
            for ( int i = 0; i < inds.size(0); ++i )
                index_within_bounds &= (max_inds[i][0].item<scalar_t>() <= volSize.x - 1 &&
                                        min_inds[i][0].item<scalar_t>() >= 0 &&
                                        max_inds[i][1].item<scalar_t>() <= volSize.y - 1 &&
                                        min_inds[i][1].item<scalar_t>() >= 0 &&
                                        max_inds[i][2].item<scalar_t>() <= volSize.z - 1 &&
                                        min_inds[i][2].item<scalar_t>() >= 0);
            assert(("Query index out of bounds!", index_within_bounds));
        }
        kernel_gridInterp<<<blocks, threads>>> (
            vol.packed_accessor32<scalar_t, 5>(), inds.data_ptr<scalar_t>(), volSize, (int) inds.size(0),
            (int) inds.size(1), (int) vol.size(1), vals.packed_accessor32<scalar_t, 3>(), fill_value );
    }));

    gpuErrchk( cudaGetLastError() );

    return vals;
}
