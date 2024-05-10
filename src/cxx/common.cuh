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

#include <cuda.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Overloaded helper function to generate the right vector type.
__inline__ __host__ __device__ auto T3(int) {
    return int3();
}

__inline__ __host__ __device__ auto T3(float) {
    return float3();
}

__inline__ __host__ __device__ auto T3(double) {
    return double3();
}

template<typename T>
__inline__ __host__ __device__ auto make_T3(T x, T y, T z) {
    auto v = T3(x);
    v.x = x; v.y = y; v.z = z;
    return v;
}

// Some required operators for vector types
inline __host__ __device__ float3 operator+ ( const float3& v1, const float3& v2 ) {
    return make_float3 ( v1.x + v2.x, v1.y + v2.y, v1.z + v2.z );
}

inline __host__ __device__ double3 operator+ ( const double3& v1, const double3& v2 ) {
    return make_double3 ( v1.x + v2.x, v1.y + v2.y, v1.z + v2.z );
}

inline __host__ __device__ float3 operator- ( const float3& v1, const float3& v2 ) {
    return make_float3 ( v1.x - v2.x, v1.y - v2.y, v1.z - v2.z );
}

inline __host__ __device__ double3 operator- ( const double3& v1, const double3& v2 ) {
    return make_double3 ( v1.x - v2.x, v1.y - v2.y, v1.z - v2.z );
}

inline __host__ __device__ float3 operator* ( const float f, const float3& v ) {
    return make_float3 ( f * v.x, f * v.y, f * v.z );
}

inline __host__ __device__ double3 operator* ( const double f, const double3& v ) {
    return make_double3 ( f * v.x, f * v.y, f * v.z );
}

inline __host__ __device__ float3 operator- ( const float3& v1, const int3& v2 ) {
    return make_float3 ( v1.x - static_cast<float> ( v2.x ),
                         v1.y - static_cast<float> ( v2.y ),
                         v1.z - static_cast<float> ( v2.z ) );
}

inline __host__ __device__ double3 operator- ( const double3& v1, const int3& v2 ) {
    return make_double3 ( v1.x - static_cast<double> ( v2.x ),
                          v1.y - static_cast<double> ( v2.y ),
                          v1.z - static_cast<double> ( v2.z ) );
}
