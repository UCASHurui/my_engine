#pragma once

#include <cuda_runtime.h>

namespace MyEngine {

// CUDA vector math helpers
inline __host__ __device__ float3 vadd(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 vsub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 vmul(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ float3 vdiv(float3 a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

inline __host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline __host__ __device__ float length(float3 a) {
    return sqrtf(dot(a, a));
}

inline __host__ __device__ float3 normalize(float3 a) {
    float len = length(a);
    if (len > 0) {
        return vdiv(a, len);
    }
    return a;
}

inline __host__ __device__ float3 fminv(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

inline __host__ __device__ float3 fmaxv(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

} // namespace MyEngine
