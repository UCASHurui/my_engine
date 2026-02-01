#pragma once

#include "ray_types.h"
#include "bvh_builder.h"
#include <cuda_runtime.h>

namespace MyEngine {

// Simple path tracing kernel
__global__ void pathTraceKernel(
    float3* output,
    float3* accumulation,
    Camera camera,
    int width,
    int height,
    int sample_index,
    int max_bounces,
    float Russian_roulette,
    float min_contribution,
    const BVHNode* bvh_nodes,
    const int* bvh_indices,
    const Triangle* triangles
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_idx = y * width + x;
    SamplerState sampler;
    sampler.init(sample_index * width * height + pixel_idx);

    float3 pixel_color = make_float3(0, 0, 0);
    float3 throughput = make_float3(1, 1, 1);

    // Generate primary ray
    Ray ray = camera.getRay(
        (x + sampler.nextFloat()) / (float)width,
        (y + sampler.nextFloat()) / (float)height,
        sampler
    );

    HitRecord hit;
    hit.t = 1e20f;

    BVHTraverser bvh(bvh_nodes, bvh_indices, triangles);

    // Trace path
    for (int bounce = 0; bounce < max_bounces; bounce++) {
        hit.t = 1e20f;
        bool hit_something = bvh.traverse(ray, hit, 1e20f);

        if (!hit_something) {
            // Sky color gradient
            float t = 0.5f * (ray.direction.y + 1.0f);
            float3 sky = make_float3(
                (1.0f - t) + 0.5f * t,
                (1.0f - t) + 0.7f * t,
                (1.0f - t) + 1.0f * t
            );
            pixel_color = make_float3(
                pixel_color.x + throughput.x * sky.x,
                pixel_color.y + throughput.y * sky.y,
                pixel_color.z + throughput.z * sky.z
            );
            break;
        }

        // Lambertian scatter
        float3 scatter_dir = sampler.randomOnHemisphere(hit.normal);
        float cos_theta = dot(hit.normal, scatter_dir);
        throughput = make_float3(
            throughput.x * 0.9f * cos_theta * 2.0f,
            throughput.y * 0.9f * cos_theta * 2.0f,
            throughput.z * 0.9f * cos_theta * 2.0f
        );

        // Russian roulette
        if (bounce > 2) {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (sampler.nextFloat() > p * Russian_roulette) {
                break;
            }
            throughput = make_float3(
                throughput.x / (p * Russian_roulette),
                throughput.y / (p * Russian_roulette),
                throughput.z / (p * Russian_roulette)
            );
        }

        // Check contribution threshold
        if (fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)) < min_contribution) {
            break;
        }

        ray = Ray(
            make_float3(
                hit.point.x + hit.normal.x * 0.001f,
                hit.point.y + hit.normal.y * 0.001f,
                hit.point.z + hit.normal.z * 0.001f
            ),
            scatter_dir
        );
    }

    // Store result
    if (accumulation != nullptr) {
        accumulation[pixel_idx] = make_float3(
            accumulation[pixel_idx].x + pixel_color.x,
            accumulation[pixel_idx].y + pixel_color.y,
            accumulation[pixel_idx].z + pixel_color.z
        );
        float inv_samples = 1.0f / (sample_index + 1);
        output[pixel_idx] = make_float3(
            accumulation[pixel_idx].x * inv_samples,
            accumulation[pixel_idx].y * inv_samples,
            accumulation[pixel_idx].z * inv_samples
        );
    } else {
        output[pixel_idx] = pixel_color;
    }
}

// Path tracer class
class PathTracer {
public:
    PathTracer() : _stream(0), _bvh_nodes(nullptr),
                   _bvh_indices(nullptr), _triangles(nullptr) {}

    bool initialize(int max_bounces = 4, float russian_roulette = 0.3f) {
        _config.max_bounces = max_bounces;
        _config.Russian_roulette = russian_roulette;
        _config.min_contribution = 0.01f;

        cudaError_t err = cudaStreamCreate(&_stream);
        return err == cudaSuccess;
    }

    void setBVH(const BVHNode* nodes, const int* indices, const Triangle* triangles) {
        _bvh_nodes = nodes;
        _bvh_indices = indices;
        _triangles = triangles;
    }

    bool render(float3* output, float3* accumulation,
                const Camera& camera, int width, int height,
                int sample_index) {
        if (!_bvh_nodes || !_triangles) return false;

        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);

        pathTraceKernel<<<grid, block, 0, _stream>>>(
            output, accumulation, camera, width, height, sample_index,
            _config.max_bounces, _config.Russian_roulette, _config.min_contribution,
            _bvh_nodes, _bvh_indices, _triangles
        );

        return cudaGetLastError() == cudaSuccess;
    }

    void shutdown() {
        if (_stream) {
            cudaStreamDestroy(_stream);
            _stream = 0;
        }
    }

private:
    struct Config {
        int max_bounces;
        float Russian_roulette;
        float min_contribution;
    } _config;

    cudaStream_t _stream;
    const BVHNode* _bvh_nodes;
    const int* _bvh_indices;
    const Triangle* _triangles;
};

} // namespace MyEngine
