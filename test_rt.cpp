#include <iostream>
#include <vector>
#include <cmath>
#include "renderer/cuda/cuda_runtime_engine.h"
#include "renderer/cuda/rt/ray_types.h"
#include "renderer/cuda/rt/bvh_builder.h"

using namespace MyEngine;

// Simple test scene
void createTestTriangles(std::vector<Triangle>& triangles) {
    // Ground plane (two triangles)
    triangles.push_back(Triangle(
        make_float3(-10, 0, -10),
        make_float3(10, 0, -10),
        make_float3(-10, 0, 10),
        0
    ));
    triangles.push_back(Triangle(
        make_float3(-10, 0, 10),
        make_float3(10, 0, -10),
        make_float3(10, 0, 10),
        0
    ));

    // Red sphere approximation (octahedron)
    float s = 1.0f;
    triangles.push_back(Triangle(make_float3(0, s, 0), make_float3(s, 0, 0), make_float3(0, 0, s), 1));
    triangles.push_back(Triangle(make_float3(0, s, 0), make_float3(0, 0, s), make_float3(-s, 0, 0), 1));
    triangles.push_back(Triangle(make_float3(0, s, 0), make_float3(-s, 0, 0), make_float3(0, 0, -s), 1));
    triangles.push_back(Triangle(make_float3(0, s, 0), make_float3(0, 0, -s), make_float3(s, 0, 0), 1));
    triangles.push_back(Triangle(make_float3(0, -s, 0), make_float3(0, 0, s), make_float3(s, 0, 0), 1));
    triangles.push_back(Triangle(make_float3(0, -s, 0), make_float3(s, 0, 0), make_float3(0, 0, -s), 1));
    triangles.push_back(Triangle(make_float3(0, -s, 0), make_float3(0, 0, -s), make_float3(-s, 0, 0), 1));
    triangles.push_back(Triangle(make_float3(0, -s, 0), make_float3(-s, 0, 0), make_float3(0, 0, s), 1));
}

int main() {
    std::cout << "=== MyEngine Ray Tracing Test (CPU) ===" << std::endl;

    // Initialize CUDA
    if (!CUDARuntime::initialize()) {
        std::cerr << "CUDA init failed!" << std::endl;
        return 1;
    }

    std::cout << "Device: " << CUDARuntime::getDeviceName() << std::endl;
    std::cout << "Compute Capability: " << CUDARuntime::getComputeCapability() << std::endl;
    std::cout << "VRAM: " << (CUDARuntime::getTotalMemory() / 1024 / 1024) << " MB" << std::endl;
    std::cout << "RTX Support: " << (CUDARuntime::supportsRTX() ? "Yes" : "No") << std::endl;

    // Create test scene
    std::vector<Triangle> triangles;
    createTestTriangles(triangles);
    std::cout << "\nScene: " << triangles.size() << " triangles" << std::endl;

    // Build BVH
    std::cout << "Building BVH..." << std::endl;
    BVHBuilder bvh_builder;
    BVHBuildConfig config;
    config.max_leaf_primitives = 4;

    if (!bvh_builder.build(triangles.data(), (int)triangles.size(), config)) {
        std::cerr << "BVH build failed!" << std::endl;
        return 1;
    }

    BVHStats stats = bvh_builder.getStats();
    std::cout << "BVH built: " << stats.node_count << " nodes, "
              << stats.leaf_count << " leaves, depth " << stats.max_depth << std::endl;
    std::cout << "Memory usage: " << bvh_builder.getMemoryUsage() << " bytes" << std::endl;

    // Test camera
    std::cout << "\nTesting camera..." << std::endl;
    Camera camera;
    camera.initialize(
        make_float3(0, 2, 5),
        make_float3(0, 0.5, 0),
        make_float3(0, 1, 0),
        60.0f,
        1.0f
    );
    std::cout << "Camera initialized." << std::endl;

    // Test sampler
    std::cout << "Testing sampler..." << std::endl;
    SamplerState sampler;
    sampler.init(12345);
    float rnd = sampler.nextFloat();
    std::cout << "Random float: " << rnd << std::endl;

    // Shutdown
    CUDARuntime::shutdown();

    std::cout << "\n=== Ray Tracing Test Complete ===" << std::endl;

    return 0;
}
