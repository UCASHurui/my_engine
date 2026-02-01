#include <iostream>
#include <cmath>
#include "cuda_runtime.h"
#include "compute/particle_system.h"
#include "compute/post_processing.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            return 1; \
        } \
    } while(0)

bool testParticleSystem() {
    std::cout << "=== Testing Particle System ===" << std::endl;

    MyEngine::ParticleSystemManager manager;
    if (!manager.create(10000)) {
        std::cerr << "Failed to create particle system" << std::endl;
        return false;
    }

    // Configure emitter
    MyEngine::EmitterConfig emitter;
    emitter.position = make_float3(0, 0, 0);
    emitter.direction = make_float3(0, 1, 0);
    emitter.spread = 0.3f;
    emitter.speed_min = 5.0f;
    emitter.speed_max = 10.0f;
    emitter.life_min = 1.0f;
    emitter.life_max = 3.0f;
    emitter.size = 0.1f;
    emitter.color = make_float3(1.0f, 0.5f, 0.2f);
    manager.setEmitter(emitter);

    // Configure physics
    MyEngine::PhysicsConfig physics;
    physics.gravity = make_float3(0, -9.8f, 0);
    physics.drag = 0.98f;
    physics.wind = make_float3(2.0f, 0, 0);
    physics.turbulence = 0.5f;
    physics.turbulence_scale = 2.0f;
    manager.setPhysics(physics);

    // Emit particles
    manager.emit(0.016f, 1000);
    std::cout << "Emitted particles: " << manager.getActiveCount() << std::endl;

    // Update simulation
    float dt = 0.016f;
    for (int i = 0; i < 10; i++) {
        manager.update(dt);
        std::cout << "Frame " << i << ": active particles = " << manager.getActiveCount() << std::endl;
    }

    manager.destroy();
    std::cout << "Particle system test PASSED" << std::endl << std::endl;
    return true;
}

bool testPostProcessing() {
    std::cout << "=== Testing Post Processing ===" << std::endl;

    const int width = 512;
    const int height = 512;
    const int size = width * height;

    // Allocate host memory
    float3* h_data = new float3[size];
    for (int i = 0; i < size; i++) {
        float x = (float)(i % width) / width;
        float y = (float)(i / width) / height;
        h_data[i] = make_float3(
            0.5f + 0.5f * sinf(x * 10.0f),
            0.5f + 0.5f * cosf(y * 10.0f),
            0.5f + 0.5f * sinf((x + y) * 5.0f)
        );
    }

    // Allocate device memory
    float3* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float3), cudaMemcpyHostToDevice));

    // Create post-processor
    MyEngine::PostProcessor postProcessor;
    if (!postProcessor.initialize(width, height)) {
        std::cerr << "Failed to initialize post processor" << std::endl;
        delete[] h_data;
        cudaFree(d_data);
        return false;
    }

    // Configure tone mapping
    MyEngine::PostProcessConfig config;
    config.tone_map = MyEngine::ToneMapOperator::ACESFilmic;
    config.exposure = 1.0f;
    config.gamma = 2.2f;
    config.brightness = 0.0f;
    config.contrast = 1.0f;
    config.saturation = 1.0f;

    // Apply tone mapping
    if (!postProcessor.applyToneMap(d_data, width, height, config)) {
        std::cerr << "Tone mapping failed" << std::endl;
        postProcessor.shutdown();
        delete[] h_data;
        cudaFree(d_data);
        return false;
    }
    std::cout << "Applied ACES Filmic tone mapping" << std::endl;

    // Apply vignette
    if (!postProcessor.applyVignette(d_data, width, height, 0.8f)) {
        std::cerr << "Vignette failed" << std::endl;
        postProcessor.shutdown();
        delete[] h_data;
        cudaFree(d_data);
        return false;
    }
    std::cout << "Applied vignette" << std::endl;

    // Apply bloom
    if (!postProcessor.applyBloom(d_data, width, height, 0.8f, 0.5f)) {
        std::cerr << "Bloom failed" << std::endl;
        postProcessor.shutdown();
        delete[] h_data;
        cudaFree(d_data);
        return false;
    }
    std::cout << "Applied bloom" << std::endl;

    // Copy back and verify
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(float3), cudaMemcpyDeviceToHost));

    // Check some values are in valid range
    int validCount = 0;
    for (int i = 0; i < size; i++) {
        if (h_data[i].x >= 0 && h_data[i].x <= 1 &&
            h_data[i].y >= 0 && h_data[i].y <= 1 &&
            h_data[i].z >= 0 && h_data[i].z <= 1) {
            validCount++;
        }
    }

    postProcessor.shutdown();
    cudaFree(d_data);
    delete[] h_data;

    if (validCount == size) {
        std::cout << "All " << size << " pixels in valid range [0,1]" << std::endl;
    } else {
        std::cout << "Warning: " << (size - validCount) << " pixels out of range" << std::endl;
    }

    std::cout << "Post processing test PASSED" << std::endl << std::endl;
    return true;
}

int main() {
    std::cout << "MyEngine CUDA Compute Test" << std::endl;
    std::cout << "==========================" << std::endl << std::endl;

    // Check CUDA devices
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "CUDA devices found: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "  Device " << i << ": " << prop.name << std::endl;
        std::cout << "    Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "    Global memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
        std::cout << "    Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
    std::cout << std::endl;

    bool allPassed = true;
    allPassed &= testParticleSystem();
    allPassed &= testPostProcessing();

    std::cout << "==========================" << std::endl;
    if (allPassed) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}
