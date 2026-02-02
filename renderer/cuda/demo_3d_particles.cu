/**
 * 3D Particle System Demo
 *
 * Demonstrates CUDA compute primitives for 3D graphics:
 * - Parallel reduction for bounding box, center of mass, total energy
 * - Prefix sum (scan) for spatial partitioning
 * - Stream compaction for active particle filtering
 * - GPU-based statistics for performance analysis
 *
 * Compile: nvcc -o demo_3d demo_3d_particles.cu -O3 --use_fast_math -arch=sm_89
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <filesystem>

#include "cuda_runtime.h"
#include "cuda_math.h"
#include "compute/particle_system.h"
#include "compute/reduction.h"
#include "compute/scan.h"
#include "profiler/cuda_profiler.h"

// Include implementations
#include "compute/reduction.cu"
#include "compute/scan.cu"

using namespace MyEngine;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            return 1; \
        } \
    } while(0)

// 3D Particle structure
struct Particle3D {
    float3 position;
    float3 velocity;
    float3 acceleration;
    float life;
    float max_life;
    float size;
    float3 color;
};

// Output formats
enum class OutputFormat { ASCII, OBJ, BINARY };

// Simple random number generator
__device__ __host__ inline float random(unsigned int& seed, float minVal, float maxVal) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return minVal + (float)(seed % 100000) / 100000.0f * (maxVal - minVal);
}

// GPU kernel: Initialize particles in a sphere
__global__ void initSphereKernel(Particle3D* particles, int count, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    unsigned int rng = seed + idx * 7919;

    // Random point in unit sphere
    float theta = random(rng, 0.0f, 2.0f * M_PI);
    float phi = random(rng, 0.0f, M_PI);
    float r = powf(random(rng, 0.0f, 1.0f), 1.0f/3.0f);

    Particle3D& p = particles[idx];
    p.position = make_float3(
        r * sinf(phi) * cosf(theta),
        r * sinf(phi) * sinf(theta),
        r * cosf(phi)
    );
    p.velocity = make_float3(
        random(rng, -0.5f, 0.5f),
        random(rng, -0.5f, 0.5f),
        random(rng, -0.5f, 0.5f)
    );
    p.acceleration = make_float3(0, 0, 0);
    p.life = random(rng, 0.5f, 1.0f);
    p.max_life = p.life;
    p.size = random(rng, 0.02f, 0.08f);

    // Color based on position
    p.color = make_float3(
        0.5f + p.position.x * 0.5f,
        0.5f + p.position.y * 0.5f,
        0.5f + p.position.z * 0.5f
    );
}

// GPU kernel: Initialize particles in a ring
__global__ void initRingKernel(Particle3D* particles, int count, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    unsigned int rng = seed + idx * 7919;

    float theta = random(rng, 0.0f, 2.0f * M_PI);
    float radius = 0.8f + random(rng, -0.1f, 0.1f);

    Particle3D& p = particles[idx];
    p.position = make_float3(
        radius * cosf(theta),
        random(rng, -0.2f, 0.2f),
        radius * sinf(theta)
    );
    p.velocity = make_float3(
        -sinf(theta) * 0.5f,
        random(rng, -0.2f, 0.2f),
        cosf(theta) * 0.5f
    );
    p.acceleration = make_float3(0, 0, 0);
    p.life = random(rng, 0.5f, 1.0f);
    p.max_life = p.life;
    p.size = random(rng, 0.03f, 0.06f);
    p.color = make_float3(1.0f, 0.5f + 0.5f * sinf(theta), 0.5f);
}

// GPU kernel: Update particles with physics
__global__ void updateParticlesKernel(Particle3D* particles, int count,
                                       float dt, float time, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Particle3D& p = particles[idx];
    if (p.life <= 0) return;

    unsigned int rng = seed + idx * 7919 + (unsigned int)(time * 1000);

    // Apply acceleration
    p.velocity = vadd(p.velocity, vmul(p.acceleration, dt));

    // Add gravity
    p.velocity.y -= 2.0f * dt;

    // Add some turbulence
    float noise = sinf(time * 2.0f + p.position.x) * sinf(time * 3.0f + p.position.z);
    p.velocity.x += noise * 0.1f * dt;
    p.velocity.z += noise * 0.1f * dt;

    // Update position
    p.position = vadd(p.position, vmul(p.velocity, dt));

    // Dampening
    p.velocity = vmul(p.velocity, 0.99f);

    // Floor collision
    if (p.position.y < -1.0f) {
        p.position.y = -1.0f;
        p.velocity.y *= -0.6f;
    }

    // Update life
    p.life -= dt * 0.3f;

    // Reset acceleration
    p.acceleration = make_float3(0, 0, 0);
}

// GPU kernel: Calculate speed squared for reduction
__global__ void calcSpeedSquaredKernel(const Particle3D* particles, float* speed2, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float3 v = particles[idx].velocity;
    speed2[idx] = dot(v, v);
}

// GPU kernel: Calculate bounding box extents (flat arrays)
__global__ void calcBoundingBoxKernel(const Particle3D* particles,
                                       float* mins_x, float* mins_y, float* mins_z,
                                       float* maxs_x, float* maxs_y, float* maxs_z, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    mins_x[idx] = particles[idx].position.x;
    mins_y[idx] = particles[idx].position.y;
    mins_z[idx] = particles[idx].position.z;
    maxs_x[idx] = particles[idx].position.x;
    maxs_y[idx] = particles[idx].position.y;
    maxs_z[idx] = particles[idx].position.z;
}

// GPU kernel: Create predicate for active particles above plane
__global__ void createPlanePredicate(const Particle3D* particles, int* predicate,
                                       float planeY, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    predicate[idx] = (particles[idx].position.y > planeY && particles[idx].life > 0) ? 1 : 0;
}

// GPU kernel: Render to viewport projection (simple orthographic)
__global__ void renderViewportKernel(const Particle3D* particles, int count,
                                      float* depthBuffer, int* colorBuffer,
                                      int width, int height, float3 cameraDir) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    depthBuffer[idx] = 1.0f;  // Far plane
    colorBuffer[idx] = 0;

    // Project all particles (simplified)
    for (int i = 0; i < count; i++) {
        if (particles[i].life <= 0) continue;

        // Simple orthographic projection from front
        float projX = (particles[i].position.x + 1.5f) / 3.0f * width;
        float projY = (particles[i].position.y + 1.0f) / 2.0f * height;

        int pixX = (int)projX;
        int pixY = (int)projY;

        if (pixX >= 0 && pixX < width && pixY >= 0 && pixY < height) {
            int pixIdx = pixY * width + pixX;
            float depth = particles[i].position.z;

            if (depth < depthBuffer[pixIdx]) {
                depthBuffer[pixIdx] = depth;
                // Pack color: R in low bits, G in middle, B in high
                int r = (int)(particles[i].color.x * 255);
                int g = (int)(particles[i].color.y * 255);
                int b = (int)(particles[i].color.z * 255);
                colorBuffer[pixIdx] = (b << 16) | (g << 8) | r;
            }
        }
    }
}

// Export particles to OBJ format
void exportOBJ(const std::string& filename, const Particle3D* particles, int count) {
    std::ofstream file(filename);
    if (!file.is_open()) return;

    file << "# 3D Particle Export (CUDA Demo)" << std::endl;
    file << "# Vertices: " << count << std::endl << std::endl;

    // Export as point cloud (vertices only)
    for (int i = 0; i < count; i++) {
        const auto& p = particles[i];
        if (p.life <= 0) continue;
        file << "v " << p.position.x << " " << p.position.y << " " << p.position.z << std::endl;
    }

    // Add camera info as comment
    file << std::endl << "# End of particle cloud" << std::endl;
    file.close();
}

// Export particles to binary format for further processing
void exportBinary(const std::string& filename, const Particle3D* particles, int count) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return;

    file.write(reinterpret_cast<const char*>(&count), sizeof(int));

    for (int i = 0; i < count; i++) {
        const auto& p = particles[i];
        file.write(reinterpret_cast<const char*>(&p.position), sizeof(float3));
        file.write(reinterpret_cast<const char*>(&p.velocity), sizeof(float3));
        file.write(reinterpret_cast<const char*>(&p.color), sizeof(float3));
        file.write(reinterpret_cast<const char*>(&p.life), sizeof(float));
        file.write(reinterpret_cast<const char*>(&p.size), sizeof(float));
    }

    file.close();
}

// Print 3D info as ASCII art with depth
void printViewport(int* colorBuffer, float* depthBuffer, int width, int height) {
    const char* density = " .:-=+*#%@";

    std::cout << "+" << std::string(width, '-') << "+" << std::endl;

    for (int y = 0; y < height; y++) {
        std::cout << "|";
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int color = colorBuffer[idx];
            int r = color & 255;
            int g = (color >> 8) & 255;
            int b = (color >> 16) & 255;

            // Use brightness for ASCII
            int brightness = (r + g + b) / 3;
            char c = density[brightness / 26];
            std::cout << c;
        }
        std::cout << "|" << std::endl;
    }

    std::cout << "+" << std::string(width, '-') << "+" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   3D Particle System Demo (CUDA)       " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Configuration
    const int num_particles = 10000;
    const int width = 60;
    const int height = 25;
    const int max_frames = 50;
    OutputFormat outputFormat = OutputFormat::BINARY;

    // Check CUDA
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "CUDA devices: " << deviceCount << std::endl;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    std::cout << std::endl;

    // Allocate memory
    Particle3D* d_particles;
    float* d_speed2;
    int* d_predicate;
    float* d_depthBuffer;
    int* d_colorBuffer;

    CUDA_CHECK(cudaMalloc(&d_particles, num_particles * sizeof(Particle3D)));
    CUDA_CHECK(cudaMalloc(&d_speed2, num_particles * sizeof(float)));
    // Use flat arrays for bounding box components (float3 reduction not supported)
    float* d_mins_x;
    float* d_mins_y;
    float* d_mins_z;
    float* d_maxs_x;
    float* d_maxs_y;
    float* d_maxs_z;
    CUDA_CHECK(cudaMalloc(&d_mins_x, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mins_y, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mins_z, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxs_x, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxs_y, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxs_z, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predicate, num_particles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_depthBuffer, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_colorBuffer, width * height * sizeof(int)));

    std::vector<float> h_depthBuffer(width * height);
    std::vector<int> h_colorBuffer(width * height);

    // Initialize particles
    dim3 block(256);
    dim3 grid((num_particles + 255) / 256);
    initSphereKernel<<<grid, block>>>(d_particles, num_particles, 12345);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Initialized " << num_particles << " particles in sphere formation" << std::endl;
    std::cout << std::endl;

    // Create profiler
    auto& profiler = MyEngine::CUDAProfiler::instance();

    // Create output directory
    std::string outputDir = "/tmp/cuda_demo_output";
    std::filesystem::create_directories(outputDir);

    // Main loop
    float dt = 0.016f;
    unsigned int frame = 0;

    std::cout << "Running for " << max_frames << " frames..." << std::endl;
    std::cout << "Output will be saved to: " << outputDir << std::endl << std::endl;

    while (frame < max_frames) {
        float time = frame * dt;

        // Update particles
        {
            auto scope = profiler.profile("update");
            updateParticlesKernel<<<grid, block>>>(d_particles, num_particles, dt, time, frame);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate statistics using reduction
        float totalEnergy = 0.0f;
        float maxSpeed = 0.0f;
        float minSpeed = 1000.0f;

        {
            auto scope = profiler.profile("speed_calc");
            calcSpeedSquaredKernel<<<grid, block>>>(d_particles, d_speed2, num_particles);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        {
            auto scope = profiler.profile("reduce_total_energy");
            MyEngine::CUDAReduction<float> reducer(MyEngine::ReductionOp::Sum, num_particles);
            auto result = reducer.reduce(d_speed2, num_particles);
            totalEnergy = result.value * 0.5f;  // E = 0.5 * m * v^2, assume m=1
        }

        {
            auto scope = profiler.profile("reduce_max_speed");
            MyEngine::CUDAReduction<float> reducer(MyEngine::ReductionOp::Max, num_particles);
            auto result = reducer.reduce(d_speed2, num_particles);
            maxSpeed = sqrtf(result.value);
        }

        {
            auto scope = profiler.profile("reduce_min_speed");
            MyEngine::CUDAReduction<float> reducer(MyEngine::ReductionOp::Min, num_particles);
            auto result = reducer.reduce(d_speed2, num_particles);
            minSpeed = sqrtf(result.value);
        }

        // Calculate bounding box using component-wise reduction
        {
            auto scope = profiler.profile("bbox_calc");
            calcBoundingBoxKernel<<<grid, block>>>(d_particles,
                                                     d_mins_x, d_mins_y, d_mins_z,
                                                     d_maxs_x, d_maxs_y, d_maxs_z,
                                                     num_particles);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Reduce each component separately
            MyEngine::CUDAReduction<float> minReducer(MyEngine::ReductionOp::Min, num_particles);
            MyEngine::CUDAReduction<float> maxReducer(MyEngine::ReductionOp::Max, num_particles);

            auto minX = minReducer.reduce(d_mins_x, num_particles);
            auto minY = minReducer.reduce(d_mins_y, num_particles);
            auto minZ = minReducer.reduce(d_mins_z, num_particles);
            auto maxX = maxReducer.reduce(d_maxs_x, num_particles);
            auto maxY = maxReducer.reduce(d_maxs_y, num_particles);
            auto maxZ = maxReducer.reduce(d_maxs_z, num_particles);

            // Print frame info every 5 frames
            if (frame % 5 == 0) {
                float3 bboxMin = make_float3(minX.value, minY.value, minZ.value);
                float3 bboxMax = make_float3(maxX.value, maxY.value, maxZ.value);
                float3 center = vadd(bboxMin, bboxMax);
                center = vmul(center, 0.5f);

                std::cout << "Frame " << std::setw(3) << frame
                          << " | Energy: " << std::fixed << std::setprecision(1) << totalEnergy
                          << " | Speed: [" << std::setprecision(2) << minSpeed << ", " << maxSpeed << "]"
                          << " | Center: (" << center.x << ", " << center.y << ", " << center.z << ")"
                          << std::endl;
            }
        }

        // Use scan for spatial filtering (particles above y=0 plane)
        {
            auto scope = profiler.profile("spatial_filter");
            createPlanePredicate<<<grid, block>>>(d_particles, d_predicate, 0.0f, num_particles);
            CUDA_CHECK(cudaDeviceSynchronize());

            auto scan = MyEngine::CUDAScan::create<int>(num_particles);
            auto scan_result = scan.scanInclusive(d_predicate, num_particles);

            // Count filtered particles
            int filteredCount = 0;
            if (scan_result.ok() && num_particles > 0) {
                cudaMemcpy(&filteredCount, scan_result.d_output + num_particles - 1, sizeof(int), cudaMemcpyDeviceToHost);
            }

            if (frame % 10 == 0) {
                std::cout << "  -> Particles above Y=0: " << filteredCount << std::endl;
            }
        }

        // Render viewport
        {
            dim3 renderBlock(8, 8);
            dim3 renderGrid((width + 7) / 8, (height + 7) / 8);
            float3 cameraDir = make_float3(0, 0, -1);

            renderViewportKernel<<<renderGrid, renderBlock>>>(d_particles, num_particles,
                                                               d_depthBuffer, d_colorBuffer,
                                                               width, height, cameraDir);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Print ASCII viewport every 10 frames
        if (frame % 10 == 0) {
            cudaMemcpy(h_depthBuffer.data(), d_depthBuffer, width * height * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_colorBuffer.data(), d_colorBuffer, width * height * sizeof(int), cudaMemcpyDeviceToHost);
            printViewport(h_colorBuffer.data(), h_depthBuffer.data(), width, height);
        }

        // Export frame every 20 frames
        if (frame % 20 == 0 && frame > 0) {
            std::string filename = outputDir + "/particles_" + std::to_string(frame) + ".bin";

            // Copy particles to host for export
            std::vector<Particle3D> h_particles(num_particles);
            cudaMemcpy(h_particles.data(), d_particles, num_particles * sizeof(Particle3D), cudaMemcpyDeviceToHost);

            exportBinary(filename, h_particles.data(), num_particles);
            std::cout << "  Exported to: " << filename << std::endl;
        }

        frame++;
    }

    // Export final frame
    std::cout << std::endl << "Exporting final frame..." << std::endl;
    {
        std::vector<Particle3D> h_particles(num_particles);
        cudaMemcpy(h_particles.data(), d_particles, num_particles * sizeof(Particle3D), cudaMemcpyDeviceToHost);

        exportOBJ(outputDir + "/particles_final.obj", h_particles.data(), num_particles);
        exportBinary(outputDir + "/particles_final.bin", h_particles.data(), num_particles);
    }

    // Print profiler stats
    std::cout << std::endl << "=== Profiler Statistics ===" << std::endl;
    auto stats = profiler.getStats();
    for (const auto& stat : stats) {
        std::cout << "  " << stat.toString() << std::endl;
    }

    // Export profiler stats
    profiler.exportStatsToCSV(outputDir + "/profiler_stats.csv");

    // Cleanup
    cudaFree(d_particles);
    cudaFree(d_speed2);
    cudaFree(d_mins_x);
    cudaFree(d_mins_y);
    cudaFree(d_mins_z);
    cudaFree(d_maxs_x);
    cudaFree(d_maxs_y);
    cudaFree(d_maxs_z);
    cudaFree(d_predicate);
    cudaFree(d_depthBuffer);
    cudaFree(d_colorBuffer);

    std::cout << std::endl << "Demo complete!" << std::endl;
    std::cout << "Output files: " << outputDir << std::endl;
    return 0;
}
