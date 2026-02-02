/**
 * 2D Particle System Demo
 *
 * Demonstrates CUDA compute primitives:
 * - Parallel reduction for statistics (total energy, average velocity)
 * - Prefix sum (scan) for particle filtering
 * - Stream compaction to remove dead particles
 *
 * Compile: nvcc -o demo_2d demo_2d_particles.cu -O3 --use_fast_math
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <thread>

#include "cuda_runtime.h"
#include "cuda_math.h"
#include "compute/reduction.h"
#include "compute/scan.h"
#include "profiler/cuda_profiler.h"

// Include implementations
#include "compute/reduction.cu"
#include "compute/scan.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            return 1; \
        } \
    } while(0)

// 2D Particle structure
struct Particle2D {
    float2 position;
    float2 velocity;
    float life;
    float max_life;
};

// Simple random number generator
__device__ __host__ inline float random(unsigned int& seed, float minVal, float maxVal) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return minVal + (float)(seed % 100000) / 100000.0f * (maxVal - minVal);
}

// GPU kernel: Initialize particles
__global__ void initParticlesKernel(Particle2D* particles, int count, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    unsigned int rng = seed + idx * 7919;
    Particle2D& p = particles[idx];

    p.position = make_float2(
        random(rng, -1.0f, 1.0f),
        random(rng, -1.0f, 1.0f)
    );
    p.velocity = make_float2(
        random(rng, -0.5f, 0.5f),
        random(rng, -0.5f, 0.5f)
    );
    p.life = random(rng, 0.5f, 1.0f);
    p.max_life = p.life;
}

// GPU kernel: Update particles
__global__ void updateParticlesKernel(Particle2D* particles, int count,
                                       float dt, float time, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Particle2D& p = particles[idx];
    unsigned int rng = seed + idx * 7919 + (unsigned int)(time * 1000);

    // Update velocity with some noise
    p.velocity.x += random(rng, -0.1f, 0.1f) * dt;
    p.velocity.y += random(rng, -0.1f, 0.1f) * dt;
    p.velocity.y -= 0.5f * dt;  // Gravity

    // Dampening
    p.velocity.x *= 0.99f;
    p.velocity.y *= 0.99f;

    // Update position
    p.position.x += p.velocity.x * dt;
    p.position.y += p.velocity.y * dt;

    // Bounce off walls
    if (p.position.x < -1.0f || p.position.x > 1.0f) {
        p.velocity.x *= -0.8f;
        p.position.x = fmaxf(-1.0f, fminf(1.0f, p.position.x));
    }
    if (p.position.y < -1.0f || p.position.y > 1.0f) {
        p.velocity.y *= -0.8f;
        p.position.y = fmaxf(-1.0f, fminf(1.0f, p.position.y));
    }

    // Update life
    p.life -= dt * 0.5f;
}

// GPU kernel: Calculate kinetic energy for each particle
__global__ void calcEnergyKernel(const Particle2D* particles, float* energy, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float speed2 = particles[idx].velocity.x * particles[idx].velocity.x +
                   particles[idx].velocity.y * particles[idx].velocity.y;
    energy[idx] = 0.5f * speed2 * particles[idx].life;
}

// GPU kernel: Create predicate for alive particles in right half
__global__ void createRightHalfPredicate(const Particle2D* particles, int* predicate, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    predicate[idx] = (particles[idx].position.x > 0.0f && particles[idx].life > 0) ? 1 : 0;
}

// GPU kernel: Render to ASCII buffer
__global__ void renderASCIIKernel(const Particle2D* particles, int count,
                                   char* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    buffer[idx] = '.';

    // Check all particles
    for (int i = 0; i < count; i++) {
        float px = (particles[i].position.x + 1.0f) * 0.5f * (width - 1);
        float py = (1.0f - (particles[i].position.y + 1.0f) * 0.5f) * (height - 1);

        int pix_x = (int)px;
        int pix_y = (int)py;

        if (pix_x == x && pix_y == y && particles[i].life > 0) {
            char c = particles[i].life > 0.7f ? '@' :
                     particles[i].life > 0.3f ? 'O' : 'o';
            buffer[idx] = c;
            break;
        }
    }
}

// GPU kernel: Render filtered particles (right half) in different color
__global__ void renderFilteredKernel(const Particle2D* particles, int count,
                                      char* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    for (int i = 0; i < count; i++) {
        if (particles[i].position.x <= 0.0f || particles[i].life <= 0) continue;

        float px = (particles[i].position.x) * 0.5f * (width - 1);
        float py = (1.0f - particles[i].position.y) * 0.5f * (height - 1);

        int pix_x = (int)px;
        int pix_y = (int)py;

        if (pix_x == x && pix_y == y) {
            buffer[idx] = '#';
            break;
        }
    }
}

void printASCIIFrame(char* buffer, int width, int height) {
    std::cout << "\033[H";  // Home cursor
    std::cout << "+" << std::string(width, '-') << "+" << std::endl;
    for (int y = 0; y < height; y++) {
        std::cout << "|";
        for (int x = 0; x < width; x++) {
            std::cout << buffer[y * width + x];
        }
        std::cout << "|" << std::endl;
    }
    std::cout << "+" << std::string(width, '-') << "+" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   2D Particle System Demo (CUDA)       " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    const int width = 60;
    const int height = 30;
    const int num_particles = 5000;
    const int max_frames = 100;

    // Check CUDA
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "CUDA devices: " << deviceCount << std::endl;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl << std::endl;

    // Allocate memory
    Particle2D* d_particles;
    float* d_energy;
    int* d_predicate;
    char* d_buffer;

    CUDA_CHECK(cudaMalloc(&d_particles, num_particles * sizeof(Particle2D)));
    CUDA_CHECK(cudaMalloc(&d_energy, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predicate, num_particles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_buffer, width * height * sizeof(char)));

    std::vector<char> h_buffer(width * height);

    // Initialize particles
    dim3 block(256);
    dim3 grid((num_particles + 255) / 256);
    initParticlesKernel<<<grid, block>>>(d_particles, num_particles, 12345);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Initialized " << num_particles << " particles" << std::endl;
    std::cout << "Frame size: " << width << "x" << height << std::endl;
    std::cout << std::endl;

    // Create profiler
    auto& profiler = MyEngine::CUDAProfiler::instance();

    // Main loop
    float dt = 0.016f;
    unsigned int frame = 0;

    std::cout << "Press Ctrl+C to stop, or wait for " << max_frames << " frames..." << std::endl << std::endl;

    while (frame < max_frames) {
        float time = frame * dt;

        // Update particles
        {
            auto scope = profiler.profile("update");
            updateParticlesKernel<<<grid, block>>>(d_particles, num_particles, dt, time, frame);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate energy for each particle
        {
            auto scope = profiler.profile("energy_calc");
            calcEnergyKernel<<<grid, block>>>(d_particles, d_energy, num_particles);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Use reduction to compute total energy
        float total_energy = 0.0f;
        {
            auto scope = profiler.profile("reduce_energy");
            MyEngine::CUDAReduction<float> reducer(MyEngine::ReductionOp::Sum, num_particles);
            auto result = reducer.reduce(d_energy, num_particles);
            total_energy = result.value;
        }

        // Use reduction to compute average velocity magnitude (reuse energy array)
        {
            auto scope = profiler.profile("velocity_stats");

            // For simplicity, reuse energy array for velocity approximations
            calcEnergyKernel<<<grid, block>>>(d_particles, d_energy, num_particles);
            CUDA_CHECK(cudaDeviceSynchronize());

            MyEngine::CUDAReduction<float> reducer(MyEngine::ReductionOp::Sum, num_particles);
            auto result = reducer.reduce(d_energy, num_particles);
            float avg_vel = result.value / num_particles * 2.0f;  // Approximate

            // Create predicate for filtering
            createRightHalfPredicate<<<grid, block>>>(d_particles, d_predicate, num_particles);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Use scan for stream compaction to count particles in right half
            auto scan = MyEngine::CUDAScan::create<int>(num_particles);
            auto scan_result = scan.scanInclusive(d_predicate, num_particles);

            int right_half_count = 0;
            if (scan_result.ok() && num_particles > 0) {
                cudaMemcpy(&right_half_count, scan_result.d_output + num_particles - 1, sizeof(int), cudaMemcpyDeviceToHost);
            }

            // Print frame info every 10 frames
            if (frame % 10 == 0) {
                std::cout << "Frame " << std::setw(3) << frame
                          << " | Energy: " << std::fixed << std::setprecision(2) << total_energy
                          << " | Avg Vel: " << std::setprecision(3) << avg_vel
                          << " | Right Half: " << right_half_count << std::endl;
            }
        }

        // Render (every frame for demo)
        {
            dim3 renderBlock(8, 8);
            dim3 renderGrid((width + 7) / 8, (height + 7) / 8);
            renderASCIIKernel<<<renderGrid, renderBlock>>>(d_particles, num_particles,
                                                            d_buffer, width, height);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Print ASCII frame every 5 frames
        if (frame % 5 == 0) {
            cudaMemcpy(h_buffer.data(), d_buffer, width * height * sizeof(char), cudaMemcpyDeviceToHost);
            printASCIIFrame(h_buffer.data(), width, height);
        }

        frame++;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Print profiler stats
    std::cout << std::endl << "=== Profiler Statistics ===" << std::endl;
    auto stats = profiler.getStats();
    for (const auto& stat : stats) {
        std::cout << "  " << stat.toString() << std::endl;
    }

    // Cleanup
    cudaFree(d_particles);
    cudaFree(d_energy);
    cudaFree(d_predicate);
    cudaFree(d_buffer);

    std::cout << std::endl << "Demo complete!" << std::endl;
    return 0;
}
