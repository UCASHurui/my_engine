/**
 * N-Body Gravity Simulation Demo
 *
 * Features:
 * - N-body gravitational simulation (O(n^2) on GPU)
 * - Real-time statistics and profiling
 * - Multiple visualization modes
 * - Interactive controls (simulated via input file or network)
 * - Particle trails and glow effects
 *
 * Compile: nvcc -o demo_nbody demo_nbody.cu -O3 --use_fast_math -arch=sm_89
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
#include <atomic>
#include <mutex>
#include <queue>

#include "cuda_runtime.h"
#include "cuda_math.h"
#include "compute/reduction.h"
#include "compute/scan.h"

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

using namespace MyEngine;

// Simulation constants
const int NUM_BODIES = 5000;
const float G = 0.1f;              // Gravitational constant
const float SOFTENING = 0.1f;      // Prevent singularities
const float DT = 0.016f;           // Time step
const float BODY_MASS = 1.0f;
const float MAX_MASS = 100.0f;

// Body structure
struct Body {
    float3 position;
    float3 velocity;
    float3 acceleration;
    float mass;
    float3 color;
    float trail;  // Trail intensity
};

// Input state for interaction
struct InputState {
    std::atomic<bool> paused{false};
    std::atomic<bool> reset{false};
    std::atomic<bool> addExplosion{false};
    std::atomic<bool> toggleTrails{true};
    std::atomic<int> viewMode{0};  // 0: normal, 1: velocity, 2: mass, 3: acceleration
    std::atomic<float> timeScale{1.0f};
    std::atomic<int> explodeX{0};
    std::atomic<int> explodeY{0};
    std::mutex inputMutex;
    std::queue<std::string> commandQueue;
};

// HSV to RGB conversion (must be before kernels that use it)
__device__ __host__ inline float3 hsvToRgb(float h, float s, float v) {
    float3 rgb = make_float3(v, v, v);
    if (s > 0) {
        h = fmodf(fmodf(h, 1.0f), 1.0f);
        if (h < 0) h += 1;
        int i = (int)(h * 6);
        float f = h * 6 - i;
        float p = v * (1 - s);
        float q = v * (1 - s * f);
        float t = v * (1 - s * (1 - f));
        switch (i) {
            case 0: rgb = make_float3(v, t, p); break;
            case 1: rgb = make_float3(q, v, p); break;
            case 2: rgb = make_float3(p, v, t); break;
            case 3: rgb = make_float3(p, q, v); break;
            case 4: rgb = make_float3(t, p, v); break;
            case 5: rgb = make_float3(v, p, q); break;
        }
    }
    return rgb;
}

// Read input commands from file/stdin
void inputReader(InputState& input) {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::lock_guard<std::mutex> lock(input.inputMutex);
        input.commandQueue.push(line);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// Process input commands
void processInput(InputState& input) {
    std::lock_guard<std::mutex> lock(input.inputMutex);
    while (!input.commandQueue.empty()) {
        std::string cmd = input.commandQueue.front();
        input.commandQueue.pop();

        if (cmd == "space" || cmd == "p") {
            input.paused = !input.paused;
            std::cout << "[Input] " << (input.paused ? "PAUSED" : "RESUMED") << std::endl;
        } else if (cmd == "r" || cmd == "reset") {
            input.reset = true;
            std::cout << "[Input] RESET" << std::endl;
        } else if (cmd == "t" || cmd == "trails") {
            input.toggleTrails = !input.toggleTrails;
            std::cout << "[Input] Trails: " << (input.toggleTrails ? "ON" : "OFF") << std::endl;
        } else if (cmd == "q" || cmd == "quit") {
            exit(0);
        } else if (cmd.substr(0, 5) == "view ") {
            int mode = std::stoi(cmd.substr(5));
            input.viewMode = mode;
            std::cout << "[Input] View mode: " << mode << std::endl;
        } else if (cmd.substr(0, 4) == "tspd") {
            float ts = std::stof(cmd.substr(4));
            input.timeScale = ts;
            std::cout << "[Input] Time scale: " << ts << std::endl;
        } else if (cmd.substr(0, 4) == "boom") {
            input.addExplosion = true;
            std::cout << "[Input] EXPLOSION" << std::endl;
        } else if (cmd.substr(0, 4) == "goto") {
            // goto x y - move camera (simulated)
            std::cout << "[Input] Camera move" << std::endl;
        }
    }
}

// GPU kernel: Initialize galaxy spiral
__global__ void initSpiralKernel(Body* bodies, int count, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    unsigned int rng = seed + idx * 7919;

    // Galaxy spiral parameters
    float armCount = 2.0f;
    float armSpread = 0.5f;
    float radius = powf((float)idx / count, 0.5f) * 2.0f;
    float angle = idx * armCount * 2.0f * M_PI / count + radius * 2.0f;

    // Add some randomness
    float noise = ((float)(rng % 10000) / 10000.0f - 0.5f) * armSpread;

    float x = radius * cosf(angle + noise);
    float y = ((float)(rng % 10000) / 10000.0f - 0.5f) * 0.1f;
    float z = radius * sinf(angle + noise) * 0.3f;

    // Orbital velocity for spiral
    float velMag = sqrtf(G * 500.0f / (radius + 0.1f)) * 0.8f;
    float3 velDir = make_float3(-sinf(angle), 0, cosf(angle));

    Body& b = bodies[idx];
    b.position = make_float3(x, y, z);
    b.velocity = vmul(velDir, velMag);
    b.acceleration = make_float3(0, 0, 0);
    b.mass = BODY_MASS * (0.5f + (float)(rng % 10000) / 20000.0f);
    b.trail = 0.0f;

    // Color based on mass/distance
    float hue = radius * 0.3f;
    b.color = hsvToRgb(hue, 0.8f, 1.0f);
}

// GPU kernel: Compute gravitational forces (simplified O(n^2) with loop unrolling)
// For production, use shared memory or hierarchical algorithm
__global__ void computeForcesKernel(Body* bodies, int count, float softening) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    Body& body = bodies[tid];
    float3 my_pos = body.position;

    float3 acc = make_float3(0, 0, 0);

    // Simple O(n) force computation - in production use tree or tile-based
    // Process in chunks for better cache utilization
    const int chunkSize = 64;
    for (int chunk = 0; chunk < count; chunk += chunkSize) {
        int end = min(chunk + chunkSize, count);
        for (int j = chunk; j < end; j++) {
            if (j == tid) continue;

            float3 r = vsub(bodies[j].position, my_pos);
            float distSq = dot(r, r) + softening * softening;
            float dist = sqrtf(distSq);
            float f = G * bodies[j].mass / (distSq * dist);

            acc = vadd(acc, vmul(r, f));
        }
    }

    body.acceleration = acc;
}

// GPU kernel: Integrate position and velocity (symplectic Euler)
__global__ void integrateKernel(Body* bodies, int count, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Body& b = bodies[idx];

    // Update velocity
    b.velocity = vadd(b.velocity, vmul(b.acceleration, dt));

    // Update position
    b.position = vadd(b.position, vmul(b.velocity, dt));

    // Update trail
    b.trail = fminf(1.0f, b.trail * 0.99f);

    // Boundary wrap
    const float bound = 3.0f;
    if (b.position.x > bound) b.position.x -= 2*bound;
    if (b.position.x < -bound) b.position.x += 2*bound;
    if (b.position.y > bound) b.position.y -= 2*bound;
    if (b.position.y < -bound) b.position.y += 2*bound;
    if (b.position.z > bound) b.position.z -= 2*bound;
    if (b.position.z < -bound) b.position.z += 2*bound;
}

// GPU kernel: Apply explosion force
__global__ void explosionKernel(Body* bodies, int count, float3 center, float strength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Body& b = bodies[idx];
    float3 r = vsub(b.position, center);
    float dist = length(r) + 0.1f;
    float f = strength / (dist * dist);

    b.velocity = vadd(b.velocity, vmul(normalize(r), f));
    b.trail = 1.0f;
}

// GPU kernel: Calculate statistics
__global__ void calcStatsKernel(const Body* bodies, float* kinetic, float* potential,
                                  float* maxVel, float* maxAcc, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float3 v = bodies[idx].velocity;
    float3 a = bodies[idx].acceleration;

    kinetic[idx] = 0.5f * bodies[idx].mass * dot(v, v);

    // Potential energy (simplified)
    float3 r = bodies[idx].position;
    potential[idx] = -G * bodies[idx].mass * 500.0f / (length(r) + 0.1f);

    float velMag = length(v);
    float accMag = length(a);

    if (velMag > maxVel[0]) maxVel[0] = velMag;
    if (accMag > maxAcc[0]) maxAcc[0] = accMag;
}

// GPU kernel: Render to viewport with glow
__global__ void renderKernel(const Body* bodies, int count,
                              float* depthBuffer, int* colorBuffer,
                              int width, int height, int viewMode) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    depthBuffer[idx] = 1.0f;
    colorBuffer[idx] = 0;

    // Clear buffer
    for (int i = 0; i < count; i++) {
        const Body& b = bodies[i];
        if (b.trail < 0.1f && viewMode == 0) continue;

        // Project to screen
        float aspect = (float)width / height;
        float scale = 1.5f;

        float projX = (b.position.x / aspect + 1.0f) * 0.5f * width;
        float projY = (1.0f - (b.position.y + 1.0f) * 0.5f) * height;
        float projZ = b.position.z;

        int pixX = (int)projX;
        int pixY = (int)projY;

        if (pixX >= 0 && pixX < width && pixY >= 0 && pixY < height) {
            int pixIdx = pixY * width + pixX;
            float depth = projZ;

            if (depth < depthBuffer[pixIdx]) {
                depthBuffer[pixIdx] = depth;

                int r, g, blue;

                switch (viewMode) {
                    case 1:  // Velocity
                        {
                            float v = length(b.velocity);
                            float t = fminf(1.0f, v * 0.5f);
                            r = (int)(255 * t);
                            g = (int)(255 * (1 - t) * 0.5f);
                            blue = (int)(255 * (1 - t));
                        }
                        break;
                    case 2:  // Mass
                        {
                            float t = fminf(1.0f, (b.mass - BODY_MASS) / (MAX_MASS - BODY_MASS));
                            r = (int)(100 + 155 * t);
                            g = (int)(200 - 100 * t);
                            blue = (int)(255 - 150 * t);
                        }
                        break;
                    case 3:  // Acceleration
                        {
                            float a = length(b.acceleration);
                            float t = fminf(1.0f, a * 2.0f);
                            r = (int)(255 * t);
                            g = (int)(255 * (1 - t));
                            blue = 50;
                        }
                        break;
                    default:  // Normal (color based)
                        {
                            r = (int)(b.color.x * 255);
                            g = (int)(b.color.y * 255);
                            blue = (int)(b.color.z * 255);
                        }
                }

                // Add glow based on trail
                if (viewMode == 0 && b.trail > 0.1f) {
                    int glow = (int)(b.trail * 100);
                    r = fminf(255, r + glow);
                    g = fminf(255, g + glow);
                    blue = fminf(255, blue + glow);
                }

                colorBuffer[pixIdx] = (blue << 16) | (g << 8) | r;
            }
        }
    }
}

// Print viewport
void printViewport(int* colorBuffer, float* depthBuffer, int width, int height,
                   int viewMode, float kinetic, float potential, float maxVel) {
    const char* density = " .,-~:;=!*#$@";

    std::cout << "+" << std::string(width, '-') << "+" << std::endl;
    std::cout << "| Mode: ";
    const char* modeNames[] = {"Normal", "Velocity", "Mass", "Acceleration"};
    std::cout << modeNames[viewMode] << std::flush;
    std::cout << " | KE: " << std::fixed << std::setprecision(0) << kinetic
              << " | PE: " << potential << " | MaxV: " << std::setprecision(2) << maxVel << " |" << std::endl;
    std::cout << "+" << std::string(width, '-') << "+" << std::endl;

    for (int y = 0; y < height; y++) {
        std::cout << "|";
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int color = colorBuffer[idx];
            int brightness = ((color & 255) + ((color >> 8) & 255) + ((color >> 16) & 255)) / 3;
            char c = density[min(24, brightness / 11)];
            std::cout << c;
        }
        std::cout << "|" << std::endl;
    }
    std::cout << "+" << std::string(width, '-') << "+" << std::endl;
}

void printHelp() {
    std::cout << R"(
=== N-Body Gravity Simulation Controls ===
Commands:
  space/p    - Pause/Resume simulation
  r/reset    - Reset simulation
  t/trails   - Toggle particle trails
  view 0-3   - Change view mode (0:Normal, 1:Velocity, 2:Mass, 3:Acceleration)
  tspd N     - Set time scale (e.g., tspd 2.0)
  boom       - Create explosion at center
  q/quit     - Exit
  h/help     - Show this help

Controls are read from stdin. Pipe a file or use echo:
  echo "boom" | ./demo_nbody
  echo -e "boom\nspace\nq" | ./demo_nbody
)";
}

int main() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════╗
║         N-Body Gravitational Simulation (CUDA)               ║
║                                                          ║
║  Features:                                               ║
║    - O(n^2) gravitational force computation              ║
║    - Real-time statistics with CUDA profiling            ║
║    - Multiple visualization modes                        ║
║    - Interactive controls via stdin                      ║
║    - Particle trails and glow effects                    ║
╚══════════════════════════════════════════════════════════════╝
)" << std::endl;

    // Configuration
    const int width = 60;
    const int height = 30;
    const int maxFrames = 200;

    // Check CUDA
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "CUDA devices: " << deviceCount << std::endl;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Particles: " << NUM_BODIES << std::endl << std::endl;

    // Allocate memory
    Body* d_bodies;
    float* d_kinetic;
    float* d_potential;
    float* d_maxVel;
    float* d_maxAcc;
    float* d_depthBuffer;
    int* d_colorBuffer;

    CUDA_CHECK(cudaMalloc(&d_bodies, NUM_BODIES * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&d_kinetic, NUM_BODIES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_potential, NUM_BODIES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxVel, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxAcc, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depthBuffer, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_colorBuffer, width * height * sizeof(int)));

    std::vector<float> h_kinetic(NUM_BODIES);
    std::vector<float> h_potential(NUM_BODIES);
    std::vector<float> h_depthBuffer(width * height);
    std::vector<int> h_colorBuffer(width * height);

    // Initialize bodies
    dim3 block(256);
    dim3 grid((NUM_BODIES + 255) / 256);
    initSpiralKernel<<<grid, block>>>(d_bodies, NUM_BODIES, 12345);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Initialized " << NUM_BODIES << " bodies in spiral galaxy" << std::endl;

    // Input handling
    InputState input;
    std::thread inputThread(inputReader, std::ref(input));

    // Main loop
    float frame = 0;
    float totalKE = 0, totalPE = 0;
    float maxVelocity = 0;

    std::cout << std::endl << "Starting simulation..." << std::endl;
    printHelp();
    std::cout << std::endl << "Press 'h' + Enter for help during simulation." << std::endl;
    std::cout << "Note: Use 'echo \"command\" | ./demo_nbody' to send commands." << std::endl << std::endl;

    while (frame < maxFrames) {
        // Process input
        processInput(input);

        // Check reset
        if (input.reset.exchange(false)) {
            initSpiralKernel<<<grid, block>>>(d_bodies, NUM_BODIES, (unsigned int)frame);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cout << "[System] Simulation reset" << std::endl;
            continue;
        }

        // Check explosion
        if (input.addExplosion.exchange(false)) {
            dim3 expBlock(256);
            dim3 expGrid((NUM_BODIES + 255) / 256);
            explosionKernel<<<expGrid, expBlock>>>(d_bodies, NUM_BODIES,
                                                     make_float3(0, 0, 0), 50.0f);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        float dt = DT * input.timeScale.load();
        int viewMode = input.viewMode.load();
        bool showTrails = input.toggleTrails.load();

        if (!input.paused.load()) {
            // Compute forces
            computeForcesKernel<<<grid, block>>>(d_bodies, NUM_BODIES, SOFTENING);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Integrate
            integrateKernel<<<grid, block>>>(d_bodies, NUM_BODIES, dt);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate statistics
        {
            dim3 statBlock(256);
            dim3 statGrid((NUM_BODIES + 255) / 256);
            calcStatsKernel<<<statGrid, statBlock>>>(d_bodies, d_kinetic, d_potential,
                                                       d_maxVel, d_maxAcc, NUM_BODIES);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Reduce to get totals
            CUDAReduction<float> keReducer(ReductionOp::Sum, NUM_BODIES);
            CUDAReduction<float> peReducer(ReductionOp::Sum, NUM_BODIES);
            auto keResult = keReducer.reduce(d_kinetic, NUM_BODIES);
            auto peResult = peReducer.reduce(d_potential, NUM_BODIES);

            totalKE = keResult.value;
            totalPE = peResult.value;

            CUDA_CHECK(cudaMemcpy(&maxVelocity, d_maxVel, sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Render viewport
        {
            dim3 renderBlock(8, 8);
            dim3 renderGrid((width + 7) / 8, (height + 7) / 8);
            renderKernel<<<renderGrid, renderBlock>>>(d_bodies, NUM_BODIES,
                                                        d_depthBuffer, d_colorBuffer,
                                                        width, height, viewMode);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Copy to host and print
        cudaMemcpy(h_depthBuffer.data(), d_depthBuffer, width * height * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_colorBuffer.data(), d_colorBuffer, width * height * sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "\033[H";  // Home cursor
        printViewport(h_colorBuffer.data(), h_depthBuffer.data(), width, height,
                      viewMode, totalKE, totalPE, maxVelocity);

        // Every 20 frames, show profiling summary
        if ((int)frame % 20 == 0 && frame > 0) {
            std::cout << "--- Frame " << frame << " ---" << std::endl;
        }

        frame++;
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    // Cleanup
    inputThread.detach();
    cudaFree(d_bodies);
    cudaFree(d_kinetic);
    cudaFree(d_potential);
    cudaFree(d_maxVel);
    cudaFree(d_maxAcc);
    cudaFree(d_depthBuffer);
    cudaFree(d_colorBuffer);

    std::cout << std::endl << "Simulation complete!" << std::endl;
    return 0;
}
