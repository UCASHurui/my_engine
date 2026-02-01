#include "particle_system.h"

namespace MyEngine {

// GPU kernel: Update particles
__global__ void updateParticlesKernel(
    ParticleSystem system,
    float dt,
    float time
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= system.active_count) return;

    Particle& p = system.particles[idx];
    unsigned int rng = system.seed + idx * 7919 + (unsigned int)(time * 1000.0f);

    // Apply forces
    p.velocity = vadd(p.velocity, vmul(p.acceleration, dt));
    p.velocity = vadd(p.velocity, vmul(system.physics.gravity, dt));
    p.velocity = vmul(p.velocity, system.physics.drag);

    // Add wind and turbulence
    float noise = sinf(time * system.physics.turbulence_scale + p.position.x * 0.5f) *
                  sinf(time * system.physics.turbulence_scale + p.position.y * 0.5f);
    p.velocity = vadd(p.velocity, vmul(system.physics.wind, dt));
    p.velocity = vadd(p.velocity, make_float3(
        noise * system.physics.turbulence * dt,
        noise * system.physics.turbulence * 0.5f * dt,
        noise * system.physics.turbulence * dt
    ));

    // Update position
    p.position = vadd(p.position, vmul(p.velocity, dt));

    // Update life
    p.life -= dt / p.max_life;

    // Reset acceleration
    p.acceleration = make_float3(0, 0, 0);
}

// GPU kernel: Emit particles
__global__ void emitParticlesKernel(
    ParticleSystem system,
    int count,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    if (system.active_count + idx >= system.max_count) return;

    unsigned int rng = seed + idx * 7919;

    // Random direction in cone
    float rand1 = (float)(rng % 10000) / 10000.0f;
    float rand2 = (float)((rng >> 16) % 10000) / 10000.0f;
    float rand3 = (float)((rng >> 8) % 10000) / 10000.0f;
    float rand4 = (float)((rng >> 24) % 10000) / 10000.0f;

    float3 dir = system.emitter.direction;
    float spread = system.emitter.spread;

    // Random offset from cone axis
    float theta = rand1 * 2.0f * 3.14159265f;
    float phi = rand2 * spread - spread * 0.5f;

    float3 offset = make_float3(
        cosf(theta) * sinf(phi),
        cosf(phi),
        sinf(theta) * sinf(phi)
    );

    // Orthonormal basis
    float3 up = make_float3(0, 1, 0);
    if (fabsf(dir.y) > 0.99f) up = make_float3(1, 0, 0);
    float3 right = normalize(cross(up, dir));
    float3 forward = cross(right, dir);

    float3 final_dir = vadd(vadd(vmul(right, offset.x), vmul(forward, offset.z)), vmul(dir, offset.y + 1.0f));
    final_dir = normalize(final_dir);

    float speed = system.emitter.speed_min + rand3 * (system.emitter.speed_max - system.emitter.speed_min);
    float life = system.emitter.life_min + rand4 * (system.emitter.life_max - system.emitter.life_min);

    int particle_idx = system.active_count + idx;
    Particle& p = system.particles[particle_idx];

    p.position = system.emitter.position;
    p.velocity = vmul(final_dir, speed);
    p.acceleration = make_float3(0, 0, 0);
    p.life = life;
    p.max_life = life;
    p.size = system.emitter.size;
    p.color = system.emitter.color;
}

// Particle system manager
ParticleSystemManager::ParticleSystemManager() : _stream(0) {
    _system.particles = nullptr;
    _system.max_count = 0;
    _system.active_count = 0;
}

ParticleSystemManager::~ParticleSystemManager() {
    destroy();
}

bool ParticleSystemManager::create(int max_particles) {
    destroy();

    cudaError_t err = cudaMallocManaged(&_system.particles, max_particles * sizeof(Particle));
    if (err != cudaSuccess) return false;

    _system.max_count = max_particles;
    _system.active_count = 0;

    err = cudaStreamCreate(&_stream);
    if (err != cudaSuccess) {
        cudaFree(_system.particles);
        return false;
    }

    return true;
}

void ParticleSystemManager::emit(float dt, int count) {
    int emit_count = min(count, _system.max_count - _system.active_count);
    if (emit_count <= 0) return;

    dim3 block(256);
    dim3 grid((emit_count + 255) / 256);

    emitParticlesKernel<<<grid, block, 0, _stream>>>(_system, emit_count, _system.seed);

    // Wait for kernel to complete before updating counters
    cudaStreamSynchronize(_stream);

    _system.active_count += emit_count;
    _system.seed += emit_count * 7919;
}

void ParticleSystemManager::update(float dt) {
    if (_system.active_count == 0) return;

    dim3 block(256);
    dim3 grid((_system.active_count + 255) / 256);

    float time = (float)(clock() % 1000000) / 1000.0f;
    updateParticlesKernel<<<grid, block, 0, _stream>>>(_system, dt, time);

    // Wait for kernel to complete before accessing particles on host
    cudaStreamSynchronize(_stream);

    // Remove dead particles (compact)
    int write_idx = 0;
    for (int i = 0; i < _system.active_count; i++) {
        if (_system.particles[i].life > 0) {
            if (write_idx != i) {
                _system.particles[write_idx] = _system.particles[i];
            }
            write_idx++;
        }
    }
    _system.active_count = write_idx;
}

void ParticleSystemManager::destroy() {
    if (_stream) {
        cudaStreamDestroy(_stream);
        _stream = 0;
    }
    if (_system.particles) {
        cudaFree(_system.particles);
        _system.particles = nullptr;
    }
    _system.max_count = 0;
    _system.active_count = 0;
}

} // namespace MyEngine
