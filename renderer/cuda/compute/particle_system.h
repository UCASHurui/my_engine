#pragma once

#include "cuda_runtime.h"
#include "cuda_math.h"
#include <vector>

namespace MyEngine {

// Particle data structure
struct Particle {
    float3 position;
    float3 velocity;
    float3 acceleration;
    float life;
    float max_life;
    float size;
    float3 color;

    __host__ __device__ Particle()
        : position(make_float3(0, 0, 0))
        , velocity(make_float3(0, 0, 0))
        , acceleration(make_float3(0, 0, 0))
        , life(1.0f)
        , max_life(1.0f)
        , size(1.0f)
        , color(make_float3(1, 1, 1)) {}
};

// Emitter configuration
struct EmitterConfig {
    float3 position;
    float3 direction;
    float spread;
    float rate;
    float speed_min;
    float speed_max;
    float life_min;
    float life_max;
    float size;
    float3 color;

    __host__ __device__ EmitterConfig()
        : position(make_float3(0, 0, 0))
        , direction(make_float3(0, 1, 0))
        , spread(0.5f)
        , rate(100.0f)
        , speed_min(1.0f)
        , speed_max(5.0f)
        , life_min(1.0f)
        , life_max(3.0f)
        , size(0.1f)
        , color(make_float3(1, 1, 1)) {}
};

// Physics parameters
struct PhysicsConfig {
    float3 gravity;
    float drag;
    float restitution;
    float3 wind;
    float turbulence;
    float turbulence_scale;

    __host__ __device__ PhysicsConfig()
        : gravity(make_float3(0, -9.8f, 0))
        , drag(0.99f)
        , restitution(0.5f)
        , wind(make_float3(0, 0, 0))
        , turbulence(0.1f)
        , turbulence_scale(1.0f) {}
};

// Particle system state
struct ParticleSystem {
    Particle* particles;
    int max_count;
    int active_count;
    EmitterConfig emitter;
    PhysicsConfig physics;
    unsigned int seed;

    __host__ __device__ ParticleSystem()
        : particles(nullptr)
        , max_count(0)
        , active_count(0)
        , seed(12345) {}
};

// Particle system manager (host)
class ParticleSystemManager {
public:
    ParticleSystemManager();
    ~ParticleSystemManager();

    // Create particle system
    bool create(int max_particles);

    // Emit particles
    void emit(float dt, int count);

    // Update physics
    void update(float dt);

    // Get particle data
    const Particle* getParticles() const { return _system.particles; }
    int getActiveCount() const { return _system.active_count; }
    int getMaxCount() const { return _system.max_count; }

    // Set emitter config
    void setEmitter(const EmitterConfig& config) { _system.emitter = config; }

    // Set physics config
    void setPhysics(const PhysicsConfig& config) { _system.physics = config; }

    // Cleanup
    void destroy();

private:
    ParticleSystem _system;
    cudaStream_t _stream;
};

// GPU kernel: Update particles
__global__ void updateParticlesKernel(
    ParticleSystem system,
    float dt,
    float time
);

// GPU kernel: Emit particles
__global__ void emitParticlesKernel(
    ParticleSystem system,
    int count,
    unsigned int seed
);

// GPU kernel: Render particles to point buffer
__global__ void renderParticlesKernel(
    const Particle* particles,
    int count,
    float4* output,  // position + size
    float4* colors   // color + life
);

// Inline implementations
__device__ inline void updateParticle(Particle& p, const PhysicsConfig& phys,
                                        float dt, float time, unsigned int& rng) {
    // Simple RNG
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;

    // Apply forces
    p.velocity = vadd(p.velocity, vmul(p.acceleration, dt));
    p.velocity = vadd(p.velocity, vmul(phys.gravity, dt));

    // Apply drag
    p.velocity = vmul(p.velocity, phys.drag);

    // Add wind and turbulence
    float noise = sinf(time * phys.turbulence_scale + p.position.x * 0.5f) *
                  sinf(time * phys.turbulence_scale + p.position.y * 0.5f);
    p.velocity = vadd(p.velocity, vmul(phys.wind, dt));
    p.velocity = vadd(p.velocity, make_float3(
        noise * phys.turbulence * dt,
        noise * phys.turbulence * 0.5f * dt,
        noise * phys.turbulence * dt
    ));

    // Update position
    p.position = vadd(p.position, vmul(p.velocity, dt));

    // Update life
    p.life -= dt / p.max_life;

    // Reset acceleration
    p.acceleration = make_float3(0, 0, 0);
}

} // namespace MyEngine
