#include "GPUParticles3D.h"
#include "core/math/Math.h"

namespace MyEngine {

GPUParticles3D::GPUParticles3D() = default;
GPUParticles3D::~GPUParticles3D() = default;

void GPUParticles3D::restart() {
    _time = 0;
    _accumulator = 0;
    _active_count = 0;
    for (auto& p : _particles) {
        p.active = false;
        p.age = 0;
    }
}

void GPUParticles3D::emit_particle() {
    int index = _get_free_particle_index();
    if (index >= 0) {
        _emit_particle_at(index);
    }
}

void GPUParticles3D::update(float delta) {
    if (!_emitting) return;

    _time += delta;

    // 发射新粒子
    if (!_one_shot || _active_count < _amount) {
        float emission_rate = _amount / _lifetime;
        _accumulator += emission_rate * delta;

        while (_accumulator >= 1.0f && (_one_shot ? _active_count < _amount : true)) {
            _accumulator -= 1.0f;
            emit_particle();
        }
    }

    // 更新现有粒子
    for (auto& particle : _particles) {
        if (particle.active) {
            _update_particle(particle, delta);
            particle.age += delta;

            if (particle.age >= particle.lifetime) {
                particle.active = false;
                _active_count--;
            }
        }
    }
}

void GPUParticles3D::_init_particles() {
    _particles.resize(_amount);
    for (auto& p : _particles) {
        p = Particle();
        p.active = false;
        p.age = 0;
        p.position = Vector3::ZERO;
        p.velocity = Vector3::ZERO;
        p.scale = Vector3::ONE;
        p.color = Color::WHITE();
    }
}

void GPUParticles3D::_emit_particle_at(int index) {
    if (index < 0 || index >= (int)_particles.size()) return;

    Particle& p = _particles[index];
    p.active = true;
    p.age = 0;
    p.lifetime = _lifetime + Math::random_range(-_lifetime * 0.1f, _lifetime * 0.1f);
    p.position = Vector3::ZERO;

    // 随机速度方向
    float theta = Math::deg_to_rad(Math::random_range(0, 360));
    float phi = Math::deg_to_rad(Math::random_range(-45, 45));

    float speed = Math::random_range(1.0f, 5.0f);
    p.velocity.x = Math::cos(phi) * Math::cos(theta) * speed;
    p.velocity.y = Math::sin(phi) * speed + 1.0f;
    p.velocity.z = Math::cos(phi) * Math::sin(theta) * speed;

    // 随机缩放
    float scale = Math::random_range(0.1f, 0.5f);
    p.scale = Vector3(scale, scale, scale);

    // 随机颜色
    p.color = Color::WHITE();

    _active_count++;
}

void GPUParticles3D::_update_particle(Particle& particle, float delta) {
    // 应用加速度
    particle.velocity += Vector3(0, -9.8f, 0) * delta;

    // 应用阻尼
    particle.velocity *= (1.0f - 0.1f * delta);

    // 更新位置
    particle.position += particle.velocity * delta;

    // 更新缩放（淡出）
    float life_ratio = particle.age / particle.lifetime;
    float scale = Math::lerp(0.5f, 0.0f, life_ratio);
    particle.scale = Vector3(scale, scale, scale);

    // 更新颜色（淡出）
    particle.color.a() = 1.0f - life_ratio;
}

int GPUParticles3D::_get_free_particle_index() {
    for (int i = 0; i < (int)_particles.size(); i++) {
        if (!_particles[i].active) {
            return i;
        }
    }
    return -1;
}

} // namespace MyEngine
