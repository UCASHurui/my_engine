#pragma once

#include "math/Vector3.h"
#include "math/Color.h"
#include "object/RefCounted.h"
#include "math/Transform.h"

namespace MyEngine {

// 粒子发射模式
enum class ParticleEmitMode {
    ONCE,           // 一次性发射
    LOOP,           // 循环发射
    CONTINUOUS      // 持续发射
};

// 粒子分布模式
enum class ParticleDistribution {
    POINT,
    SPHERE,
    BOX,
    RING,
    LINE
};

// 粒子初始速度模式
enum class ParticleVelocityMode {
    CONSTANT,
    RANDOM,
    CURVE
};

// 粒子发射器配置
struct ParticleEmitterConfig {
    ParticleEmitMode emit_mode = ParticleEmitMode::CONTINUOUS;
    ParticleDistribution distribution = ParticleDistribution::SPHERE;

    // 发射数量
    int amount = 100;
    float lifetime = 1.0f;
    float lifetime_variance = 0.0f;
    float emission_rate = 10.0f;

    // 位置分布
    Vector3 distribution_extents = Vector3::ONE;
    float distribution_radius = 1.0f;

    // 初始速度
    ParticleVelocityMode velocity_mode = ParticleVelocityMode::RANDOM;
    Vector3 initial_velocity = Vector3(0, 1, 0);
    float initial_velocity_variance = 0.5f;
    Vector3 direction = Vector3(0, 1, 0);
    float spread = 15.0f;

    // 加速度
    Vector3 acceleration = Vector3::ZERO;
    float damping = 0.0f;

    // 缩放
    float scale_min = 0.1f;
    float scale_max = 0.5f;
    float scale_curve = 1.0f;

    // 颜色
    Color color_begin = Color::WHITE();
    Color color_end = Color(1, 1, 1, 0);
    float color_curve = 1.0f;

    // 旋转
    Vector3 angular_velocity_min = Vector3::ZERO;
    Vector3 angular_velocity_max = Vector3::ZERO;
    Vector3 initial_rotation = Vector3::ZERO;

    // 纹理
    RefCounted* texture;
};

// 粒子数据（单个粒子）
struct Particle {
    Vector3 position;
    Vector3 velocity;
    Vector3 scale;
    Color color;
    float lifetime;
    float age;
    Vector3 rotation;
    int seed;
    bool active;
};

// GPU 粒子系统
class GPUParticles3D : public RefCounted {
public:
    GPUParticles3D();
    ~GPUParticles3D() override;

    const char* get_class_name() const override { return "GPUParticles3D"; }

    // 发射配置
    void set_emitting(bool emitting) { _emitting = emitting; }
    bool is_emitting() const { return _emitting; }

    void set_amount(int amount) { _amount = amount; }
    int get_amount() const { return _amount; }

    void set_lifetime(float lifetime) { _lifetime = lifetime; }
    float get_lifetime() const { return _lifetime; }

    void set_one_shot(bool one_shot) { _one_shot = one_shot; }
    bool is_one_shot() const { return _one_shot; }

    void set_explosiveness(float explosiveness) { _explosiveness = explosiveness; }
    float get_explosiveness() const { return _explosiveness; }

    void set_pre_process(float pre_process) { _pre_process = pre_process; }
    float get_pre_process() const { return _pre_process; }

    // 变换
    void set_global_transform(const Transform3D& transform) { _global_transform = transform; }
    Transform3D get_global_transform() const { return _global_transform; }

    // 材质
    void set_material(RefCounted* material) { _material = material; }
    RefCounted* get_material() const { return _material; }

    // 粒子数据
    const std::vector<Particle>& get_particles() const { return _particles; }

    // 控制
    void restart();
    void emit_particle();

    // 更新
    void update(float delta);

private:
    bool _emitting = false;
    int _amount = 100;
    float _lifetime = 1.0f;
    bool _one_shot = false;
    float _explosiveness = 0.0f;
    float _pre_process = 0.0f;

    Transform3D _global_transform;
    RefCounted* _material = nullptr;

    std::vector<Particle> _particles;
    float _accumulator = 0.0f;
    float _time = 0.0f;
    int _active_count = 0;

    void _init_particles();
    void _emit_particle_at(int index);
    void _update_particle(Particle& particle, float delta);
    int _get_free_particle_index();
};

} // namespace MyEngine
