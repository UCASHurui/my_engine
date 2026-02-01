#pragma once

#include "core/math/Vector3.h"
#include "core/math/Transform.h"
#include "core/object/RefCounted.h"

namespace MyEngine {

// 碰撞层和遮罩
using PhysicsCollisionLayer = uint32_t;
using PhysicsCollisionMask = uint32_t;

// 物理材质
class PhysicsMaterial : public RefCounted {
public:
    PhysicsMaterial();
    ~PhysicsMaterial() override;

    const char* get_class_name() const override { return "PhysicsMaterial"; }

    void set_friction(float friction) { _friction = friction; }
    float get_friction() const { return _friction; }

    void set_bounce(float bounce) { _bounce = bounce; }
    float get_bounce() const { return _bounce; }

    void set_bounce_threshold(float threshold) { _bounce_threshold = threshold; }
    float get_bounce_threshold() const { return _bounce_threshold; }

    void set_absorbent(bool absorbent) { _absorbent = absorbent; }
    bool is_absorbent() const { return _absorbent; }

    void set_linear_damp(float damp) { _linear_damp = damp; }
    float get_linear_damp() const { return _linear_damp; }

    void set_angular_damp(float damp) { _angular_damp = damp; }
    float get_angular_damp() const { return _angular_damp; }

private:
    float _friction = 0.5f;
    float _bounce = 0.0f;
    float _bounce_threshold = 0.5f;
    bool _absorbent = false;
    float _linear_damp = 0.0f;
    float _angular_damp = 0.0f;
};

// 碰撞检测信息
struct CollisionResult {
    bool collided = false;
    Vector3 position;
    Vector3 normal;
    Vector3 collider_position;
    RefCounted* collider = nullptr;
    float depth = 0.0f;
};

// 刚体类型
enum class RigidBodyType {
    STATIC,      // 静态刚体
    DYNAMIC,     // 动态刚体
    KINEMATIC,   // 运动学刚体
    CHARACTER    // 角色控制器
};

// 3D 刚体
class RigidBody3D : public RefCounted {
public:
    RigidBody3D();
    ~RigidBody3D() override;

    const char* get_class_name() const override { return "RigidBody3D"; }

    // 刚体类型
    void set_body_type(RigidBodyType type) { _body_type = type; }
    RigidBodyType get_body_type() const { return _body_type; }

    // 质量
    void set_mass(float mass);
    float get_mass() const { return _mass; }

    void set_inverse_mass(float inv_mass);
    float get_inverse_mass() const { return _inverse_mass; }

    // 线性物理
    Vector3 get_linear_velocity() const { return _linear_velocity; }
    void set_linear_velocity(const Vector3& vel) { _linear_velocity = vel; }

    void apply_central_impulse(const Vector3& impulse);
    void apply_impulse(const Vector3& impulse, const Vector3& position);

    void add_force(const Vector3& force);
    void add_central_force(const Vector3& force);

    void set_linear_damp(float damp) { _linear_damp = damp; }
    float get_linear_damp() const { return _linear_damp; }

    // 角物理
    Vector3 get_angular_velocity() const { return _angular_velocity; }
    void set_angular_velocity(const Vector3& vel) { _angular_velocity = vel; }

    void apply_torque_impulse(const Vector3& torque);
    void add_torque(const Vector3& torque);

    void set_angular_damp(float damp) { _angular_damp = damp; }
    float get_angular_damp() const { return _angular_damp; }

    // 旋转
    void set_gravity_scale(float scale) { _gravity_scale = scale; }
    float get_gravity_scale() const { return _gravity_scale; }

    // 变换
    Vector3 get_position() const { return _position; }
    void set_position(const Vector3& pos) { _position = pos; }

    Quaternion get_rotation() const { return _rotation; }
    void set_rotation(const Quaternion& rot) { _rotation = rot; }

    Transform3D get_transform() const {
        Transform3D t;
        t.origin = _position;
        Matrix4 m = Quaternion::to_matrix(_rotation);
        t.basis_x = Vector3(m.m[0][0], m.m[1][0], m.m[2][0]);
        t.basis_y = Vector3(m.m[0][1], m.m[1][1], m.m[2][1]);
        t.basis_z = Vector3(m.m[0][2], m.m[1][2], m.m[2][2]);
        return t;
    }

    // 碰撞层
    void set_collision_layer(PhysicsCollisionLayer layer) { _collision_layer = layer; }
    PhysicsCollisionLayer get_collision_layer() const { return _collision_layer; }

    void set_collision_mask(PhysicsCollisionMask mask) { _collision_mask = mask; }
    PhysicsCollisionMask get_collision_mask() const { return _collision_mask; }

    // 材质
    void set_physics_material(Ref<PhysicsMaterial> material) { _material = material; }
    Ref<PhysicsMaterial> get_physics_material() const { return _material; }

    // 睡眠
    void set_can_sleep(bool can_sleep) { _can_sleep = can_sleep; }
    bool can_sleep() const { return _can_sleep; }

    void set_sleeping(bool sleeping) { _sleeping = sleeping; }
    bool is_sleeping() const { return _sleeping; }

    // 碰撞检测
    bool test_collision(const Vector3& motion, CollisionResult& result);

    // 内部更新
    void _integrate_forces(float delta);
    void _integrate_velocity(float delta);
    void _apply_transform();

    // 状态
    bool _active = true;

private:
    RigidBodyType _body_type = RigidBodyType::DYNAMIC;
    float _mass = 1.0f;
    float _inverse_mass = 1.0f;

    Vector3 _position;
    Quaternion _rotation;

    Vector3 _linear_velocity;
    Vector3 _angular_velocity;

    float _linear_damp = 0.0f;
    float _angular_damp = 0.0f;
    float _gravity_scale = 1.0f;

    PhysicsCollisionLayer _collision_layer = 1;
    PhysicsCollisionMask _collision_mask = 1;

    Ref<PhysicsMaterial> _material;

    bool _can_sleep = true;
    bool _sleeping = false;

    Vector3 _force;
    Vector3 _torque;
};

} // namespace MyEngine
