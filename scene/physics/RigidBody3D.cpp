#include "RigidBody3D.h"
#include "PhysicsServer.h"
#include "math/Transform.h"

namespace MyEngine {

PhysicsMaterial::PhysicsMaterial() = default;
PhysicsMaterial::~PhysicsMaterial() = default;

RigidBody3D::RigidBody3D() = default;
RigidBody3D::~RigidBody3D() = default;

void RigidBody3D::set_mass(float mass) {
    _mass = mass > 0 ? mass : 0.001f;
    _inverse_mass = 1.0f / _mass;
}

void RigidBody3D::set_inverse_mass(float inv_mass) {
    _inverse_mass = inv_mass;
    _mass = inv_mass > 0 ? 1.0f / inv_mass : 0.0f;
}

void RigidBody3D::apply_central_impulse(const Vector3& impulse) {
    if (_body_type == RigidBodyType::STATIC) return;
    _linear_velocity += impulse * _inverse_mass;
}

void RigidBody3D::apply_impulse(const Vector3& impulse, const Vector3& position) {
    if (_body_type == RigidBodyType::STATIC) return;
    _linear_velocity += impulse * _inverse_mass;
    _angular_velocity += Quaternion::cross(position - _position, impulse) * _inverse_mass;
}

void RigidBody3D::add_force(const Vector3& force) {
    if (_body_type == RigidBodyType::STATIC) return;
    _force += force;
}

void RigidBody3D::add_central_force(const Vector3& force) {
    add_force(force);
}

void RigidBody3D::apply_torque_impulse(const Vector3& torque) {
    if (_body_type == RigidBodyType::STATIC) return;
    _angular_velocity += torque * _inverse_mass;
}

void RigidBody3D::add_torque(const Vector3& torque) {
    if (_body_type == RigidBodyType::STATIC) return;
    _torque += torque;
}

bool RigidBody3D::test_collision(const Vector3& motion, CollisionResult& result) {
    (void)motion;
    result.collided = false;
    return false;
}

void RigidBody3D::_integrate_forces(float delta) {
    if (_body_type == RigidBodyType::STATIC || _sleeping) return;

    // 应用重力
    if (_gravity_scale != 0) {
        _force += PhysicsServer::get()->get_gravity() * _mass * _gravity_scale;
    }

    // 应用阻尼
    _force -= _linear_velocity * _linear_damp * _mass;
    _torque -= _angular_velocity * _angular_damp * _mass;
}

void RigidBody3D::_integrate_velocity(float delta) {
    if (_body_type == RigidBodyType::STATIC || _sleeping) return;

    // 更新速度
    _linear_velocity += _force * _inverse_mass * delta;
    _angular_velocity += _torque * _inverse_mass * delta;

    // 清除累积的力和扭矩
    _force = Vector3::ZERO;
    _torque = Vector3::ZERO;
}

void RigidBody3D::_apply_transform() {
    if (_body_type == RigidBodyType::STATIC) return;

    _position += _linear_velocity * 0.016f;
    // 更新旋转...
}

} // namespace MyEngine
