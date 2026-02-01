#include "PhysicsServer.h"

namespace MyEngine {

PhysicsServer* PhysicsServer::_singleton = nullptr;

PhysicsServer::PhysicsServer() {
    _singleton = this;
}

PhysicsServer::~PhysicsServer() {
    if (_singleton == this) {
        _singleton = nullptr;
    }
}

void PhysicsServer::initialize() {
    // 初始化物理世界
}

void PhysicsServer::shutdown() {
    // 清理物理世界
    for (auto& [id, body] : _bodies) {
        delete body;
    }
    _bodies.clear();
}

Ref<RigidBody3D> PhysicsServer::body_create() {
    RigidBody3D* body = new RigidBody3D();
    body->_active = true;
    _bodies[_next_body_id++] = body;
    return body;
}

void PhysicsServer::body_free(RigidBody3D* body) {
    for (auto it = _bodies.begin(); it != _bodies.end(); ++it) {
        if (it->second == body) {
            _bodies.erase(it);
            delete body;
            return;
        }
    }
}

void PhysicsServer::body_set_state(RigidBody3D* body, const Vector3& position, const Quaternion& rotation) {
    body->set_position(position);
    body->set_rotation(rotation);
}

void PhysicsServer::body_get_state(RigidBody3D* body, Vector3& position, Quaternion& rotation) {
    position = body->get_position();
    rotation = body->get_rotation();
}

void PhysicsServer::body_set_active(RigidBody3D* body, bool active) {
    body->_active = active;
}

bool PhysicsServer::body_is_active(RigidBody3D* body) const {
    return body->_active;
}

void PhysicsServer::body_add_collision_shape(RigidBody3D* body, CollisionShape3D* shape) {
    (void)body; (void)shape;
}

void PhysicsServer::body_remove_collision_shape(RigidBody3D* body, CollisionShape3D* shape) {
    (void)body; (void)shape;
}

bool PhysicsServer::body_test_collision(RigidBody3D* body, const Vector3& motion, CollisionResult& result) {
    (void)body; (void)motion;
    result.collided = false;
    return false;
}

bool PhysicsServer::body_test_move(RigidBody3D* body, const Vector3& motion, CollisionResult& result) {
    return body_test_collision(body, motion, result);
}

bool PhysicsServer::ray_cast(const Vector3& from, const Vector3& to, CollisionResult& result) {
    (void)from; (void)to;
    result.collided = false;
    return false;
}

bool PhysicsServer::ray_cast_segment(const Vector3& from, const Vector3& to, CollisionResult& result) {
    return ray_cast(from, to, result);
}

void PhysicsServer::step(float delta) {
    _step_bodies(delta);
    _step_solver(delta);
    _check_sleeping(delta);
}

void PhysicsServer::_step_bodies(float delta) {
    for (auto& [id, body] : _bodies) {
        if (!body->_active) continue;
        if (body->get_body_type() == RigidBodyType::STATIC) continue;

        body->_integrate_forces(delta);
    }

    for (auto& [id, body] : _bodies) {
        if (!body->_active) continue;
        if (body->get_body_type() == RigidBodyType::STATIC) continue;

        body->_integrate_velocity(delta);
        body->_apply_transform();
    }
}

void PhysicsServer::_step_solver(float delta) {
    (void)delta;
    // 简化的约束求解器
}

void PhysicsServer::_check_sleeping(float delta) {
    for (auto& [id, body] : _bodies) {
        if (!body->_active) continue;
        if (!body->can_sleep()) continue;
        if (body->get_body_type() == RigidBodyType::STATIC) continue;

        float linear_vel = body->get_linear_velocity().length();
        float angular_vel = body->get_angular_velocity().length();

        if (linear_vel < _sleep_threshold_linear && angular_vel < _sleep_threshold_angular) {
            body->set_sleeping(true);
            body->set_linear_velocity(Vector3::ZERO);
            body->set_angular_velocity(Vector3::ZERO);
        }
    }
}

void PhysicsServer::set_collision_callback(CollisionCallback callback) {
    _collision_callback = callback;
}

} // namespace MyEngine
