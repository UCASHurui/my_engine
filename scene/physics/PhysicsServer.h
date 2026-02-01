#pragma once

#include "core/math/Vector3.h"
#include "core/object/RefCounted.h"
#include "RigidBody3D.h"
#include "CollisionShape3D.h"
#include <vector>
#include <unordered_map>

namespace MyEngine {

// 物理服务器
class PhysicsServer : public RefCounted {
public:
    PhysicsServer();
    ~PhysicsServer() override;

    const char* get_class_name() const override { return "PhysicsServer"; }

    // 单例
    static PhysicsServer* get() { return _singleton; }

    // 初始化/清理
    void initialize();
    void shutdown();

    // 物理世界设置
    void set_gravity(const Vector3& gravity) { _gravity = gravity; }
    Vector3 get_gravity() const { return _gravity; }

    void set_default_solver_iterations(int iterations) { _solver_iterations = iterations; }
    int get_default_solver_iterations() const { return _solver_iterations; }

    void set_default_contact_penetration(float penetration) { _contact_penetration = penetration; }
    float get_default_contact_penetration() const { return _contact_penetration; }

    void set_sleep_threshold_linear(float threshold) { _sleep_threshold_linear = threshold; }
    float get_sleep_threshold_linear() const { return _sleep_threshold_linear; }

    void set_sleep_threshold_angular(float threshold) { _sleep_threshold_angular = threshold; }
    float get_sleep_threshold_angular() const { return _sleep_threshold_angular; }

    void set_sleep_time(float time) { _sleep_time = time; }
    float get_sleep_time() const { return _sleep_time; }

    // 刚体管理
    Ref<RigidBody3D> body_create();
    void body_free(RigidBody3D* body);

    void body_set_state(RigidBody3D* body, const Vector3& position, const Quaternion& rotation);
    void body_get_state(RigidBody3D* body, Vector3& position, Quaternion& rotation);

    void body_set_active(RigidBody3D* body, bool active);
    bool body_is_active(RigidBody3D* body) const;

    void body_add_collision_shape(RigidBody3D* body, CollisionShape3D* shape);
    void body_remove_collision_shape(RigidBody3D* body, CollisionShape3D* shape);

    // 碰撞检测
    bool body_test_collision(RigidBody3D* body, const Vector3& motion, CollisionResult& result);
    bool body_test_move(RigidBody3D* body, const Vector3& motion, CollisionResult& result);

    // 射线检测
    bool ray_cast(const Vector3& from, const Vector3& to, CollisionResult& result);
    bool ray_cast_segment(const Vector3& from, const Vector3& to, CollisionResult& result);

    // 步进
    void step(float delta);

    // 碰撞回调
    using CollisionCallback = std::function<void(RigidBody3D*, RigidBody3D*, const Vector3&, const Vector3&)>;
    void set_collision_callback(CollisionCallback callback);

private:
    static PhysicsServer* _singleton;

    Vector3 _gravity = Vector3(0, -9.8f, 0);
    int _solver_iterations = 8;
    float _contact_penetration = 0.001f;
    float _sleep_threshold_linear = 0.1f;
    float _sleep_threshold_angular = 0.1f;
    float _sleep_time = 0.5f;

    std::unordered_map<uint32_t, RigidBody3D*> _bodies;
    uint32_t _next_body_id = 0;

    CollisionCallback _collision_callback;

    void _step_bodies(float delta);
    void _step_solver(float delta);
    void _check_sleeping(float delta);
};

} // namespace MyEngine
