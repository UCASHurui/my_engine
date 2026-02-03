#pragma once

#include "math/Vector3.h"
#include "math/Transform.h"
#include "math/Matrix4.h"
#include "object/RefCounted.h"

namespace MyEngine {

// 碰撞形状类型
enum class CollisionShapeType {
    SPHERE,
    BOX,
    CAPSULE,
    CYLINDER,
    CONVEX_POLYGON,
    CONCAVE_POLYGON,
    HEIGHTMAP,
    RAY
};

// 3D 碰撞形状
class CollisionShape3D : public RefCounted {
public:
    CollisionShape3D();
    ~CollisionShape3D() override;

    const char* get_class_name() const override { return "CollisionShape3D"; }

    // 形状类型
    void set_shape_type(CollisionShapeType type) { _shape_type = type; }
    CollisionShapeType get_shape_type() const { return _shape_type; }

    // 局部变换
    void set_local_transform(const Transform3D& transform) { _local_transform = transform; }
    Transform3D get_local_transform() const { return _local_transform; }

    void set_local_position(const Vector3& pos) { _local_transform.origin = pos; }
    Vector3 get_local_position() const { return _local_transform.origin; }

    void set_local_rotation(const Quaternion& rot) {
        Matrix4 m = Quaternion::to_matrix(rot);
        _local_transform.basis_x = Vector3(m.m[0][0], m.m[1][0], m.m[2][0]);
        _local_transform.basis_y = Vector3(m.m[0][1], m.m[1][1], m.m[2][1]);
        _local_transform.basis_z = Vector3(m.m[0][2], m.m[1][2], m.m[2][2]);
    }
    Quaternion get_local_rotation() const {
        Matrix4 m;
        m.m[0][0] = _local_transform.basis_x.x; m.m[1][0] = _local_transform.basis_x.y; m.m[2][0] = _local_transform.basis_x.z;
        m.m[0][1] = _local_transform.basis_y.x; m.m[1][1] = _local_transform.basis_y.y; m.m[2][1] = _local_transform.basis_y.z;
        m.m[0][2] = _local_transform.basis_z.x; m.m[1][2] = _local_transform.basis_z.y; m.m[2][2] = _local_transform.basis_z.z;
        return Quaternion::from_matrix(m);
    }

    void set_local_scale(const Vector3& scale) { _local_scale = scale; }
    Vector3 get_local_scale() const { return _local_scale; }

    // 尺寸
    void set_size(const Vector3& size) { _size = size; }
    Vector3 get_size() const { return _size; }

    void set_radius(float radius) { _radius = radius; }
    float get_radius() const { return _radius; }

    void set_height(float height) { _height = height; }
    float get_height() const { return _height; }

    // 形状参数
    float get_volume() const;
    Vector3 get_inertia() const;

    // 碰撞检测
    bool intersect(const Transform3D& global_transform, const Vector3& point) const;
    bool intersect_ray(const Transform3D& global_transform, const Vector3& from, const Vector3& to, float& out_distance) const;

    // 碰撞检测（两个形状之间）
    bool intersect_shape(const Transform3D& transform_a, const CollisionShape3D* shape_b, const Transform3D& transform_b) const;

private:
    CollisionShapeType _shape_type = CollisionShapeType::SPHERE;
    Transform3D _local_transform;
    Vector3 _local_scale = Vector3::ONE;
    Vector3 _size = Vector3::ONE;
    float _radius = 0.5f;
    float _height = 1.0f;
};

} // namespace MyEngine
