#include "CollisionShape3D.h"

namespace MyEngine {

CollisionShape3D::CollisionShape3D() = default;
CollisionShape3D::~CollisionShape3D() = default;

float CollisionShape3D::get_volume() const {
    switch (_shape_type) {
        case CollisionShapeType::SPHERE:
            return 4.0f / 3.0f * 3.14159f * _radius * _radius * _radius;
        case CollisionShapeType::BOX:
            return _size.x * _size.y * _size.z;
        case CollisionShapeType::CAPSULE:
        case CollisionShapeType::CYLINDER:
            return 3.14159f * _radius * _radius * _height;
        default:
            return 0.0f;
    }
}

Vector3 CollisionShape3D::get_inertia() const {
    float mass = 1.0f; // 假设质量为1
    switch (_shape_type) {
        case CollisionShapeType::SPHERE: {
            float inertia = 0.4f * mass * _radius * _radius;
            return Vector3(inertia, inertia, inertia);
        }
        case CollisionShapeType::BOX: {
            float inertia_x = mass * (_size.y * _size.y + _size.z * _size.z) / 12.0f;
            float inertia_y = mass * (_size.x * _size.x + _size.z * _size.z) / 12.0f;
            float inertia_z = mass * (_size.x * _size.x + _size.y * _size.y) / 12.0f;
            return Vector3(inertia_x, inertia_y, inertia_z);
        }
        case CollisionShapeType::CAPSULE:
        case CollisionShapeType::CYLINDER: {
            float inertia = mass * (_radius * _radius / 4.0f + _height * _height / 12.0f);
            return Vector3(inertia, inertia, inertia);
        }
        default:
            return Vector3::ONE;
    }
}

bool CollisionShape3D::intersect(const Transform3D& global_transform, const Vector3& point) const {
    (void)global_transform; (void)point;
    return false;
}

bool CollisionShape3D::intersect_ray(const Transform3D& global_transform, const Vector3& from, const Vector3& to, float& out_distance) const {
    (void)global_transform; (void)from; (void)to; (void)out_distance;
    return false;
}

bool CollisionShape3D::intersect_shape(const Transform3D& transform_a, const CollisionShape3D* shape_b, const Transform3D& transform_b) const {
    (void)transform_a; (void)shape_b; (void)transform_b;
    return false;
}

} // namespace MyEngine
