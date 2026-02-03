#include "Node3D.h"
#include "math/Matrix4.h"

namespace MyEngine {

Node3D::Node3D() {}

Node3D::~Node3D() {}

void Node3D::set_global_position(const Vector3& pos) {
    Node* parent = static_cast<Node*>(get_parent());
    if (parent) {
        Transform3D parent_global = parent->get_global_transform_3d();
        Transform3D inv = parent_global.affine_inverse();
        _position = inv * pos;
    } else {
        _position = pos;
    }
}

Vector3 Node3D::get_global_position() const {
    return get_global_transform().origin;
}

void Node3D::set_global_rotation(const Vector3& euler) {
    // 简化处理
    _rotation = euler;
}

Vector3 Node3D::get_global_rotation() const {
    // 简化处理
    return _rotation;
}

void Node3D::set_global_scale(const Vector3& scale) {
    Node* parent = static_cast<Node*>(get_parent());
    if (parent) {
        Vector3 parent_scale = parent->get_global_scale_3d();
        _scale = Vector3(
            parent_scale.x != 0 ? scale.x / parent_scale.x : 0,
            parent_scale.y != 0 ? scale.y / parent_scale.y : 0,
            parent_scale.z != 0 ? scale.z / parent_scale.z : 0
        );
    } else {
        _scale = scale;
    }
}

Vector3 Node3D::get_global_scale() const {
    return get_global_transform().get_scale();
}

Transform3D Node3D::get_transform() const {
    Transform3D t = Transform3D::translation(_position);

    Matrix4 rot_x = Matrix4::rotation_x(_rotation.x);
    Matrix4 rot_y = Matrix4::rotation_y(_rotation.y);
    Matrix4 rot_z = Matrix4::rotation_z(_rotation.z);
    Matrix4 rot = rot_z * rot_y * rot_x;
    Matrix4 scale = Matrix4::scaling(_scale);

    Matrix4 combined = rot * scale * t.to_matrix4();

    Transform3D result;
    result.basis_x = Vector3(combined.m[0][0], combined.m[1][0], combined.m[2][0]);
    result.basis_y = Vector3(combined.m[0][1], combined.m[1][1], combined.m[2][1]);
    result.basis_z = Vector3(combined.m[0][2], combined.m[1][2], combined.m[2][2]);
    result.origin = _position;
    return result;
}

Transform3D Node3D::get_global_transform() const {
    if (_global_transform_dirty) {
        Node3D* self = const_cast<Node3D*>(this);
        self->_update_global_transform();
    }
    return _global_transform;
}

void Node3D::_update_global_transform() {
    if (_toplevel || !get_parent()) {
        _global_transform = get_transform();
    } else {
        Node* parent = static_cast<Node*>(get_parent());
        _global_transform = parent->get_global_transform_3d() * get_transform();
    }
    _global_transform_dirty = false;
}

Vector3 Node3D::get_forward() const {
    Transform3D t = get_global_transform();
    return -t.basis_z.normalized();
}

Vector3 Node3D::get_up() const {
    Transform3D t = get_global_transform();
    return t.basis_y.normalized();
}

Vector3 Node3D::get_right() const {
    Transform3D t = get_global_transform();
    return t.basis_x.normalized();
}

void Node3D::look_at(const Vector3& target, const Vector3& up) {
    Vector3 pos = get_global_position();
    look_at_from_position(pos, target, up);
}

void Node3D::look_at_from_position(const Vector3& position, const Vector3& target, const Vector3& up) {
    _position = position;

    Vector3 forward = (target - position).normalized();
    Vector3 right = up.cross(forward).normalized();
    Vector3 new_up = forward.cross(right);

    Transform3D t;
    t.basis_x = right;
    t.basis_y = new_up;
    t.basis_z = -forward;
    t.origin = position;

    // 从矩阵提取欧拉角（简化）
    _rotation = Vector3::ZERO;

    _global_transform = t;
    _global_transform_dirty = false;
}

void Node3D::set_as_toplevel(bool toplevel) {
    if (_toplevel != toplevel) {
        _toplevel = toplevel;
        _global_transform_dirty = true;
    }
}

void Node3D::force_update_transform() {
    _global_transform_dirty = true;
    _update_global_transform();
}

} // namespace MyEngine
