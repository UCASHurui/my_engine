#pragma once

#include "core/math/Vector3.h"
#include "core/math/Transform.h"
#include "scene/main/Node.h"

namespace MyEngine {

// 3D 节点基类
class Node3D : public Node {

public:
    Node3D();
    virtual ~Node3D();

    virtual const char* get_class_name() const override { return "Node3D"; }

    // 变换
    void set_position(const Vector3& pos) { _position = pos; }
    Vector3 get_position() const { return _position; }
    void set_rotation(const Vector3& euler_radians) { _rotation = euler_radians; }
    Vector3 get_rotation() const { return _rotation; }
    void set_rotation_degrees(const Vector3& degrees) { _rotation = degrees * 0.0174533f; }
    Vector3 get_rotation_degrees() const { return _rotation * 57.2958f; }
    void set_scale(const Vector3& scale) { _scale = scale; }
    Vector3 get_scale() const { return _scale; }

    void set_global_position(const Vector3& pos);
    Vector3 get_global_position() const;
    void set_global_rotation(const Vector3& euler);
    Vector3 get_global_rotation() const;
    void set_global_scale(const Vector3& scale);
    Vector3 get_global_scale() const;

    // 变换矩阵
    Transform3D get_transform() const;
    Transform3D get_global_transform() const;

    // 方向
    Vector3 get_forward() const;
    Vector3 get_up() const;
    Vector3 get_right() const;

    void look_at(const Vector3& target, const Vector3& up = Vector3::UP);
    void look_at_from_position(const Vector3& position, const Vector3& target, const Vector3& up = Vector3::UP);

    // 空间通知
    void set_as_toplevel(bool toplevel);
    bool is_toplevel() const { return _toplevel; }
    void force_update_transform();

protected:
    Vector3 _position;
    Vector3 _rotation;
    Vector3 _scale = Vector3::ONE;
    bool _toplevel = false;
    Transform3D _global_transform;
    bool _global_transform_dirty = true;

    void _update_global_transform();
};

} // namespace MyEngine
