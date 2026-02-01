#pragma once

#include "Node3D.h"
#include "core/math/Matrix4.h"
#include "core/math/Ray.h"

namespace MyEngine {

// 投影类型
enum class ProjectionType {
    PERSPECTIVE,
    ORTHOGRAPHIC
};

// 3D 相机
class Camera3D : public Node3D {
public:
    Camera3D();
    ~Camera3D() override;

    const char* get_class_name() const override { return "Camera3D"; }

    // 投影设置
    void set_projection(ProjectionType type) { _projection_type = type; _update_projection(); }
    ProjectionType get_projection() const { return _projection_type; }

    void set_fov(float fov_degrees) { _fov = fov_degrees; _update_projection(); }
    float get_fov() const { return _fov; }

    void set_size(float size) { _size = size; _update_projection(); }
    float get_size() const { return _size; }

    void set_aspect(float aspect) { _aspect = aspect; _update_projection(); }
    float get_aspect() const { return _aspect; }

    void set_near(float near_z) { _near = near_z; _update_projection(); }
    float get_near() const { return _near; }

    void set_far(float far_z) { _far = far_z; _update_projection(); }
    float get_far() const { return _far; }

    // 投影矩阵
    Matrix4 get_projection_matrix() const { return _projection_matrix; }
    Matrix4 get_view_matrix() const { return _view_matrix; }
    Matrix4 get_view_projection_matrix() const { return _projection_matrix * _view_matrix; }

    // 设置为当前相机
    void make_current();
    static Camera3D* get_current() { return _current_camera; }

    // 屏幕射线
    Vector3 project_position(const Vector3& world_pos) const;
    Vector3 unproject_position(const Vector3& screen_pos) const;
    Ray get_ray_from_screen(const Vector2& screen_pos) const;

    // 视锥体剔除
    bool is_point_in_frustum(const Vector3& point) const;
    bool is_sphere_in_frustum(const Vector3& center, float radius) const;

    // 清除
    void set_clear_mode(bool enabled) { _clear_enabled = enabled; }
    bool get_clear_enabled() const { return _clear_enabled; }

protected:
    void _update_projection();
    void _update_view();

private:
    ProjectionType _projection_type = ProjectionType::PERSPECTIVE;
    float _fov = 60.0f;
    float _size = 10.0f;
    float _aspect = 16.0f / 9.0f;
    float _near = 0.1f;
    float _far = 1000.0f;

    Matrix4 _projection_matrix;
    Matrix4 _view_matrix;

    bool _clear_enabled = true;

    static Camera3D* _current_camera;
};

} // namespace MyEngine
