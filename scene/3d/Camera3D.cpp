#include "Camera3D.h"
#include "os/OS.h"
#include "math/Math.h"

namespace MyEngine {

Camera3D* Camera3D::_current_camera = nullptr;

Camera3D::Camera3D() = default;

Camera3D::~Camera3D() {
    if (_current_camera == this) {
        _current_camera = nullptr;
    }
}

void Camera3D::_update_projection() {
    float aspect = _aspect > 0 ? _aspect : 16.0f / 9.0f;

    if (_projection_type == ProjectionType::PERSPECTIVE) {
        _projection_matrix = Matrix4::perspective(
            Math::deg_to_rad(_fov),
            aspect,
            _near,
            _far
        );
    } else {
        float half_size = _size * 0.5f;
        _projection_matrix = Matrix4::orthographic(
            -half_size * aspect, half_size * aspect,
            -half_size, half_size,
            _near, _far
        );
    }
}

void Camera3D::_update_view() {
    _view_matrix = get_global_transform().to_matrix4().inverse();
}

void Camera3D::make_current() {
    _current_camera = this;
    _update_view();
}

Vector3 Camera3D::project_position(const Vector3& world_pos) const {
    Vector4 clip_pos = _projection_matrix * _view_matrix * Vector4(world_pos.x, world_pos.y, world_pos.z, 1.0f);
    if (clip_pos.w == 0) return Vector3::ZERO;

    Vector3 ndc;
    ndc.x = clip_pos.x / clip_pos.w;
    ndc.y = clip_pos.y / clip_pos.w;
    ndc.z = clip_pos.z / clip_pos.w;

    // 转换到屏幕坐标 (假设 1280x720)
    Vector3 screen;
    screen.x = (ndc.x + 1.0f) * 0.5f * 1280.0f;
    screen.y = (1.0f - ndc.y) * 0.5f * 720.0f;
    screen.z = ndc.z;
    return screen;
}

Vector3 Camera3D::unproject_position(const Vector3& screen_pos) const {
    // 转换屏幕坐标到 NDC
    Vector3 ndc;
    ndc.x = screen_pos.x / 1280.0f * 2.0f - 1.0f;
    ndc.y = 1.0f - screen_pos.y / 720.0f * 2.0f;
    ndc.z = screen_pos.z;

    // 逆投影
    Matrix4 inv_proj = _projection_matrix.inverse();
    Matrix4 inv_view = _view_matrix.inverse();

    Vector4 clip(ndc.x, ndc.y, -1.0f, 1.0f);
    Vector4 eye = inv_proj * clip;
    eye.z = -1.0f;
    eye.w = 0.0f;

    Vector4 world = inv_view * eye;
    return Vector3(world.x, world.y, world.z).normalized();
}

Ray Camera3D::get_ray_from_screen(const Vector2& screen_pos) const {
    Vector3 origin = get_global_position();
    Vector3 direction = unproject_position(Vector3(screen_pos.x, screen_pos.y, 0.0f));
    return Ray(origin, direction);
}

bool Camera3D::is_point_in_frustum(const Vector3& point) const {
    (void)point;
    // 简化实现：始终返回 true
    return true;
}

bool Camera3D::is_sphere_in_frustum(const Vector3& center, float radius) const {
    (void)center;
    (void)radius;
    // 简化实现：始终返回 true
    return true;
}

} // namespace MyEngine
