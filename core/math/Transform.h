#pragma once

#include "Vector2.h"
#include "Vector3.h"
#include "Matrix4.h"

namespace MyEngine {

// 2D 变换
struct Transform2D {
    Vector2 origin;
    Vector2 basis_x;
    Vector2 basis_y;

    Transform2D() : origin(0, 0), basis_x(1, 0), basis_y(0, 1) {}
    Transform2D(Vector2 origin, Vector2 basis_x, Vector2 basis_y)
        : origin(origin), basis_x(basis_x), basis_y(basis_y) {}

    static Transform2D identity() { return Transform2D(); }
    static Transform2D translation(Vector2 v) { return Transform2D(v, Vector2(1, 0), Vector2(0, 1)); }
    static Transform2D scaling(Vector2 s) { return Transform2D(Vector2(0, 0), Vector2(s.x, 0), Vector2(0, s.y)); }
    static Transform2D rotation(float angle) {
        float c = std::cos(angle);
        float s = std::sin(angle);
        return Transform2D(Vector2(0, 0), Vector2(c, s), Vector2(-s, c));
    }

    Transform2D operator*(const Transform2D& other) const {
        Transform2D r;
        r.basis_x = Vector2(basis_x.x * other.basis_x.x + basis_x.y * other.basis_y.x,
                           basis_x.x * other.basis_x.y + basis_x.y * other.basis_y.y);
        r.basis_y = Vector2(basis_y.x * other.basis_x.x + basis_y.y * other.basis_y.x,
                           basis_y.x * other.basis_x.y + basis_y.y * other.basis_y.y);
        r.origin = Vector2(origin.x + basis_x.x * other.origin.x + basis_x.y * other.origin.y,
                          origin.y + basis_y.x * other.origin.x + basis_y.y * other.origin.y);
        return r;
    }

    Vector2 operator*(const Vector2& v) const {
        return Vector2(basis_x.x * v.x + basis_y.x * v.y + origin.x,
                      basis_x.y * v.x + basis_y.y * v.y + origin.y);
    }

    Transform2D affine_inverse() const {
        float det = basis_x.x * basis_y.y - basis_x.y * basis_y.x;
        if (det == 0) return identity();

        float inv_det = 1.0f / det;
        Transform2D inv;
        inv.basis_x.x = basis_y.y * inv_det;
        inv.basis_x.y = -basis_x.y * inv_det;
        inv.basis_y.x = -basis_y.x * inv_det;
        inv.basis_y.y = basis_x.x * inv_det;
        inv.origin.x = -(inv.basis_x.x * origin.x + inv.basis_x.y * origin.y);
        inv.origin.y = -(inv.basis_y.x * origin.x + inv.basis_y.y * origin.y);
        return inv;
    }

    Vector2 get_scale() const {
        return Vector2(basis_x.length(), basis_y.length());
    }

    float get_rotation() const {
        return std::atan2(basis_x.y, basis_x.x);
    }
};

// 3D 变换
struct Transform3D {
    Vector3 origin;
    Vector3 basis_x;
    Vector3 basis_y;
    Vector3 basis_z;

    Transform3D() : origin(0, 0, 0), basis_x(1, 0, 0), basis_y(0, 1, 0), basis_z(0, 0, 1) {}

    static Transform3D identity() { return Transform3D(); }
    static Transform3D translation(Vector3 v) {
        Transform3D t;
        t.origin = v;
        return t;
    }
    static Transform3D scaling(Vector3 s) {
        Transform3D t;
        t.basis_x.x = s.x;
        t.basis_y.y = s.y;
        t.basis_z.z = s.z;
        return t;
    }

    Transform3D operator*(const Transform3D& other) const {
        Transform3D r;
        r.basis_x = basis_x * other.basis_x.x + basis_y * other.basis_y.x + basis_z * other.basis_z.x;
        r.basis_y = basis_x * other.basis_x.y + basis_y * other.basis_y.y + basis_z * other.basis_z.y;
        r.basis_z = basis_x * other.basis_x.z + basis_y * other.basis_y.z + basis_z * other.basis_z.z;
        r.origin = basis_x * other.origin.x + basis_y * other.origin.y + basis_z * other.origin.z + origin;
        return r;
    }

    Vector3 operator*(const Vector3& v) const {
        return basis_x * v.x + basis_y * v.y + basis_z * v.z + origin;
    }

    Transform3D affine_inverse() const {
        Matrix4 m = to_matrix4();
        Transform3D inv;
        Matrix4 inv_m = m.inverse();
        inv.basis_x = Vector3(inv_m.m[0][0], inv_m.m[1][0], inv_m.m[2][0]);
        inv.basis_y = Vector3(inv_m.m[0][1], inv_m.m[1][1], inv_m.m[2][1]);
        inv.basis_z = Vector3(inv_m.m[0][2], inv_m.m[1][2], inv_m.m[2][2]);
        inv.origin = Vector3(inv_m.m[0][3], inv_m.m[1][3], inv_m.m[2][3]);
        return inv;
    }

    Matrix4 to_matrix4() const {
        Matrix4 m = Matrix4::identity();
        m.m[0][0] = basis_x.x; m.m[0][1] = basis_x.y; m.m[0][2] = basis_x.z;
        m.m[1][0] = basis_y.x; m.m[1][1] = basis_y.y; m.m[1][2] = basis_y.z;
        m.m[2][0] = basis_z.x; m.m[2][1] = basis_z.y; m.m[2][2] = basis_z.z;
        m.m[3][0] = origin.x; m.m[3][1] = origin.y; m.m[3][2] = origin.z;
        return m;
    }

    Vector3 get_scale() const {
        return Vector3(basis_x.length(), basis_y.length(), basis_z.length());
    }
};

} // namespace MyEngine
