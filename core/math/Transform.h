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

// 四元数
struct Quaternion {
    float x = 0, y = 0, z = 0, w = 1;

    Quaternion() = default;
    Quaternion(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    static Quaternion identity() { return Quaternion(0, 0, 0, 1); }

    static Quaternion from_euler(const Vector3& euler) {
        float cx = std::cos(euler.x * 0.5f);
        float cy = std::cos(euler.y * 0.5f);
        float cz = std::cos(euler.z * 0.5f);
        float sx = std::sin(euler.x * 0.5f);
        float sy = std::sin(euler.y * 0.5f);
        float sz = std::sin(euler.z * 0.5f);

        return Quaternion(
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cz * sy + sx * sz * cy,
            cx * cy * cz - sx * sy * sz
        );
    }

    static Quaternion from_axis_angle(const Vector3& axis, float angle) {
        float half = angle * 0.5f;
        float s = std::sin(half);
        return Quaternion(axis.x * s, axis.y * s, axis.z * s, std::cos(half));
    }

    static Quaternion from_matrix(const Matrix4& m) {
        float trace = m.m[0][0] + m.m[1][1] + m.m[2][2];
        Quaternion q;

        if (trace > 0) {
            float s = 0.5f / std::sqrt(trace + 1.0f);
            q.w = 0.25f / s;
            q.x = (m.m[2][1] - m.m[1][2]) * s;
            q.y = (m.m[0][2] - m.m[2][0]) * s;
            q.z = (m.m[1][0] - m.m[0][1]) * s;
        } else if (m.m[0][0] > m.m[1][1] && m.m[0][0] > m.m[2][2]) {
            float s = 2.0f * std::sqrt(1.0f + m.m[0][0] - m.m[1][1] - m.m[2][2]);
            q.w = (m.m[2][1] - m.m[1][2]) / s;
            q.x = 0.25f * s;
            q.y = (m.m[0][1] + m.m[1][0]) / s;
            q.z = (m.m[0][2] + m.m[2][0]) / s;
        } else if (m.m[1][1] > m.m[2][2]) {
            float s = 2.0f * std::sqrt(1.0f + m.m[1][1] - m.m[0][0] - m.m[2][2]);
            q.w = (m.m[0][2] - m.m[2][0]) / s;
            q.x = (m.m[0][1] + m.m[1][0]) / s;
            q.y = 0.25f * s;
            q.z = (m.m[1][2] + m.m[2][1]) / s;
        } else {
            float s = 2.0f * std::sqrt(1.0f + m.m[2][2] - m.m[0][0] - m.m[1][1]);
            q.w = (m.m[1][0] - m.m[0][1]) / s;
            q.x = (m.m[0][2] + m.m[2][0]) / s;
            q.y = (m.m[1][2] + m.m[2][1]) / s;
            q.z = 0.25f * s;
        }

        return q;
    }

    static Matrix4 to_matrix(const Quaternion& q) {
        float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
        float xx = q.x * x2, xy = q.x * y2, xz = q.x * z2;
        float yy = q.y * y2, yz = q.y * z2, zz = q.z * z2;
        float wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;

        Matrix4 m = Matrix4::identity();
        m.m[0][0] = 1 - (yy + zz);
        m.m[1][0] = xy + wz;
        m.m[2][0] = xz - wy;
        m.m[0][1] = xy - wz;
        m.m[1][1] = 1 - (xx + zz);
        m.m[2][1] = yz + wx;
        m.m[0][2] = xz + wy;
        m.m[1][2] = yz - wx;
        m.m[2][2] = 1 - (xx + yy);
        return m;
    }

    Quaternion operator*(const Quaternion& other) const {
        return Quaternion(
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w,
            w * other.w - x * other.x - y * other.y - z * other.z
        );
    }

    Vector3 operator*(const Vector3& v) const {
        Vector3 q(x, y, z);
        return v + cross(q, cross(q, v) + v * w) * 2.0f;
    }

    float length() const { return std::sqrt(x * x + y * y + z * z + w * w); }

    Quaternion normalized() const {
        float len = length();
        if (len == 0) return Quaternion::identity();
        return Quaternion(x / len, y / len, z / len, w / len);
    }

    static float dot(const Quaternion& a, const Quaternion& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    static Vector3 cross(const Vector3& a, const Vector3& b) {
        return Vector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }

    static Quaternion slerp(const Quaternion& a, const Quaternion& b, float t) {
        float cos = dot(a, b);
        Quaternion from = b;
        if (cos < 0) {
            from = Quaternion(-b.x, -b.y, -b.z, -b.w);
            cos = -cos;
        }

        if (cos > 0.9999f) {
            return Quaternion(
                a.x + t * (from.x - a.x),
                a.y + t * (from.y - a.y),
                a.z + t * (from.z - a.z),
                a.w + t * (from.w - a.w)
            ).normalized();
        }

        float angle = std::acos(cos);
        float sin = std::sin(angle);
        float s1 = std::sin((1 - t) * angle) / sin;
        float s2 = std::sin(t * angle) / sin;

        return Quaternion(
            s1 * a.x + s2 * from.x,
            s1 * a.y + s2 * from.y,
            s1 * a.z + s2 * from.z,
            s1 * a.w + s2 * from.w
        );
    }
};

} // namespace MyEngine
