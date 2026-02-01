#pragma once

#include "Vector3.h"
#include "Vector4.h"
#include <cstring>
#include <cmath>

namespace MyEngine {

struct Matrix4 {
    float m[4][4];

    // 构造函数
    Matrix4() { identity(); }

    static Matrix4 identity() {
        Matrix4 r;
        r.m[0][0] = 1; r.m[0][1] = 0; r.m[0][2] = 0; r.m[0][3] = 0;
        r.m[1][0] = 0; r.m[1][1] = 1; r.m[1][2] = 0; r.m[1][3] = 0;
        r.m[2][0] = 0; r.m[2][1] = 0; r.m[2][2] = 1; r.m[2][3] = 0;
        r.m[3][0] = 0; r.m[3][1] = 0; r.m[3][2] = 0; r.m[3][3] = 1;
        return r;
    }

    static Matrix4 zero() {
        Matrix4 r;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                r.m[i][j] = 0;
        return r;
    }

    // 变换矩阵创建
    static Matrix4 translation(const Vector3& v) {
        Matrix4 r = identity();
        r.m[3][0] = v.x;
        r.m[3][1] = v.y;
        r.m[3][2] = v.z;
        return r;
    }

    static Matrix4 scaling(const Vector3& v) {
        Matrix4 r = identity();
        r.m[0][0] = v.x;
        r.m[1][1] = v.y;
        r.m[2][2] = v.z;
        return r;
    }

    static Matrix4 rotation_x(float angle) {
        Matrix4 r = identity();
        float c = std::cos(angle);
        float s = std::sin(angle);
        r.m[1][1] = c; r.m[1][2] = -s;
        r.m[2][1] = s; r.m[2][2] = c;
        return r;
    }

    static Matrix4 rotation_y(float angle) {
        Matrix4 r = identity();
        float c = std::cos(angle);
        float s = std::sin(angle);
        r.m[0][0] = c; r.m[0][2] = s;
        r.m[2][0] = -s; r.m[2][2] = c;
        return r;
    }

    static Matrix4 rotation_z(float angle) {
        Matrix4 r = identity();
        float c = std::cos(angle);
        float s = std::sin(angle);
        r.m[0][0] = c; r.m[0][1] = -s;
        r.m[1][0] = s; r.m[1][1] = c;
        return r;
    }

    static Matrix4 rotation_axis(const Vector3& axis, float angle) {
        Vector3 n = axis.normalized();
        float c = std::cos(angle);
        float s = std::sin(angle);
        float t = 1 - c;

        Matrix4 r = identity();
        r.m[0][0] = t * n.x * n.x + c;
        r.m[0][1] = t * n.x * n.y - s * n.z;
        r.m[0][2] = t * n.x * n.z + s * n.y;

        r.m[1][0] = t * n.x * n.y + s * n.z;
        r.m[1][1] = t * n.y * n.y + c;
        r.m[1][2] = t * n.y * n.z - s * n.x;

        r.m[2][0] = t * n.x * n.z - s * n.y;
        r.m[2][1] = t * n.y * n.z + s * n.x;
        r.m[2][2] = t * n.z * n.z + c;

        return r;
    }

    // 投影矩阵
    static Matrix4 orthographic(float left, float right, float bottom, float top,
                                 float near, float far) {
        Matrix4 r = identity();
        r.m[0][0] = 2 / (right - left);
        r.m[1][1] = 2 / (top - bottom);
        r.m[2][2] = -2 / (far - near);
        r.m[3][0] = -(right + left) / (right - left);
        r.m[3][1] = -(top + bottom) / (top - bottom);
        r.m[3][2] = -(far + near) / (far - near);
        return r;
    }

    static Matrix4 perspective(float fov_y, float aspect, float near, float far) {
        Matrix4 r = zero();
        float f = 1.0f / std::tan(fov_y / 2.0f);
        r.m[0][0] = f / aspect;
        r.m[1][1] = f;
        r.m[2][2] = (far + near) / (near - far);
        r.m[2][3] = -1.0f;
        r.m[3][2] = (2 * far * near) / (near - far);
        return r;
    }

    // LookAt 矩阵
    static Matrix4 look_at(const Vector3& eye, const Vector3& target, const Vector3& up) {
        Vector3 z = (eye - target).normalized();
        Vector3 x = up.cross(z).normalized();
        Vector3 y = z.cross(x);

        Matrix4 r = identity();
        r.m[0][0] = x.x; r.m[0][1] = x.y; r.m[0][2] = x.z;
        r.m[1][0] = y.x; r.m[1][1] = y.y; r.m[1][2] = y.z;
        r.m[2][0] = z.x; r.m[2][1] = z.y; r.m[2][2] = z.z;
        r.m[3][0] = -x.dot(eye);
        r.m[3][1] = -y.dot(eye);
        r.m[3][2] = -z.dot(eye);
        return r;
    }

    // 运算
    Matrix4 operator*(const Matrix4& other) const {
        Matrix4 r;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                r.m[i][j] = m[i][0] * other.m[0][j] +
                            m[i][1] * other.m[1][j] +
                            m[i][2] * other.m[2][j] +
                            m[i][3] * other.m[3][j];
            }
        }
        return r;
    }

    Vector3 operator*(const Vector3& v) const {
        float w = m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3];
        if (w == 0) w = 1;
        return Vector3(
            (m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3]) / w,
            (m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3]) / w,
            (m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3]) / w
        );
    }

    Matrix4 operator+(const Matrix4& other) const {
        Matrix4 r;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                r.m[i][j] = m[i][j] + other.m[i][j];
        return r;
    }

    Matrix4& operator*=(const Matrix4& other) { *this = *this * other; return *this; }

    // 行列式和逆
    float determinant() const;
    Matrix4 inverse() const;
    Matrix4 transposed() const;

    // 获取变换分量
    Vector3 get_origin() const { return Vector3(m[3][0], m[3][1], m[3][2]); }
    Vector3 get_scale() const;
};

} // namespace MyEngine
