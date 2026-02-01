#pragma once

#include <cmath>
#include <cstddef>
#include <algorithm>
#include "Vector2.h"

namespace MyEngine {

struct Vector3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    // 构造函数
    Vector3() = default;
    Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

    // 从 Vector2 构造
    explicit Vector3(const Vector2& v, float z = 0.0f) : x(v.x), y(v.y), z(z) {}

    // 常量
    static const Vector3 ZERO;
    static const Vector3 ONE;
    static const Vector3 UP;
    static const Vector3 DOWN;
    static const Vector3 LEFT;
    static const Vector3 RIGHT;
    static const Vector3 FORWARD;
    static const Vector3 BACK;

    // 属性
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    float length_squared() const { return x * x + y * y + z * z; }
    Vector3 normalized() const {
        float l = length();
        if (l > 0) return Vector3(x / l, y / l, z / l);
        return Vector3::ZERO;
    }
    Vector3 inverse() const { return Vector3(-x, -y, -z); }

    // 运算
    Vector3 operator+(const Vector3& other) const { return Vector3(x + other.x, y + other.y, z + other.z); }
    Vector3 operator-(const Vector3& other) const { return Vector3(x - other.x, y - other.y, z - other.z); }
    Vector3 operator*(float s) const { return Vector3(x * s, y * s, z * s); }
    Vector3 operator/(float s) const { return Vector3(x / s, y / s, z / s); }
    Vector3 operator-() const { return Vector3(-x, -y, -z); }

    Vector3& operator+=(const Vector3& other) { x += other.x; y += other.y; z += other.z; return *this; }
    Vector3& operator-=(const Vector3& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }
    Vector3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    Vector3& operator/=(float s) { x /= s; y /= s; z /= s; return *this; }

    bool operator==(const Vector3& other) const { return x == other.x && y == other.y && z == other.z; }
    bool operator!=(const Vector3& other) const { return x != other.x || y != other.y || z != other.z; }

    float dot(const Vector3& other) const { return x * other.x + y * other.y + z * other.z; }
    Vector3 cross(const Vector3& other) const {
        return Vector3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    // 插值
    Vector3 lerp(const Vector3& to, float weight) const {
        return Vector3(
            x + (to.x - x) * weight,
            y + (to.y - y) * weight,
            z + (to.z - z) * weight
        );
    }

    // 坐标轴投影
    Vector3 abs() const { return Vector3(std::abs(x), std::abs(y), std::abs(z)); }
    float max_axis() const { return std::max({x, y, z}); }
    float min_axis() const { return std::min({x, y, z}); }
};

} // namespace MyEngine
