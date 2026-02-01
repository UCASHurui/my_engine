#pragma once

#include <cmath>
#include <cstddef>

namespace MyEngine {

struct Vector2 {
    float x = 0.0f;
    float y = 0.0f;

    // 构造函数
    Vector2() = default;
    Vector2(float x, float y) : x(x), y(y) {}

    // 常量
    static const Vector2 ZERO;
    static const Vector2 ONE;
    static const Vector2 UP;
    static const Vector2 DOWN;
    static const Vector2 LEFT;
    static const Vector2 RIGHT;

    // 属性
    float length() const { return std::sqrt(x * x + y * y); }
    float length_squared() const { return x * x + y * y; }
    float angle() const { return std::atan2(y, x); }
    Vector2 normalized() const {
        float l = length();
        if (l > 0) return Vector2(x / l, y / l);
        return Vector2::ZERO;
    }
    Vector2 limit_length(float max_length) const {
        float l = length();
        if (l > max_length && l > 0) {
            return Vector2(x / l * max_length, y / l * max_length);
        }
        return *this;
    }
    Vector2 rotated(float angle) const {
        float s = std::sin(angle);
        float c = std::cos(angle);
        return Vector2(x * c - y * s, x * s + y * c);
    }

    // 运算
    Vector2 operator+(const Vector2& other) const { return Vector2(x + other.x, y + other.y); }
    Vector2 operator-(const Vector2& other) const { return Vector2(x - other.x, y - other.y); }
    Vector2 operator*(float s) const { return Vector2(x * s, y * s); }
    Vector2 operator/(float s) const { return Vector2(x / s, y / s); }
    Vector2 operator-() const { return Vector2(-x, -y); }

    // Vector2 * Vector2 (element-wise multiplication)
    Vector2 operator*(const Vector2& other) const { return Vector2(x * other.x, y * other.y); }

    Vector2& operator+=(const Vector2& other) { x += other.x; y += other.y; return *this; }
    Vector2& operator-=(const Vector2& other) { x -= other.x; y -= other.y; return *this; }
    Vector2& operator*=(float s) { x *= s; y *= s; return *this; }
    Vector2& operator/=(float s) { x /= s; y /= s; return *this; }

    bool operator==(const Vector2& other) const { return x == other.x && y == other.y; }
    bool operator!=(const Vector2& other) const { return x != other.x || y != other.y; }
    bool operator<(const Vector2& other) const {
        return x < other.x && y < other.y;
    }

    float dot(const Vector2& other) const { return x * other.x + y * other.y; }
    float cross(const Vector2& other) const { return x * other.y - y * other.x; }
};

} // namespace MyEngine
