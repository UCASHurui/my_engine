#pragma once

#include <cmath>
#include "Vector3.h"

namespace MyEngine {

struct Vector4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;

    Vector4() = default;
    Vector4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    explicit Vector4(const Vector3& v, float w = 1.0f) : x(v.x), y(v.y), z(v.z), w(w) {}

    static const Vector4 ZERO;
    static const Vector4 ONE;

    float length() const { return std::sqrt(x * x + y * y + z * z + w * w); }
    Vector4 normalized() const {
        float l = length();
        if (l > 0) return Vector4(x / l, y / l, z / l, w / l);
        return Vector4::ZERO;
    }

    Vector4 operator+(const Vector4& o) const { return Vector4(x + o.x, y + o.y, z + o.z, w + o.w); }
    Vector4 operator-(const Vector4& o) const { return Vector4(x - o.x, y - o.y, z - o.z, w - o.w); }
    Vector4 operator*(float s) const { return Vector4(x * s, y * s, z * s, w * s); }
    Vector4 operator/(float s) const { return Vector4(x / s, y / s, z / s, w / s); }

    float dot(const Vector4& o) const { return x * o.x + y * o.y + z * o.z + w * o.w; }
};

} // namespace MyEngine
