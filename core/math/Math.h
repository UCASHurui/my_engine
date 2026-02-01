#pragma once

#include <cmath>

namespace MyEngine {

class Math {
public:
    // 常量
    static constexpr float PI = 3.14159265358979323846f;
    static constexpr float TWO_PI = PI * 2.0f;
    static constexpr float HALF_PI = PI * 0.5f;
    static constexpr float EPSILON = 1e-6f;
    static constexpr float INF = 1e30f;

    // 基本函数
    static float abs(float v) { return std::abs(v); }
    static float sqrt(float v) { return std::sqrt(v); }
    static float sin(float v) { return std::sin(v); }
    static float cos(float v) { return std::cos(v); }
    static float tan(float v) { return std::tan(v); }
    static float asin(float v) { return std::asin(v); }
    static float acos(float v) { return std::acos(v); }
    static float atan(float v) { return std::atan(v); }
    static float atan2(float y, float x) { return std::atan2(y, x); }

    static float pow(float base, float exp) { return std::pow(base, exp); }
    static float exp(float v) { return std::exp(v); }
    static float log(float v) { return std::log(v); }
    static float log10(float v) { return std::log10(v); }

    static float floor(float v) { return std::floor(v); }
    static float ceil(float v) { return std::ceil(v); }
    static float round(float v) { return std::round(v); }
    static float frac(float v) { return v - floor(v); }

    static float clamp(float v, float min, float max) {
        if (v < min) return min;
        if (v > max) return max;
        return v;
    }

    static float clamp01(float v) { return clamp(v, 0.0f, 1.0f); }

    static float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }

    static float smoothstep(float edge0, float edge1, float x) {
        float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }

    static float ease_in(float t) { return t * t; }
    static float ease_out(float t) { return t * (2.0f - t); }
    static float ease_in_out(float t) { return t < 0.5f ? 2.0f * t * t : -1.0f + (4.0f - 2.0f * t) * t; }

    static float deg_to_rad(float deg) { return deg * PI / 180.0f; }
    static float rad_to_deg(float rad) { return rad * 180.0f / PI; }

    static bool is_equal(float a, float b) {
        return abs(a - b) < EPSILON;
    }

    static bool is_zero(float v) {
        return abs(v) < EPSILON;
    }

    static float random() {
        return (float)rand() / (float)RAND_MAX;
    }

    static float random_range(float min, float max) {
        return lerp(min, max, random());
    }
};

} // namespace MyEngine
