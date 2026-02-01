#pragma once

#include "Vector4.h"
#include <cstdint>

namespace MyEngine {

struct Color : public Vector4 {
    Color() : Vector4(0, 0, 0, 1) {}
    Color(float r, float g, float b, float a = 1.0f) : Vector4(r, g, b, a) {}

    static Color WHITE() { return Color(1, 1, 1, 1); }
    static Color BLACK() { return Color(0, 0, 0, 1); }
    static Color RED() { return Color(1, 0, 0, 1); }
    static Color GREEN() { return Color(0, 1, 0, 1); }
    static Color BLUE() { return Color(0, 0, 1, 1); }

    float& r() { return x; }
    float& g() { return y; }
    float& b() { return z; }
    float& a() { return w; }

    float r() const { return x; }
    float g() const { return y; }
    float b() const { return z; }
    float a() const { return w; }

    static Color from_rgb(uint32_t rgb) {
        return Color(
            ((rgb >> 16) & 0xFF) / 255.0f,
            ((rgb >> 8) & 0xFF) / 255.0f,
            (rgb & 0xFF) / 255.0f,
            1.0f
        );
    }

    static Color from_rgba(uint32_t rgba) {
        return Color(
            ((rgba >> 24) & 0xFF) / 255.0f,
            ((rgba >> 16) & 0xFF) / 255.0f,
            ((rgba >> 8) & 0xFF) / 255.0f,
            (rgba & 0xFF) / 255.0f
        );
    }

    uint32_t to_rgb() const {
        return (uint32_t)(r() * 255) << 16 |
               (uint32_t)(g() * 255) << 8 |
               (uint32_t)(b() * 255);
    }

    uint32_t to_rgba() const {
        return (uint32_t)(a() * 255) << 24 |
               (uint32_t)(r() * 255) << 16 |
               (uint32_t)(g() * 255) << 8 |
               (uint32_t)(b() * 255);
    }
};

} // namespace MyEngine
