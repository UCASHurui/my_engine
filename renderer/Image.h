#pragma once

#include <cstdint>

namespace MyEngine {

struct Color {
    float r, g, b, a;

    Color() : r(0), g(0), b(0), a(1) {}
    Color(float r, float g, float b, float a = 1.0f) : r(r), g(g), b(b), a(a) {}

    static const Color WHITE;
    static const Color BLACK;
    static const Color RED;
    static const Color GREEN;
    static const Color BLUE;
    static const Color YELLOW;
    static const Color TRANSPARENT;
};

inline const Color Color::WHITE(1, 1, 1, 1);
inline const Color Color::BLACK(0, 0, 0, 1);
inline const Color Color::RED(1, 0, 0, 1);
inline const Color Color::GREEN(0, 1, 0, 1);
inline const Color Color::BLUE(0, 0, 1, 1);
inline const Color Color::YELLOW(1, 1, 0, 1);
inline const Color Color::TRANSPARENT(0, 0, 0, 0);

class Image {
public:
    int get_width() const { return 0; }
    int get_height() const { return 0; }
    void set_size(int, int) {}
    void set_pixel(int, int, const Color&) {}
    Color get_pixel(int, int) const { return Color(0, 0, 0, 1); }
};

} // namespace MyEngine
