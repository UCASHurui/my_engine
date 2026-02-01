#pragma once

#include "core/math/Vector4.h"

namespace MyEngine {

// 使用 Vector4 作为颜色 (r, g, b, a)
using Color = Vector4;

class Image {
public:
    int get_width() const { return 0; }
    int get_height() const { return 0; }
    void set_size(int, int) {}
    void set_pixel(int, int, const Color&) {}
    Color get_pixel(int, int) const { return Color(0, 0, 0, 1); }
};

} // namespace MyEngine
