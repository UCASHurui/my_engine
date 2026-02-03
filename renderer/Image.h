#pragma once

#include "math/Vector4.h"
#include "math/Color.h"

namespace MyEngine {

// 像素格式
enum PixelFormat {
    PF_R,
    PF_RG,
    PF_RGB,
    PF_RGBA,
    PF_DEPTH,
    PF_DEPTH_STENCIL,
    PF_LUMINANCE,
    PF_LUMINANCE_ALPHA
};

class Image {
public:
    int get_width() const { return 0; }
    int get_height() const { return 0; }
    void set_size(int, int) {}
    void set_pixel(int, int, const Color&) {}
    Color get_pixel(int, int) const { return Color(0, 0, 0, 1); }
};

} // namespace MyEngine
