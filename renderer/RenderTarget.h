#pragma once

#include "core/math/Vector2.h"
#include "scene/resources/Texture2D.h"
#include <cstdint>

namespace MyEngine {

// 渲染目标
class RenderTarget {
public:
    RenderTarget();
    ~RenderTarget();

    // 创建
    bool create(int width, int height, bool has_depth = true, bool has_stencil = false);
    void destroy();

    // 尺寸
    int get_width() const { return _width; }
    int get_height() const { return _height; }
    Vector2 get_size() const { return Vector2((float)_width, (float)_height); }

    // 纹理
    Ref<Texture2D> get_color_texture() const { return _color_texture; }
    Ref<Texture2D> get_depth_texture() const { return _depth_texture; }

    // 帧缓冲区
    uint32_t get_framebuffer() const { return _framebuffer; }
    bool is_valid() const { return _framebuffer != 0; }

    // 清除
    void clear(const Color& color, float depth = 1.0f, int stencil = 0);

    // 绑定/解绑
    void bind();
    void unbind();

    // 读取像素
    Color read_pixel(int x, int y);

    // 复制到另一个渲染目标或纹理
    void blit_to(RenderTarget* dest, bool color = true, bool depth = false);

private:
    int _width = 0;
    int _height = 0;
    uint32_t _framebuffer = 0;
    uint32_t _depth_renderbuffer = 0;
    Ref<Texture2D> _color_texture;
    Ref<Texture2D> _depth_texture;
};

} // namespace MyEngine
