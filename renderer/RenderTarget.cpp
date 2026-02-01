#include "RenderTarget.h"
#include "RenderDevice.h"

namespace MyEngine {

RenderTarget::RenderTarget() = default;

RenderTarget::~RenderTarget() {
    destroy();
}

bool RenderTarget::create(int width, int height, bool has_depth, bool has_stencil) {
    (void)width; (void)height; (void)has_depth; (void)has_stencil;
    // 简化实现
    _width = width;
    _height = height;
    return true;
}

void RenderTarget::destroy() {
    _framebuffer = 0;
    _width = 0;
    _height = 0;
}

void RenderTarget::clear(const Color& color, float depth, int stencil) {
    (void)color; (void)depth; (void)stencil;
}

void RenderTarget::bind() {
}

void RenderTarget::unbind() {
}

Color RenderTarget::read_pixel(int x, int y) {
    (void)x; (void)y;
    return Color::BLACK();
}

void RenderTarget::blit_to(RenderTarget* dest, bool color, bool depth) {
    (void)dest; (void)color; (void)depth;
}

} // namespace MyEngine
