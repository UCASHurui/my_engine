#include "Texture2D.h"
#include "os/OS.h"
#include "io/FileAccess.h"
#include "renderer/RenderDevice.h"
#include <cstring>

// 简单的 PNG 读取 (仅支持未压缩的 8-bit RGB/RGBA)
#include <cstdio>

namespace MyEngine {

Texture2D::Texture2D() = default;

Texture2D::~Texture2D() {
    auto rd = RenderDevice::get_singleton();
    if (rd && _texture_id) {
        rd->delete_texture(_texture_id);
    }
}

bool Texture2D::create(int width, int height, PixelFormat format, const TextureParams& params) {
    _width = width;
    _height = height;
    _format = format;
    _params = params;

    return _allocate();
}

bool Texture2D::create_from_image(const Image& image, const TextureParams& params) {
    // 暂时跳过复杂实现
    (void)image;
    (void)params;
    OS::print_warning("Texture2D::create_from_image - Not fully implemented yet");
    return create(image.get_width(), image.get_height(), PF_RGBA, params);
}

bool Texture2D::create_from_data(int width, int height, PixelFormat format,
                                  const void* data, const TextureParams& params) {
    if (!create(width, height, format, params)) {
        return false;
    }
    update(data);
    return true;
}

void Texture2D::update(const void* data, int level) {
    auto rd = RenderDevice::get_singleton();
    if (!rd || !_texture_id) return;

    rd->update_texture(_texture_id, level, 0, 0, 0,
                       _width, _height, 1,
                       (RenderDevice::PixelFormat)_format, RenderDevice::PT_U8, data);
}

void Texture2D::update_region(int x, int y, int width, int height,
                               const void* data, int level) {
    auto rd = RenderDevice::get_singleton();
    if (!rd || !_texture_id) return;

    rd->update_texture(_texture_id, level, x, y, 0,
                       width, height, 1,
                       (RenderDevice::PixelFormat)_format, RenderDevice::PT_U8, data);
}

void Texture2D::set_filter(TextureFilter filter) {
    _params.min_filter = filter;
    _params.mag_filter = filter;
    _apply_params();
}

void Texture2D::set_wrap(const TextureWrap& wrap) {
    _params.wrap = wrap;
    _apply_params();
}

bool Texture2D::_allocate() {
    auto rd = RenderDevice::get_singleton();
    if (!rd) return false;

    if (_texture_id) {
        rd->delete_texture(_texture_id);
    }

    _texture_id = rd->create_texture(
        RenderDevice::TEXTURE_2D,
        (RenderDevice::PixelFormat)_format,
        _width, _height,
        1,
        _params.generate_mipmaps ? 4 : 1  // 简化处理
    );

    return _texture_id != 0;
}

void Texture2D::_apply_params() {
    // 简化实现 - 需要实际的 OpenGL 调用
}

Ref<Texture2D> Texture2D::load(const String& path) {
    Ref<Texture2D> tex = new Texture2D();

    // 简单实现 - 后续添加完整的文件读取
    OS::print_warning("Texture2D::load - File loading not fully implemented: " + path);

    return tex;
}

bool Texture2D::save_to_png(const String& path) const {
    (void)path;
    OS::print_warning("Texture2D::save_to_png - Not implemented");
    return false;
}

} // namespace MyEngine
