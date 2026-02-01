#pragma once

#include "../RenderDevice.h"
#include <vector>
#include <unordered_map>
#include <cstring>

namespace MyEngine {

// 简化的 OpenGL 存根实现 (用于无头环境)
class RenderDeviceGL : public RenderDevice {
public:
    RenderDeviceGL();
    ~RenderDeviceGL() override;

    bool initialize(void* window, int width, int height) override;
    void shutdown() override;
    void resize(int width, int height) override;

    void clear(ColorFlag flags, const Color& color, float depth, int stencil) override;

    void draw_arrays(PrimitiveMode mode, int first, int count) override;
    void draw_elements(PrimitiveMode mode, int count, IndexType type, const void* indices) override;

    void set_viewport(int x, int y, int width, int height) override;
    void set_scissor(int x, int y, int width, int height) override;
    void set_polygon_offset(float factor, float units) override;

    void set_blend_mode(BlendMode mode) override;
    void set_blend_func(BlendFunc src, BlendFunc dst) override;

    void set_depth_test(bool enable) override;
    void set_depth_write(bool enable) override;
    void set_depth_func(CompareFunc func) override;

    void set_cull_mode(CullMode mode) override;

    TextureID create_texture(TextureType type, PixelFormat format, int width, int height,
                             int depth = 1, int mipmaps = 1) override;
    void update_texture(TextureID id, int level, int x, int y, int z, int w, int h, int d,
                        PixelFormat format, PixelType type, const void* data) override;
    void set_texture(TextureUnit unit, TextureID id) override;
    void delete_texture(TextureID id) override;

    BufferID create_buffer(BufferType type, BufferUsage usage, size_t size, const void* data = nullptr) override;
    void update_buffer(BufferID id, size_t offset, size_t size, const void* data) override;
    void bind_buffer(BufferType type, BufferID id) override;
    void delete_buffer(BufferID id) override;

    ShaderID create_shader(ShaderType type, const char* source) override;
    ShaderProgramID create_program(ShaderID vs, ShaderID fs) override;
    void set_shader_program(ShaderProgramID id) override;
    void set_shader_uniform(ShaderProgramID id, const String& name, const Variant& value) override;
    void delete_shader(ShaderID id) override;
    void delete_program(ShaderProgramID id) override;

    FrameBufferID create_framebuffer() override;
    void framebuffer_attach(FrameBufferID fb, FrameBufferAttachment type, TextureID tex, int level = 0) override;
    void bind_framebuffer(FrameBufferID fb) override;
    void delete_framebuffer(FrameBufferID fb) override;

    bool is_framebuffer_complete(FrameBufferID fb) override;

    void flush() override;
    void finish() override;

    String get_device_name() const override { return "OpenGL Stub"; }
    String get_driver_version() const override { return "0.0.0"; }
    int get_max_texture_size() const override { return 2048; }
    int get_max_texture_units() const override { return 4; }

    const char* get_error_string() const override { return "OK"; }
    ErrorCode get_error() const override { return ERR_OK; }

    void swap_buffers() override;
    void screenshot(Image& image) override;

private:
    int _viewport_x = 0;
    int _viewport_y = 0;
    int _viewport_width = 1280;
    int _viewport_height = 720;

    // 简化的资源管理
    TextureID _next_texture_id = 1;
    BufferID _next_buffer_id = 1;
    ShaderID _next_shader_id = 1;
    ShaderProgramID _next_program_id = 1;
    FrameBufferID _next_framebuffer_id = 1;

    std::unordered_map<TextureID, void*> _textures;
    std::unordered_map<BufferID, void*> _buffers;
    std::unordered_map<ShaderID, void*> _shaders;
    std::unordered_map<ShaderProgramID, void*> _programs;
    std::unordered_map<FrameBufferID, void*> _framebuffers;
};

} // namespace MyEngine
