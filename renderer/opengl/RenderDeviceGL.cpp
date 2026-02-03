#include "RenderDeviceGL.h"
#include "os/OS.h"

namespace MyEngine {

RenderDeviceGL::RenderDeviceGL() = default;

RenderDeviceGL::~RenderDeviceGL() {
    shutdown();
}

bool RenderDeviceGL::initialize(void*, int width, int height) {
    _viewport_width = width;
    _viewport_height = height;
    OS::print("OpenGL Stub Renderer initialized");
    return true;
}

void RenderDeviceGL::shutdown() {
    _textures.clear();
    _buffers.clear();
    _shaders.clear();
    _programs.clear();
    _framebuffers.clear();
}

void RenderDeviceGL::resize(int width, int height) {
    _viewport_width = width;
    _viewport_height = height;
}

void RenderDeviceGL::clear(ColorFlag, const Color&, float, int) {}

void RenderDeviceGL::draw_arrays(PrimitiveMode, int, int) {}

void RenderDeviceGL::draw_elements(PrimitiveMode, int, IndexType, const void*) {}

void RenderDeviceGL::set_viewport(int x, int y, int width, int height) {
    _viewport_x = x;
    _viewport_y = y;
    _viewport_width = width;
    _viewport_height = height;
}

void RenderDeviceGL::set_scissor(int, int, int, int) {}

void RenderDeviceGL::set_polygon_offset(float, float) {}

void RenderDeviceGL::set_blend_mode(BlendMode) {}

void RenderDeviceGL::set_blend_func(BlendFunc, BlendFunc) {}

void RenderDeviceGL::set_depth_test(bool) {}

void RenderDeviceGL::set_depth_write(bool) {}

void RenderDeviceGL::set_depth_func(CompareFunc) {}

void RenderDeviceGL::set_cull_mode(CullMode) {}

RenderDevice::TextureID RenderDeviceGL::create_texture(TextureType, PixelFormat, int, int, int, int) {
    TextureID id = _next_texture_id++;
    _textures[id] = nullptr;
    return id;
}

void RenderDeviceGL::update_texture(TextureID, int, int, int, int, int, int, int, PixelFormat, PixelType, const void*) {}

void RenderDeviceGL::set_texture(TextureUnit, TextureID) {}

void RenderDeviceGL::delete_texture(TextureID id) {
    _textures.erase(id);
}

RenderDevice::BufferID RenderDeviceGL::create_buffer(BufferType, BufferUsage, size_t, const void*) {
    BufferID id = _next_buffer_id++;
    _buffers[id] = nullptr;
    return id;
}

void RenderDeviceGL::update_buffer(BufferID, size_t, size_t, const void*) {}

void RenderDeviceGL::bind_buffer(BufferType, BufferID) {}

void RenderDeviceGL::delete_buffer(BufferID id) {
    _buffers.erase(id);
}

RenderDevice::ShaderID RenderDeviceGL::create_shader(ShaderType, const char*) {
    ShaderID id = _next_shader_id++;
    _shaders[id] = nullptr;
    return id;
}

RenderDevice::ShaderProgramID RenderDeviceGL::create_program(ShaderID, ShaderID) {
    ShaderProgramID id = _next_program_id++;
    _programs[id] = nullptr;
    return id;
}

void RenderDeviceGL::set_shader_program(ShaderProgramID) {}

void RenderDeviceGL::set_shader_uniform(ShaderProgramID, const String&, const Variant&) {}

void RenderDeviceGL::delete_shader(ShaderID id) {
    _shaders.erase(id);
}

void RenderDeviceGL::delete_program(ShaderProgramID id) {
    _programs.erase(id);
}

RenderDevice::FrameBufferID RenderDeviceGL::create_framebuffer() {
    FrameBufferID id = _next_framebuffer_id++;
    _framebuffers[id] = nullptr;
    return id;
}

void RenderDeviceGL::framebuffer_attach(FrameBufferID, FrameBufferAttachment, TextureID, int) {}

void RenderDeviceGL::bind_framebuffer(FrameBufferID) {}

void RenderDeviceGL::delete_framebuffer(FrameBufferID id) {
    _framebuffers.erase(id);
}

bool RenderDeviceGL::is_framebuffer_complete(FrameBufferID) {
    return true;
}

void RenderDeviceGL::flush() {}

void RenderDeviceGL::finish() {}

void RenderDeviceGL::swap_buffers() {}

void RenderDeviceGL::screenshot(Image&) {}

} // namespace MyEngine
