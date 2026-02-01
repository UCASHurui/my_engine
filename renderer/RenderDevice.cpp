#include "RenderDevice.h"

namespace MyEngine {

class NullRenderDevice : public RenderDevice {
public:
    NullRenderDevice();
    ~NullRenderDevice() override;

    bool initialize(void*, int, int) override;
    void shutdown() override;
    void resize(int, int) override;
    void clear(ColorFlag, const Color&, float, int) override;
    void draw_arrays(PrimitiveMode, int, int) override;
    void draw_elements(PrimitiveMode, int, IndexType, const void*) override;
    void set_viewport(int, int, int, int) override;
    void set_scissor(int, int, int, int) override;
    void set_polygon_offset(float, float) override;
    void set_blend_mode(BlendMode) override;
    void set_blend_func(BlendFunc, BlendFunc) override;
    void set_depth_test(bool) override;
    void set_depth_write(bool) override;
    void set_depth_func(CompareFunc) override;
    void set_cull_mode(CullMode) override;
    TextureID create_texture(TextureType, PixelFormat, int, int, int, int) override;
    void update_texture(TextureID, int, int, int, int, int, int, int, PixelFormat, PixelType, const void*) override;
    void set_texture(TextureUnit, TextureID) override;
    void delete_texture(TextureID) override;
    BufferID create_buffer(BufferType, BufferUsage, size_t, const void*) override;
    void update_buffer(BufferID, size_t, size_t, const void*) override;
    void bind_buffer(BufferType, BufferID) override;
    void delete_buffer(BufferID) override;
    ShaderID create_shader(ShaderType, const char*) override;
    ShaderProgramID create_program(ShaderID, ShaderID) override;
    void set_shader_program(ShaderProgramID) override;
    void set_shader_uniform(ShaderProgramID, const String&, const Variant&) override;
    void delete_shader(ShaderID) override;
    void delete_program(ShaderProgramID) override;
    FrameBufferID create_framebuffer() override;
    void framebuffer_attach(FrameBufferID, FrameBufferAttachment, TextureID, int) override;
    void bind_framebuffer(FrameBufferID) override;
    void delete_framebuffer(FrameBufferID) override;
    bool is_framebuffer_complete(FrameBufferID) override;
    void flush() override;
    void finish() override;
    String get_device_name() const override;
    String get_driver_version() const override;
    int get_max_texture_size() const override;
    int get_max_texture_units() const override;
    const char* get_error_string() const override;
    ErrorCode get_error() const override;
    void swap_buffers() override;
    void screenshot(Image&) override;
};

NullRenderDevice::NullRenderDevice() = default;
NullRenderDevice::~NullRenderDevice() = default;

bool NullRenderDevice::initialize(void*, int, int) { return true; }
void NullRenderDevice::shutdown() {}
void NullRenderDevice::resize(int, int) {}
void NullRenderDevice::clear(ColorFlag, const Color&, float, int) {}
void NullRenderDevice::draw_arrays(PrimitiveMode, int, int) {}
void NullRenderDevice::draw_elements(PrimitiveMode, int, IndexType, const void*) {}
void NullRenderDevice::set_viewport(int, int, int, int) {}
void NullRenderDevice::set_scissor(int, int, int, int) {}
void NullRenderDevice::set_polygon_offset(float, float) {}
void NullRenderDevice::set_blend_mode(BlendMode) {}
void NullRenderDevice::set_blend_func(BlendFunc, BlendFunc) {}
void NullRenderDevice::set_depth_test(bool) {}
void NullRenderDevice::set_depth_write(bool) {}
void NullRenderDevice::set_depth_func(CompareFunc) {}
void NullRenderDevice::set_cull_mode(CullMode) {}
RenderDevice::TextureID NullRenderDevice::create_texture(TextureType, PixelFormat, int, int, int, int) { return 0; }
void NullRenderDevice::update_texture(RenderDevice::TextureID, int, int, int, int, int, int, int, PixelFormat, PixelType, const void*) {}
void NullRenderDevice::set_texture(TextureUnit, RenderDevice::TextureID) {}
void NullRenderDevice::delete_texture(RenderDevice::TextureID) {}
RenderDevice::BufferID NullRenderDevice::create_buffer(BufferType, BufferUsage, size_t, const void*) { return 0; }
void NullRenderDevice::update_buffer(RenderDevice::BufferID, size_t, size_t, const void*) {}
void NullRenderDevice::bind_buffer(BufferType, RenderDevice::BufferID) {}
void NullRenderDevice::delete_buffer(RenderDevice::BufferID) {}
RenderDevice::ShaderID NullRenderDevice::create_shader(ShaderType, const char*) { return 0; }
RenderDevice::ShaderProgramID NullRenderDevice::create_program(RenderDevice::ShaderID, RenderDevice::ShaderID) { return 0; }
void NullRenderDevice::set_shader_program(RenderDevice::ShaderProgramID) {}
void NullRenderDevice::set_shader_uniform(RenderDevice::ShaderProgramID, const String&, const Variant&) {}
void NullRenderDevice::delete_shader(RenderDevice::ShaderID) {}
void NullRenderDevice::delete_program(RenderDevice::ShaderProgramID) {}
RenderDevice::FrameBufferID NullRenderDevice::create_framebuffer() { return 0; }
void NullRenderDevice::framebuffer_attach(RenderDevice::FrameBufferID, FrameBufferAttachment, RenderDevice::TextureID, int) {}
void NullRenderDevice::bind_framebuffer(RenderDevice::FrameBufferID) {}
void NullRenderDevice::delete_framebuffer(RenderDevice::FrameBufferID) {}
bool NullRenderDevice::is_framebuffer_complete(RenderDevice::FrameBufferID) { return true; }
void NullRenderDevice::flush() {}
void NullRenderDevice::finish() {}
String NullRenderDevice::get_device_name() const { return "NullDevice"; }
String NullRenderDevice::get_driver_version() const { return "0.0.0"; }
int NullRenderDevice::get_max_texture_size() const { return 64; }
int NullRenderDevice::get_max_texture_units() const { return 1; }
const char* NullRenderDevice::get_error_string() const { return "OK"; }
RenderDevice::ErrorCode NullRenderDevice::get_error() const { return ERR_OK; }
void NullRenderDevice::swap_buffers() {}
void NullRenderDevice::screenshot(Image&) {}

} // namespace MyEngine
