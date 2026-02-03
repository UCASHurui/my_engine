#pragma once

#include "containers/String.h"
#include "math/Vector2.h"
#include "variant/Variant.h"
#include "Image.h"
#include <cstdint>

namespace MyEngine {

// 前向声明
class RenderDevice;

// 渲染设备抽象
class RenderDevice {
public:
    virtual ~RenderDevice() = default;

    // 单例
    static RenderDevice* get_singleton() { return _singleton; }
    static void set_singleton(RenderDevice* device) { _singleton = device; }

    // 类型定义
    using TextureID = uint32_t;
    using BufferID = uint32_t;
    using ShaderID = uint32_t;
    using ShaderProgramID = uint32_t;
    using FrameBufferID = uint32_t;

    enum PrimitiveMode {
        POINTS,
        LINES,
        LINE_STRIP,
        LINE_LOOP,
        TRIANGLES,
        TRIANGLE_STRIP,
        TRIANGLE_FAN
    };

    enum IndexType {
        INDEX_8BIT,
        INDEX_16BIT,
        INDEX_32BIT
    };

    enum TextureType {
        TEXTURE_1D,
        TEXTURE_2D,
        TEXTURE_3D,
        TEXTURE_CUBE
    };

    enum TextureUnit {
        TEX_UNIT_0, TEX_UNIT_1, TEX_UNIT_2, TEX_UNIT_3,
        TEX_UNIT_4, TEX_UNIT_5, TEX_UNIT_6, TEX_UNIT_7,
        TEX_UNIT_MAX
    };

    enum BufferType {
        VERTEX_BUFFER,
        INDEX_BUFFER,
        UNIFORM_BUFFER
    };

    enum BufferUsage {
        STREAM_DRAW,
        STREAM_READ,
        STREAM_COPY,
        STATIC_DRAW,
        STATIC_READ,
        STATIC_COPY,
        DYNAMIC_DRAW,
        DYNAMIC_READ,
        DYNAMIC_COPY
    };

    enum ShaderType {
        VERTEX_SHADER,
        FRAGMENT_SHADER,
        GEOMETRY_SHADER
    };

    enum BlendMode {
        BLEND_NONE,
        BLEND_ADD,
        BLEND_SUB,
        BLEND_MIN,
        BLEND_MAX,
        BLEND_MIX
    };

    enum BlendFunc {
        BLEND_ZERO,
        BLEND_ONE,
        BLEND_SRC_ALPHA,
        BLEND_ONE_MINUS_SRC_ALPHA,
        BLEND_DST_ALPHA,
        BLEND_ONE_MINUS_DST_ALPHA
    };

    enum CompareFunc {
        COMPARE_NEVER,
        COMPARE_LESS,
        COMPARE_EQUAL,
        COMPARE_LEQUAL,
        COMPARE_GREATER,
        COMPARE_NOTEQUAL,
        COMPARE_GEQUAL,
        COMPARE_ALWAYS
    };

    enum CullMode {
        CULL_NONE,
        CULL_FRONT,
        CULL_BACK,
        CULL_BOTH
    };

    enum ColorFlag {
        COLOR_BIT = 1,
        DEPTH_BIT = 2,
        STENCIL_BIT = 4
    };

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

    enum PixelType {
        PT_U8,
        PT_I8,
        PT_U16,
        PT_I16,
        PT_U32,
        PT_I32,
        PT_F16,
        PT_F32
    };

    enum FrameBufferAttachment {
        FB_COLOR,
        FB_DEPTH,
        FB_STENCIL,
        FB_DEPTH_STENCIL
    };

    enum ErrorCode {
        ERR_OK,
        ERR_INVALID_ENUM,
        ERR_INVALID_VALUE,
        ERR_INVALID_OPERATION,
        ERR_OUT_OF_MEMORY,
        ERR_INVALID_FRAMEBUFFER
    };

    // 初始化/销毁
    virtual bool initialize(void* window, int width, int height) = 0;
    virtual void shutdown() = 0;
    virtual void resize(int width, int height) = 0;

    // 清除
    virtual void clear(ColorFlag flags, const Color& color, float depth, int stencil) = 0;

    // 绘制
    virtual void draw_arrays(PrimitiveMode mode, int first, int count) = 0;
    virtual void draw_elements(PrimitiveMode mode, int count, IndexType type, const void* indices) = 0;

    // 状态
    virtual void set_viewport(int x, int y, int width, int height) = 0;
    virtual void set_scissor(int x, int y, int width, int height) = 0;
    virtual void set_polygon_offset(float factor, float units) = 0;

    // 混合
    virtual void set_blend_mode(BlendMode mode) = 0;
    virtual void set_blend_func(BlendFunc src, BlendFunc dst) = 0;

    // 深度
    virtual void set_depth_test(bool enable) = 0;
    virtual void set_depth_write(bool enable) = 0;
    virtual void set_depth_func(CompareFunc func) = 0;

    // 背面剔除
    virtual void set_cull_mode(CullMode mode) = 0;

    // 纹理
    virtual TextureID create_texture(TextureType type, PixelFormat format, int width, int height,
                                     int depth = 1, int mipmaps = 1) = 0;
    virtual void update_texture(TextureID id, int level, int x, int y, int z, int w, int h, int d,
                                PixelFormat format, PixelType type, const void* data) = 0;
    virtual void set_texture(TextureUnit unit, TextureID id) = 0;
    virtual void delete_texture(TextureID id) = 0;

    // 缓冲区
    virtual BufferID create_buffer(BufferType type, BufferUsage usage, size_t size, const void* data = nullptr) = 0;
    virtual void update_buffer(BufferID id, size_t offset, size_t size, const void* data) = 0;
    virtual void bind_buffer(BufferType type, BufferID id) = 0;
    virtual void delete_buffer(BufferID id) = 0;

    // 着色器
    virtual ShaderID create_shader(ShaderType type, const char* source) = 0;
    virtual ShaderProgramID create_program(ShaderID vs, ShaderID fs) = 0;
    virtual void set_shader_program(ShaderProgramID id) = 0;
    virtual void set_shader_uniform(ShaderProgramID id, const String& name, const Variant& value) = 0;
    virtual void delete_shader(ShaderID id) = 0;
    virtual void delete_program(ShaderProgramID id) = 0;

    // 帧缓冲
    virtual FrameBufferID create_framebuffer() = 0;
    virtual void framebuffer_attach(FrameBufferID fb, FrameBufferAttachment type, TextureID tex, int level = 0) = 0;
    virtual void bind_framebuffer(FrameBufferID fb) = 0;
    virtual void delete_framebuffer(FrameBufferID fb) = 0;

    // 查询
    virtual bool is_framebuffer_complete(FrameBufferID fb) = 0;

    // 同步
    virtual void flush() = 0;
    virtual void finish() = 0;

    // 信息
    virtual String get_device_name() const = 0;
    virtual String get_driver_version() const = 0;
    virtual int get_max_texture_size() const = 0;
    virtual int get_max_texture_units() const = 0;

    // 错误检查
    virtual const char* get_error_string() const = 0;
    virtual ErrorCode get_error() const = 0;

    // 交换链
    virtual void swap_buffers() = 0;

    // 截图
    virtual void screenshot(Image& image) = 0;

protected:
    RenderDevice() = default;
    static RenderDevice* _singleton;

public:
    // 禁止复制
    RenderDevice(const RenderDevice&) = delete;
    RenderDevice& operator=(const RenderDevice&) = delete;
};

} // namespace MyEngine
