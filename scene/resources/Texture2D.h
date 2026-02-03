#pragma once

#include "containers/String.h"
#include "math/Vector2.h"
#include "object/RefCounted.h"
#include <cstdint>

namespace MyEngine {

// Forward declarations to avoid circular dependencies
class Image;
class RenderDevice;

// Texture pixel format enum (defined here to avoid renderer dependency)
enum TexturePixelFormat {
    TEX_PF_RGBA,
    TEX_PF_RGB,
    TEX_PF_RGBA16F,
    TEX_PF_RGBA32F,
    TEX_PF_DEPTH
};

// Texture ID type
using TextureID = uint32_t;

// 纹理过滤模式
enum class TextureFilter {
    NEAREST,
    LINEAR,
    NEAREST_MIPMAP_NEAREST,
    LINEAR_MIPMAP_NEAREST,
    NEAREST_MIPMAP_LINEAR,
    LINEAR_MIPMAP_LINEAR
};

// 纹理重复模式
enum class TextureRepeat {
    REPEAT,
    CLAMP_TO_EDGE,
    CLAMP_TO_BORDER,
    MIRRORED_REPEAT,
    MIRROR_CLAMP_TO_EDGE
};

// 纹理包装
struct TextureWrap {
    TextureRepeat u = TextureRepeat::REPEAT;
    TextureRepeat v = TextureRepeat::REPEAT;
    TextureRepeat w = TextureRepeat::REPEAT;
};

// 纹理参数
struct TextureParams {
    TextureFilter min_filter = TextureFilter::LINEAR_MIPMAP_LINEAR;
    TextureFilter mag_filter = TextureFilter::LINEAR;
    TextureWrap wrap;
    float anisotropy = 1.0f;
    bool generate_mipmaps = true;
};

// 纹理资源
class Texture2D : public RefCounted {
public:
    Texture2D();
    ~Texture2D() override;

    // 创建空纹理
    bool create(int width, int height, TexturePixelFormat format = TexturePixelFormat::TEX_PF_RGBA,
                const TextureParams& params = TextureParams());

    // 从图像创建
    bool create_from_image(const Image& image, const TextureParams& params = TextureParams());

    // 从数据创建
    bool create_from_data(int width, int height, TexturePixelFormat format,
                          const void* data, const TextureParams& params = TextureParams());

    // 更新纹理数据
    void update(const void* data, int level = 0);
    void update_region(int x, int y, int width, int height,
                       const void* data, int level = 0);

    // 设置参数
    void set_filter(TextureFilter filter);
    void set_wrap(const TextureWrap& wrap);

    // 获取属性
    int get_width() const { return _width; }
    int get_height() const { return _height; }
    Vector2 get_size() const { return Vector2((float)_width, (float)_height); }
    TexturePixelFormat get_format() const { return _format; }

    // 资源句柄
    TextureID get_handle() const { return _texture_id; }

    // 加载纹理 (从文件)
    static Ref<Texture2D> load(const String& path);

    // 保存纹理 (到文件)
    bool save_to_png(const String& path) const;

private:
    int _width = 0;
    int _height = 0;
    TexturePixelFormat _format = TexturePixelFormat::TEX_PF_RGBA;
    TextureID _texture_id = 0;
    TextureParams _params;

    bool _allocate();
    void _apply_params();
};

} // namespace MyEngine
