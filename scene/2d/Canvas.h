#pragma once

#include "core/math/Vector2.h"
#include "core/math/Vector3.h"
#include "core/math/Matrix4.h"
#include "core/math/Math.h"
#include "core/containers/Vector.h"
#include "core/object/RefCounted.h"
#include "scene/resources/Texture2D.h"
#include "scene/resources/Mesh.h"
#include "renderer/RenderDevice.h"

namespace MyEngine {

// 2D 画布 - 用于高效批量渲染 2D 内容
class Canvas {
public:
    Canvas();
    ~Canvas();

    // 初始化
    bool initialize(int max_sprites = 1024);

    // 准备绘制
    void begin();
    void end();

    // 绘制矩形
    void draw_rect(const Vector2& pos, const Vector2& size, const Color& color);
    void draw_rect_outline(const Vector2& pos, const Vector2& size, const Color& color, float line_width = 1.0f);

    // 绘制圆形
    void draw_circle(const Vector2& center, float radius, const Color& color, int segments = 32);

    // 绘制线段
    void draw_line(const Vector2& from, const Vector2& to, const Color& color, float width = 1.0f);

    // 绘制纹理
    void draw_texture(Texture2D* texture, const Vector2& pos);
    void draw_texture(Texture2D* texture, const Vector2& pos, const Vector2& size);
    void draw_texture(Texture2D* texture, const Vector2& pos, const Vector2& size,
                      const Vector2& uv_pos, const Vector2& uv_size);
    void draw_texture(Texture2D* texture, const Vector2& pos, const Vector2& size,
                      const Vector2& uv_pos, const Vector2& uv_size, const Color& modulate);

    // 设置当前纹理
    void set_texture(Texture2D* texture);

    // 设置默认纹理
    static void set_default_texture(Texture2D* texture);

    // 清屏
    void clear(const Color& color = Color(0, 0, 0, 1));

    // 提交到 GPU
    void flush();

private:
    struct SpriteVertex {
        Vector3 position;
        Vector2 uv;
        Color color;
    };

    struct SpriteBatch {
        Texture2D* texture = nullptr;
        Vector<SpriteVertex> vertices;
        Vector<uint16_t> indices;
    };

    Vector<SpriteBatch> _batches;
    SpriteBatch* _current_batch = nullptr;
    int _max_sprites_per_batch = 1024;

    Matrix4 _transform;
    Matrix4 _projection;

    static Ref<Texture2D> _default_texture;

    Ref<Mesh> _mesh;
    RenderDevice::BufferID _vertex_buffer = 0;
    RenderDevice::BufferID _index_buffer = 0;

    bool _initialize_buffers();
    void _new_batch(Texture2D* texture);
    void _add_sprite(const Vector2& pos, const Vector2& size,
                     const Vector2& uv_pos, const Vector2& uv_size,
                     const Color& color);
};

} // namespace MyEngine
