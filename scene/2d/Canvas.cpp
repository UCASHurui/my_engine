#include "Canvas.h"
#include "core/os/OS.h"
#include "renderer/RenderDevice.h"

namespace MyEngine {

Ref<Texture2D> Canvas::_default_texture;

Canvas::Canvas() = default;

Canvas::~Canvas() {
    auto rd = RenderDevice::get_singleton();
    if (rd) {
        if (_vertex_buffer) rd->delete_buffer(_vertex_buffer);
        if (_index_buffer) rd->delete_buffer(_index_buffer);
    }
}

bool Canvas::initialize(int max_sprites) {
    _max_sprites_per_batch = max_sprites;
    _transform = Matrix4::identity();

    // 创建正交投影矩阵
    _projection = Matrix4::orthographic(0, 1280, 720, 0, -1, 1);

    return _initialize_buffers();
}

bool Canvas::_initialize_buffers() {
    auto rd = RenderDevice::get_singleton();
    if (!rd) return false;

    // 创建顶点缓冲区
    size_t vertex_size = _max_sprites_per_batch * 4 * sizeof(SpriteVertex);
    _vertex_buffer = rd->create_buffer(
        RenderDevice::VERTEX_BUFFER,
        RenderDevice::DYNAMIC_DRAW,
        vertex_size,
        nullptr
    );

    // 创建索引缓冲区
    size_t index_size = _max_sprites_per_batch * 6 * sizeof(uint16_t);
    _index_buffer = rd->create_buffer(
        RenderDevice::INDEX_BUFFER,
        RenderDevice::DYNAMIC_DRAW,
        index_size,
        nullptr
    );

    // 预生成索引
    Vector<uint16_t> indices;
    indices.resize(_max_sprites_per_batch * 6);
    for (int i = 0; i < _max_sprites_per_batch; i++) {
        uint16_t base = i * 4;
        indices[i * 6 + 0] = base + 0;
        indices[i * 6 + 1] = base + 1;
        indices[i * 6 + 2] = base + 2;
        indices[i * 6 + 3] = base + 1;
        indices[i * 6 + 4] = base + 3;
        indices[i * 6 + 5] = base + 2;
    }

    rd->bind_buffer(RenderDevice::INDEX_BUFFER, _index_buffer);
    rd->update_buffer(_index_buffer, 0, index_size, indices.data());
    rd->bind_buffer(RenderDevice::INDEX_BUFFER, 0);

    return _vertex_buffer != 0 && _index_buffer != 0;
}

void Canvas::begin() {
    _batches.clear();
    _current_batch = nullptr;
}

void Canvas::end() {
    flush();
}

void Canvas::draw_rect(const Vector2& pos, const Vector2& size, const Color& color) {
    draw_texture(nullptr, pos, size, Vector2::ZERO, Vector2::ONE, color);
}

void Canvas::draw_rect_outline(const Vector2& pos, const Vector2& size,
                                const Color& color, float line_width) {
    float w = size.x;
    float h = size.y;

    // 顶部
    draw_line(pos, Vector2(pos.x + w, pos.y), color, line_width);
    // 右侧
    draw_line(Vector2(pos.x + w, pos.y), Vector2(pos.x + w, pos.y + h), color, line_width);
    // 底部
    draw_line(Vector2(pos.x + w, pos.y + h), Vector2(pos.x, pos.y + h), color, line_width);
    // 左侧
    draw_line(Vector2(pos.x, pos.y + h), pos, color, line_width);
}

void Canvas::draw_circle(const Vector2& center, float radius, const Color& color, int segments) {
    if (segments < 3) segments = 3;

    Vector2 prev = center + Vector2(radius, 0);
    for (int i = 1; i <= segments; i++) {
        float angle = (float)i / segments * Math::TWO_PI;
        Vector2 next = center + Vector2(Math::cos(angle) * radius, Math::sin(angle) * radius);
        draw_line(prev, next, color);
        prev = next;
    }
}

void Canvas::draw_line(const Vector2& from, const Vector2& to, const Color& color, float width) {
    (void)width;
    Vector2 delta = to - from;
    Vector2 normal(-delta.y, delta.x);
    normal = normal.normalized() * width * 0.5f;

    SpriteVertex verts[4];
    verts[0].position = Vector3(from.x - normal.x, from.y - normal.y, 0);
    verts[1].position = Vector3(from.x + normal.x, from.y + normal.y, 0);
    verts[2].position = Vector3(to.x + normal.x, to.y + normal.y, 0);
    verts[3].position = Vector3(to.x - normal.x, to.y - normal.y, 0);

    for (int i = 0; i < 4; i++) {
        verts[i].uv = Vector2::ZERO;
        verts[i].color = color;
    }

    _new_batch(nullptr);
    SpriteBatch& batch = *_current_batch;
    uint16_t base = batch.vertices.size();
    batch.vertices.push_back(verts[0]);
    batch.vertices.push_back(verts[1]);
    batch.vertices.push_back(verts[2]);
    batch.vertices.push_back(verts[3]);

    batch.indices.push_back(base + 0);
    batch.indices.push_back(base + 1);
    batch.indices.push_back(base + 2);
    batch.indices.push_back(base + 1);
    batch.indices.push_back(base + 3);
    batch.indices.push_back(base + 2);
}

void Canvas::draw_texture(Texture2D* texture, const Vector2& pos) {
    draw_texture(texture, pos, Vector2(100, 100), Vector2::ZERO, Vector2::ONE, Color(1, 1, 1, 1));
}

void Canvas::draw_texture(Texture2D* texture, const Vector2& pos, const Vector2& size) {
    draw_texture(texture, pos, size, Vector2::ZERO, Vector2::ONE, Color(1, 1, 1, 1));
}

void Canvas::draw_texture(Texture2D* texture, const Vector2& pos, const Vector2& size,
                           const Vector2& uv_pos, const Vector2& uv_size) {
    draw_texture(texture, pos, size, uv_pos, uv_size, Color(1, 1, 1, 1));
}

void Canvas::draw_texture(Texture2D* texture, const Vector2& pos, const Vector2& size,
                           const Vector2& uv_pos, const Vector2& uv_size, const Color& modulate) {
    _new_batch(texture);
    _add_sprite(pos, size, uv_pos, uv_size, modulate);
}

void Canvas::set_texture(Texture2D* texture) {
    if (!_current_batch || _current_batch->texture != texture) {
        _new_batch(texture);
    }
}

void Canvas::set_default_texture(Texture2D* texture) {
    _default_texture = texture;
}

void Canvas::clear(const Color& color) {
    auto rd = RenderDevice::get_singleton();
    if (rd) {
        rd->clear(RenderDevice::COLOR_BIT, color, 1.0f, 0);
    }
}

void Canvas::flush() {
    if (_batches.empty()) return;

    auto rd = RenderDevice::get_singleton();
    if (!rd) return;

    rd->set_blend_mode(RenderDevice::BLEND_MIX);
    rd->set_depth_test(false);
    rd->set_cull_mode(RenderDevice::CULL_NONE);

    for (auto& batch : _batches) {
        if (batch.vertices.empty()) continue;

        // 绑定纹理
        if (batch.texture && batch.texture->get_handle()) {
            rd->set_texture(RenderDevice::TEX_UNIT_0, batch.texture->get_handle());
        } else if (_default_texture) {
            rd->set_texture(RenderDevice::TEX_UNIT_0, _default_texture->get_handle());
        }

        // 更新顶点数据
        size_t vertex_size = batch.vertices.size() * sizeof(SpriteVertex);
        rd->bind_buffer(RenderDevice::VERTEX_BUFFER, _vertex_buffer);
        rd->update_buffer(_vertex_buffer, 0, vertex_size, batch.vertices.data());

        // 更新索引数据
        size_t index_size = batch.indices.size() * sizeof(uint16_t);
        rd->bind_buffer(RenderDevice::INDEX_BUFFER, _index_buffer);
        rd->update_buffer(_index_buffer, 0, index_size, batch.indices.data());

        // 绘制
        rd->draw_elements(RenderDevice::TRIANGLES, batch.indices.size(),
                          RenderDevice::INDEX_16BIT, nullptr);
    }

    rd->bind_buffer(RenderDevice::VERTEX_BUFFER, 0);
    rd->bind_buffer(RenderDevice::INDEX_BUFFER, 0);
    rd->set_texture(RenderDevice::TEX_UNIT_0, 0);

    _batches.clear();
    _current_batch = nullptr;
}

void Canvas::_new_batch(Texture2D* texture) {
    if (_current_batch && _current_batch->texture == texture &&
        _current_batch->vertices.size() < (size_t)_max_sprites_per_batch * 4) {
        return;
    }

    flush();

    SpriteBatch batch;
    batch.texture = texture;
    _batches.push_back(batch);
    _current_batch = &_batches.back();
}

void Canvas::_add_sprite(const Vector2& pos, const Vector2& size,
                          const Vector2& uv_pos, const Vector2& uv_size,
                          const Color& color) {
    SpriteVertex verts[4];

    verts[0].position = Vector3(pos.x, pos.y, 0);
    verts[0].uv = uv_pos;
    verts[0].color = color;

    verts[1].position = Vector3(pos.x, pos.y + size.y, 0);
    verts[1].uv = Vector2(uv_pos.x, uv_pos.y + uv_size.y);
    verts[1].color = color;

    verts[2].position = Vector3(pos.x + size.x, pos.y + size.y, 0);
    verts[2].uv = Vector2(uv_pos.x + uv_size.x, uv_pos.y + uv_size.y);
    verts[2].color = color;

    verts[3].position = Vector3(pos.x + size.x, pos.y, 0);
    verts[3].uv = Vector2(uv_pos.x + uv_size.x, uv_pos.y);
    verts[3].color = color;

    SpriteBatch& batch = *_current_batch;
    uint16_t base = batch.vertices.size();
    batch.vertices.push_back(verts[0]);
    batch.vertices.push_back(verts[1]);
    batch.vertices.push_back(verts[2]);
    batch.vertices.push_back(verts[3]);

    batch.indices.push_back(base + 0);
    batch.indices.push_back(base + 1);
    batch.indices.push_back(base + 2);
    batch.indices.push_back(base + 1);
    batch.indices.push_back(base + 3);
    batch.indices.push_back(base + 2);
}

} // namespace MyEngine
