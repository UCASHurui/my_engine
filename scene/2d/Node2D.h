#pragma once

#include "core/math/Vector2.h"
#include "core/math/Rect2.h"
#include "core/math/Transform.h"
#include "scene/main/Node.h"

namespace MyEngine {

// 颜色
struct Color {
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;

    Color() = default;
    Color(float r, float g, float b, float a = 1.0f) : r(r), g(g), b(b), a(a) {}

    static const Color WHITE;
    static const Color BLACK;
    static const Color RED;
    static const Color GREEN;
    static const Color BLUE;
    static const Color YELLOW;
    static const Color TRANSPARENT;

    Color operator*(const Color& other) const {
        return Color(r * other.r, g * other.g, b * other.b, a * other.a);
    }
};

// 颜色常量定义
inline const Color Color::WHITE(1.0f, 1.0f, 1.0f, 1.0f);
inline const Color Color::BLACK(0.0f, 0.0f, 0.0f, 1.0f);
inline const Color Color::RED(1.0f, 0.0f, 0.0f, 1.0f);
inline const Color Color::GREEN(0.0f, 1.0f, 0.0f, 1.0f);
inline const Color Color::BLUE(0.0f, 0.0f, 1.0f, 1.0f);
inline const Color Color::YELLOW(1.0f, 1.0f, 0.0f, 1.0f);
inline const Color Color::TRANSPARENT(0.0f, 0.0f, 0.0f, 0.0f);

// 2D 节点基类
class Node2D : public Node {

public:
    Node2D();
    virtual ~Node2D();

    virtual const char* get_class_name() const override { return "Node2D"; }

    // 变换
    void set_position(const Vector2& pos) { _position = pos; }
    Vector2 get_position() const { return _position; }
    void set_rotation(float radians) { _rotation = radians; }
    float get_rotation() const { return _rotation; }
    void set_rotation_degrees(float degrees) { _rotation = degrees * 0.0174533f; }
    float get_rotation_degrees() const { return _rotation * 57.2958f; }
    void set_scale(const Vector2& scale) { _scale = scale; }
    Vector2 get_scale() const { return _scale; }
    void set_z_index(int z) { _z_index = z; }
    int get_z_index() const { return _z_index; }
    void set_z_as_relative(bool relative) { _z_as_relative = relative; }
    bool is_z_as_relative() const { return _z_as_relative; }

    // 变换操作
    void translate(const Vector2& offset) { _position = _position + offset; }
    void rotate(float radians) { _rotation += radians; }
    void rotate_degrees(float degrees) { _rotation += degrees * 0.0174533f; }
    void scale(const Vector2& factor) { _scale = _scale * factor; }

    // 全局变换
    void set_global_position(const Vector2& pos);
    Vector2 get_global_position() const;
    void set_global_rotation(float radians);
    float get_global_rotation() const;
    void set_global_scale(const Vector2& scale);
    Vector2 get_global_scale() const;

    // 变换计算
    Transform2D get_transform() const;
    Transform2D get_global_transform() const;
    Transform2D get_relative_transform() const;

    // 可见性
    void set_visible(bool visible) { _visible = visible; }
    bool is_visible() const { return _visible; }
    void set_modulate(const Color& color) { _modulate = color; }
    Color get_modulate() const { return _modulate; }
    void set_global_modulate(const Color& color) { _global_modulate = color; }
    Color get_global_modulate() const { return _global_modulate; }

    // 绘制
    virtual void _draw();

    // 绘制回调
    void draw_line(const Vector2& from, const Vector2& to, const Color& color, float width = 1.0f);
    void draw_rect(const Rect2& rect, const Color& color, bool filled = true, float width = 1.0f);
    void draw_circle(const Vector2& center, float radius, const Color& color);
    void draw_texture(class Texture2D* texture, const Vector2& pos, const Color& modulate = Color::WHITE);

protected:
    Vector2 _position;
    float _rotation = 0.0f;
    Vector2 _scale = Vector2::ONE;
    int _z_index = 0;
    bool _z_as_relative = true;
    bool _visible = true;
    Color _modulate = Color::WHITE;
    Color _global_modulate = Color::WHITE;
};

} // namespace MyEngine
