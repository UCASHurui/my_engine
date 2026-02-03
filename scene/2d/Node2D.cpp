#include "Node2D.h"
#include "math/Transform.h"

namespace MyEngine {

Node2D::Node2D() {}

Node2D::~Node2D() {}

void Node2D::set_global_position(const Vector2& pos) {
    Node* parent = static_cast<Node*>(get_parent());
    if (parent) {
        Transform2D parent_global = parent->get_global_transform_2d();
        Transform2D inv = parent_global.affine_inverse();
        _position = inv * pos;
    } else {
        _position = pos;
    }
}

Vector2 Node2D::get_global_position() const {
    return get_global_transform().origin;
}

void Node2D::set_global_rotation(float radians) {
    Node* parent = static_cast<Node*>(get_parent());
    if (parent) {
        float parent_rot = parent->get_global_rotation();
        _rotation = radians - parent_rot;
    } else {
        _rotation = radians;
    }
}

float Node2D::get_global_rotation() const {
    return get_global_transform().get_rotation();
}

void Node2D::set_global_scale(const Vector2& scale) {
    Node* parent = static_cast<Node*>(get_parent());
    if (parent) {
        Vector2 parent_scale = parent->get_global_scale();
        _scale = Vector2(
            parent_scale.x != 0 ? scale.x / parent_scale.x : 0,
            parent_scale.y != 0 ? scale.y / parent_scale.y : 0
        );
    } else {
        _scale = scale;
    }
}

Vector2 Node2D::get_global_scale() const {
    Transform2D global = get_global_transform();
    return global.get_scale();
}

Transform2D Node2D::get_transform() const {
    Transform2D t;
    t.origin = _position;
    t.basis_x = Vector2(std::cos(_rotation), std::sin(_rotation)) * _scale.x;
    t.basis_y = Vector2(-std::sin(_rotation), std::cos(_rotation)) * _scale.y;
    return t;
}

Transform2D Node2D::get_global_transform() const {
    Node* parent = static_cast<Node*>(get_parent());
    return parent ? parent->get_global_transform_2d() * get_transform() : get_transform();
}

Transform2D Node2D::get_relative_transform() const {
    return get_transform();
}

void Node2D::_draw() {
    // 子类重写
}

void Node2D::draw_line(const Vector2& from, const Vector2& to, const Color& color, float width) {
    (void)from; (void)to; (void)color; (void)width;
}

void Node2D::draw_rect(const Rect2& rect, const Color& color, bool filled, float width) {
    (void)rect; (void)color; (void)filled; (void)width;
}

void Node2D::draw_circle(const Vector2& center, float radius, const Color& color) {
    (void)center; (void)radius; (void)color;
}

void Node2D::draw_texture(Texture2D* texture, const Vector2& pos, const Color& modulate) {
    (void)texture; (void)pos; (void)modulate;
}

} // namespace MyEngine
