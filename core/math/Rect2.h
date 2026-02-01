#pragma once

#include "Vector2.h"

namespace MyEngine {

// 2D 矩形
struct Rect2 {
    Vector2 position;
    Vector2 size;

    Rect2() : position(0, 0), size(0, 0) {}
    Rect2(float x, float y, float width, float height)
        : position(x, y), size(width, height) {}
    Rect2(Vector2 pos, Vector2 size) : position(pos), size(size) {}

    // 属性
    float get_x() const { return position.x; }
    float get_y() const { return position.y; }
    float get_width() const { return size.x; }
    float get_height() const { return size.y; }
    float get_end_x() const { return position.x + size.x; }
    float get_end_y() const { return position.y + size.y; }
    Vector2 get_end() const { return position + size; }
    Vector2 get_center() const { return position + size * 0.5f; }
    float get_area() const { return size.x * size.y; }

    // 包含
    bool has_point(const Vector2& point) const {
        return point.x >= position.x && point.x <= position.x + size.x &&
               point.y >= position.y && point.y <= position.y + size.y;
    }
    bool has_area() const { return size.x > 0 && size.y > 0; }

    // 运算
    Rect2 operator+(const Vector2& offset) const {
        return Rect2(position + offset, size);
    }
    Rect2& operator+=(const Vector2& offset) {
        position += offset;
        return *this;
    }

    // 扩展
    Rect2 expand(const Vector2& point) const {
        Rect2 r = *this;
        Vector2 end = get_end();
        if (point.x < r.position.x) r.position.x = point.x;
        if (point.y < r.position.y) r.position.y = point.y;
        if (point.x > end.x) end.x = point.x;
        if (point.y > end.y) end.y = point.y;
        r.size = end - r.position;
        return r;
    }

    Rect2 grow(float margin) const {
        return Rect2(position - Vector2(margin, margin), size + Vector2(margin * 2, margin * 2));
    }

    // 交集
    Rect2 intersection(const Rect2& other) const {
        Vector2 pos1 = position;
        Vector2 pos2 = other.position;
        Vector2 end1 = get_end();
        Vector2 end2 = other.get_end();

        Vector2 pos(std::max(pos1.x, pos2.x), std::max(pos1.y, pos2.y));
        Vector2 end(std::min(end1.x, end2.x), std::min(end1.y, end2.y));

        if (end.x <= pos.x || end.y <= pos.y) {
            return Rect2();
        }
        return Rect2(pos, end - pos);
    }

    // 比较
    bool operator==(const Rect2& other) const {
        return position == other.position && size == other.size;
    }
    bool operator!=(const Rect2& other) const {
        return !(*this == other);
    }
};

} // namespace MyEngine
