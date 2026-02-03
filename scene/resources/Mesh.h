#pragma once

#include "containers/Vector.h"
#include "containers/String.h"
#include "math/Vector2.h"
#include "math/Vector3.h"
#include "variant/Variant.h"
#include "object/RefCounted.h"
#include "renderer/RenderDevice.h"
#include <cstdint>

namespace MyEngine {

// 顶点格式
struct VertexAttribute {
    enum Type {
        FLOAT,
        VEC2,
        VEC3,
        VEC4,
        INT,
        IVEC2,
        IVEC3,
        IVEC4,
        UINT,
        UVEC2,
        UVEC3,
        UVEC4
    };

    String name;
    Type type;
    int count;  // 元素数量 (1-4)
    int offset; // 字节偏移
};

// 顶点格式定义
class VertexFormat {
public:
    VertexFormat() = default;

    void add_attribute(const String& name, VertexAttribute::Type type) {
        VertexAttribute attr;
        attr.name = name;
        attr.type = type;
        attr.count = _get_type_count(type);
        attr.offset = _get_size();
        _attributes.push_back(attr);
        _stride = attr.offset + _get_type_size(type);
    }

    int get_attribute_count() const { return (int)_attributes.size(); }
    int get_stride() const { return _stride; }
    const Vector<VertexAttribute>& get_attributes() const { return _attributes; }

    int find_attribute(const String& name) const {
        for (int i = 0; i < (int)_attributes.size(); i++) {
            if (_attributes[i].name == name) return i;
        }
        return -1;
    }

private:
    Vector<VertexAttribute> _attributes;
    int _stride = 0;

    int _get_type_count(VertexAttribute::Type type) const {
        switch (type) {
            case VertexAttribute::FLOAT: return 1;
            case VertexAttribute::VEC2: return 2;
            case VertexAttribute::VEC3: return 3;
            case VertexAttribute::VEC4: return 4;
            default: return 1;
        }
    }

    int _get_type_size(VertexAttribute::Type type) const {
        switch (type) {
            case VertexAttribute::FLOAT:
            case VertexAttribute::INT:
            case VertexAttribute::UINT: return 4;
            case VertexAttribute::VEC2:
            case VertexAttribute::IVEC2:
            case VertexAttribute::UVEC2: return 8;
            case VertexAttribute::VEC3:
            case VertexAttribute::IVEC3:
            case VertexAttribute::UVEC3: return 12;
            case VertexAttribute::VEC4:
            case VertexAttribute::IVEC4:
            case VertexAttribute::UVEC4: return 16;
            default: return 4;
        }
    }

    int _get_size() const {
        int size = 0;
        for (const auto& attr : _attributes) {
            size += _get_type_size(attr.type);
        }
        return size;
    }
};

// 网格资源
class Mesh : public RefCounted {
public:
    Mesh();
    ~Mesh();

    // 设置顶点格式
    void set_format(const VertexFormat& format) { _format = format; }
    const VertexFormat& get_format() const { return _format; }

    // 设置顶点数据
    void set_vertices(const void* data, size_t count, RenderDevice::BufferUsage usage = RenderDevice::STATIC_DRAW);
    void set_indices(const uint16_t* data, size_t count, RenderDevice::BufferUsage usage = RenderDevice::STATIC_DRAW);
    void set_indices(const uint32_t* data, size_t count, RenderDevice::BufferUsage usage = RenderDevice::STATIC_DRAW);

    // 获取顶点/索引数量
    size_t get_vertex_count() const { return _vertex_count; }
    size_t get_index_count() const { return _index_count; }

    // 资源句柄
    RenderDevice::BufferID get_vertex_buffer() const { return _vertex_buffer; }
    RenderDevice::BufferID get_index_buffer() const { return _index_buffer; }
    RenderDevice::IndexType get_index_type() const { return _index_type; }

    // 包围盒
    Vector3 get_aabb_min() const { return _aabb_min; }
    Vector3 get_aabb_max() const { return _aabb_max; }

    // 更新数据
    void update_vertices(const void* data, size_t offset = 0, size_t size = 0);
    void update_indices(const void* data, size_t offset = 0, size_t size = 0);

private:
    VertexFormat _format;
    size_t _vertex_count = 0;
    size_t _index_count = 0;
    RenderDevice::BufferID _vertex_buffer = 0;
    RenderDevice::BufferID _index_buffer = 0;
    RenderDevice::IndexType _index_type = RenderDevice::INDEX_16BIT;
    RenderDevice::BufferUsage _index_usage = RenderDevice::STATIC_DRAW;

    Vector3 _aabb_min;
    Vector3 _aabb_max;

    void _calculate_aabb();
};

} // namespace MyEngine
