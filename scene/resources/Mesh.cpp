#include "Mesh.h"
#include "core/os/OS.h"
#include "renderer/RenderDevice.h"

namespace MyEngine {

Mesh::Mesh() = default;

Mesh::~Mesh() {
    auto rd = RenderDevice::get_singleton();
    if (rd) {
        if (_vertex_buffer) rd->delete_buffer(_vertex_buffer);
        if (_index_buffer) rd->delete_buffer(_index_buffer);
    }
}

void Mesh::set_vertices(const void* data, size_t count, RenderDevice::BufferUsage usage) {
    auto rd = RenderDevice::get_singleton();
    if (!rd) return;

    if (_vertex_buffer) {
        rd->delete_buffer(_vertex_buffer);
    }

    _vertex_count = count;
    _vertex_buffer = rd->create_buffer(
        RenderDevice::VERTEX_BUFFER,
        usage,
        count * _format.get_stride(),
        data
    );

    _calculate_aabb();
}

void Mesh::set_indices(const uint16_t* data, size_t count, RenderDevice::BufferUsage usage) {
    auto rd = RenderDevice::get_singleton();
    if (!rd) return;

    if (_index_buffer) {
        rd->delete_buffer(_index_buffer);
    }

    _index_count = count;
    _index_type = RenderDevice::INDEX_16BIT;
    _index_usage = usage;
    _index_buffer = rd->create_buffer(
        RenderDevice::INDEX_BUFFER,
        usage,
        count * sizeof(uint16_t),
        data
    );
}

void Mesh::set_indices(const uint32_t* data, size_t count, RenderDevice::BufferUsage usage) {
    auto rd = RenderDevice::get_singleton();
    if (!rd) return;

    if (_index_buffer) {
        rd->delete_buffer(_index_buffer);
    }

    _index_count = count;
    _index_type = RenderDevice::INDEX_32BIT;
    _index_usage = usage;
    _index_buffer = rd->create_buffer(
        RenderDevice::INDEX_BUFFER,
        usage,
        count * sizeof(uint32_t),
        data
    );
}

void Mesh::update_vertices(const void* data, size_t offset, size_t size) {
    auto rd = RenderDevice::get_singleton();
    if (!rd || !_vertex_buffer) return;

    if (size == 0) {
        size = _vertex_count * _format.get_stride() - offset;
    }
    rd->update_buffer(_vertex_buffer, offset, size, data);
}

void Mesh::update_indices(const void* data, size_t offset, size_t size) {
    auto rd = RenderDevice::get_singleton();
    if (!rd || !_index_buffer) return;

    size_t elem_size = _index_type == RenderDevice::INDEX_16BIT ? sizeof(uint16_t) : sizeof(uint32_t);
    if (size == 0) {
        size = _index_count * elem_size - offset;
    }
    rd->update_buffer(_index_buffer, offset, size, data);
}

void Mesh::_calculate_aabb() {
    // 默认包围盒，后续可以根据顶点数据计算
    _aabb_min = Vector3(-1.0f, -1.0f, -1.0f);
    _aabb_max = Vector3(1.0f, 1.0f, 1.0f);
}

} // namespace MyEngine
