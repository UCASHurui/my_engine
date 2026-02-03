#include "MeshInstance3D.h"
#include "main/Node.h"

namespace MyEngine {

MeshInstance3D::MeshInstance3D() {}

MeshInstance3D::~MeshInstance3D() {}

void MeshInstance3D::set_mesh(Mesh* mesh) {
    _mesh = mesh;
}

void MeshInstance3D::set_material_override(Material* material) {
    _material_override = material;
}

void MeshInstance3D::set_material(int surface, Material* material) {
    if (surface >= (int)_surface_materials.size()) {
        _surface_materials.resize(surface + 1);
    }
    _surface_materials[surface] = material;
}

Material* MeshInstance3D::get_material(int surface) const {
    if (surface >= 0 && surface < (int)_surface_materials.size()) {
        return _surface_materials[surface].get();
    }
    return nullptr;
}

void MeshInstance3D::set_cast_shadow(int mode) {
    _cast_shadow = mode;
}

void MeshInstance3D::set_gi_mode(int mode) {
    _gi_mode = mode;
}

void MeshInstance3D::create_trimesh_collision() {
    // TODO: 从网格创建碰撞体
}

void MeshInstance3D::create_convex_collision() {
    // TODO: 创建凸包碰撞体
}

void MeshInstance3D::_notification(int notification) {
    switch (notification) {
        case NodeNotification::ENTER_TREE:
            // 注册到渲染器
            break;
        case NodeNotification::EXIT_TREE:
            // 从渲染器移除
            break;
    }
}

} // namespace MyEngine
