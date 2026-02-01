#pragma once

#include "Node3D.h"
#include "core/containers/Vector.h"
#include "core/object/Object.h"
#include "Mesh.h"
#include "Material.h"

namespace MyEngine {

// 3D 网格实例
class MeshInstance3D : public Node3D {

public:
    MeshInstance3D();
    virtual ~MeshInstance3D();

    virtual const char* get_class_name() const override { return "MeshInstance3D"; }

    // 网格
    void set_mesh(Mesh* mesh);
    Mesh* get_mesh() const { return _mesh.get(); }

    // 材质
    void set_material_override(Material* material);
    Material* get_material_override() const { return _material_override.get(); }
    void set_material(int surface, Material* material);
    Material* get_material(int surface) const;
    int get_surface_count() const { return _surface_materials.size(); }

    // 阴影
    void set_cast_shadow(int mode);
    int get_cast_shadow() const { return _cast_shadow; }
    void set_receive_shadow(bool receive) { _receive_shadow = receive; }
    bool is_receiving_shadow() const { return _receive_shadow; }

    // 烘焙
    void set_gi_mode(int mode);
    int get_gi_mode() const { return _gi_mode; }
    void set_baked_lightmap_uv(int mode) { _lightmap_uv = mode; }
    int get_baked_lightmap_uv() const { return _lightmap_uv; }

    // LOD
    void set_lod_min_hysteresis(float h) { _lod_min_hysteresis = h; }
    float get_lod_min_hysteresis() const { return _lod_min_hysteresis; }
    void set_lod_min_distance(float d) { _lod_min_distance = d; }
    float get_lod_min_distance() const { return _lod_min_distance; }

    // 变换
    void set_extra_cull_margin(float margin) { _extra_cull_margin = margin; }
    float get_extra_cull_margin() const { return _extra_cull_margin; }

    // 碰撞
    void create_trimesh_collision();
    void create_convex_collision();

protected:
    virtual void _notification(int notification) override;

private:
    Ref<Mesh> _mesh;
    Ref<Material> _material_override;
    Vector<Ref<Material>> _surface_materials;

    int _cast_shadow = 2;  // SHADOW_CASTING_AND_RECEIVING
    bool _receive_shadow = true;
    int _gi_mode = 0;  // GI_MODE_INHERIT
    int _lightmap_uv = 0;  // UVW_USE_UV1
    float _lod_min_hysteresis = 0.0f;
    float _lod_min_distance = 0.0f;
    float _extra_cull_margin = 0.0f;
};

} // namespace MyEngine
