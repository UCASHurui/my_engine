#include "Material.h"
#include "Shader.h"
#include "Texture2D.h"

namespace MyEngine {

Material::Material() = default;

Material::~Material() = default;

void Material::set_shader_parameter(const std::string& name, const std::variant<float, Vector2, Vector3, Vector4, Color, Ref<Texture2D>>& value) {
    MaterialParam param;
    param.name = name;
    param.value = value;
    param.is_instance = false;
    _params[name] = param;
}

std::variant<float, Vector2, Vector3, Vector4, Color, Ref<Texture2D>> Material::get_shader_parameter(const std::string& name) const {
    auto it = _params.find(name);
    if (it != _params.end()) {
        return it->second.value;
    }
    return float(0);
}

void Material::_copy_from(const Material* other) {
    _shader = other->_shader;
    _blend_mode = other->_blend_mode;
    _depth_test = other->_depth_test;
    _cull_mode = other->_cull_mode;
    _flags = other->_flags;
    _render_priority = other->_render_priority;

    _albedo_color = other->_albedo_color;
    _metallic = other->_metallic;
    _roughness = other->_roughness;
    _sss_strength = other->_sss_strength;
    _transparency = other->_transparency;

    _emission_color = other->_emission_color;
    _emission_energy = other->_emission_energy;

    _albedo_texture = other->_albedo_texture;
    _normal_texture = other->_normal_texture;
    _roughness_texture = other->_roughness_texture;
    _metallic_texture = other->_metallic_texture;
    _ao_texture = other->_ao_texture;
    _emission_texture = other->_emission_texture;

    _params = other->_params;
}

Ref<Material> Material::duplicate() const {
    Ref<Material> mat = new Material();
    mat->_copy_from(this);
    return mat;
}

} // namespace MyEngine
