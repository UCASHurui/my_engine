#include "ShaderMaterial.h"
#include "Shader.h"

namespace MyEngine {

Ref<Shader> ShaderMaterial::_basic_shader;
Ref<Shader> ShaderMaterial::_pbr_shader;
Ref<Shader> ShaderMaterial::_unshaded_shader;
Ref<Shader> ShaderMaterial::_wireframe_shader;
Ref<Shader> ShaderMaterial::_particle_shader;
Ref<Shader> ShaderMaterial::_sky_shader;
Ref<Shader> ShaderMaterial::_fog_shader;

ShaderMaterial::ShaderMaterial() = default;

ShaderMaterial::~ShaderMaterial() = default;

Ref<ShaderMaterial> ShaderMaterial::create_basic() {
    Ref<ShaderMaterial> mat = new ShaderMaterial();
    mat->set_shader_parameter("albedo", Vector4(1, 1, 1, 1));
    mat->set_metallic(0.0f);
    mat->set_roughness(0.5f);
    return mat;
}

Ref<ShaderMaterial> ShaderMaterial::create_pbr() {
    Ref<ShaderMaterial> mat = new ShaderMaterial();
    mat->set_shader_parameter("albedo", Vector4(1, 1, 1, 1));
    mat->set_metallic(0.0f);
    mat->set_roughness(0.5f);
    return mat;
}

Ref<ShaderMaterial> ShaderMaterial::create_unshaded() {
    Ref<ShaderMaterial> mat = new ShaderMaterial();
    mat->set_shader_parameter("albedo", Vector4(1, 1, 1, 1));
    mat->set_flag(MaterialFlag::UNSHADED, true);
    return mat;
}

Ref<ShaderMaterial> ShaderMaterial::create_wireframe() {
    Ref<ShaderMaterial> mat = new ShaderMaterial();
    mat->set_shader_parameter("albedo", Vector4(1, 1, 1, 1));
    mat->set_flag(MaterialFlag::WIREFRAME, true);
    return mat;
}

Ref<ShaderMaterial> ShaderMaterial::create_particle() {
    Ref<ShaderMaterial> mat = new ShaderMaterial();
    mat->set_shader_parameter("albedo", Vector4(1, 1, 1, 1));
    mat->set_flag(MaterialFlag::VERTEX_COLOR, true);
    mat->set_flag(MaterialFlag::BILLBOARD, true);
    mat->set_flag(MaterialFlag::UNSHADED, true);
    mat->set_blend_mode(BlendMode::ADD);
    return mat;
}

Ref<ShaderMaterial> ShaderMaterial::create_sky() {
    Ref<ShaderMaterial> mat = new ShaderMaterial();
    mat->set_shader_parameter("albedo", Vector4(0.1f, 0.1f, 0.15f, 1.0f));
    mat->set_flag(MaterialFlag::UNSHADED, true);
    return mat;
}

Ref<ShaderMaterial> ShaderMaterial::create_fog() {
    Ref<ShaderMaterial> mat = new ShaderMaterial();
    mat->set_shader_parameter("albedo", Vector4(0.5f, 0.6f, 0.7f, 1.0f));
    mat->set_flag(MaterialFlag::UNSHADED, true);
    mat->set_flag(MaterialFlag::ALPHA_SCROLL, true);
    mat->set_transparency(0.5f);
    return mat;
}

} // namespace MyEngine
