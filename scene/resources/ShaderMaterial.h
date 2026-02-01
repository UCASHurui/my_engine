#pragma once

#include "Material.h"

namespace MyEngine {

// 基于着色器的材质
class ShaderMaterial : public Material {
public:
    ShaderMaterial();
    ~ShaderMaterial() override;

    const char* get_class_name() const override { return "ShaderMaterial"; }

    // 创建内置着色器材质
    static Ref<ShaderMaterial> create_basic();
    static Ref<ShaderMaterial> create_pbr();
    static Ref<ShaderMaterial> create_unshaded();
    static Ref<ShaderMaterial> create_wireframe();
    static Ref<ShaderMaterial> create_particle();
    static Ref<ShaderMaterial> create_sky();
    static Ref<ShaderMaterial> create_fog();

protected:
    bool _init();
    bool _init_pbr();
    bool _init_unshaded();
    bool _init_particle();
    bool _init_sky();
    bool _init_fog();

private:
    static Ref<Shader> _basic_shader;
    static Ref<Shader> _pbr_shader;
    static Ref<Shader> _unshaded_shader;
    static Ref<Shader> _wireframe_shader;
    static Ref<Shader> _particle_shader;
    static Ref<Shader> _sky_shader;
    static Ref<Shader> _fog_shader;
};

} // namespace MyEngine
