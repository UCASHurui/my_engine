#pragma once

#include "core/object/Resource.h"
#include "core/math/Color.h"
#include "core/math/Vector2.h"
#include "core/math/Vector3.h"
#include "core/math/Matrix4.h"
#include "Texture2D.h"
#include <string>
#include <unordered_map>
#include <variant>

namespace MyEngine {

class Shader;

// 材质参数
struct MaterialParam {
    std::string name;
    std::variant<float, Vector2, Vector3, Vector4, Color, Ref<Texture2D>> value;
    bool is_instance = false;
};

// 材质标志
enum class MaterialFlag {
    VERTEX_COLOR = 0,      // 使用顶点颜色
    DIFFUSE_MAP,           // 使用漫反射贴图
    SPECULAR_MAP,          // 使用高光贴图
    NORMAL_MAP,            // 使用法线贴图
    EMISSION_MAP,          // 使用自发光贴图
    AO_MAP,                // 使用环境光遮蔽贴图
    ALbedo_COLOR,          // 使用反照率颜色
    SHADOWS,               // 投射阴影
    RECEIVE_SHADOWS,       // 接收阴影
    BILLBOARD,             //  billboard 模式
    Billboard_Y_LOCKED,    // Y轴锁定的 billboard
    PARTICLE_Trails,       // 粒子拖尾
    ANIMATED,              // 动画材质
    UNSHADED,              // 无光照
    WIREFRAME,             // 线框模式
    OCCLUSION,             // 遮挡
    MIPMAP,                // 使用 mipmap
    ALPHA_SCROLL,          // alpha 滚动
    UV_SCROLL,             // UV 滚动
    MAX_FLAGS
};

// 混合模式
enum class BlendMode {
    DISABLED,      // 不混合
    MIX,           // 混合 (src_alpha, 1-src_alpha)
    ADD,           // 加法混合
    SUB,           // 减法混合
    MUL,           // 乘法混合
    PREMULT_ALPHA  // 预乘 alpha
};

// 深度测试
enum class DepthTest {
    DISABLED,
    LESS,
    EQUAL,
    LEQUAL,
    GREATER,
    NOTEQUAL,
    GEQUAL,
    ALWAYS
};

// 绘制通道
enum class DrawPass {
    MAIN,
    SHADOW,
    DEPTH,
    ALBEDO,
    NORMAL,
    MOTION,
    SSAO,
    MAX_PASSES
};

// 基础材质
class Material : public Resource {
public:
    Material();
    ~Material() override;

    const char* get_class_name() const override { return "Material"; }

    // 着色器
    void set_shader(Ref<Shader> shader) { _shader = shader; }
    Ref<Shader> get_shader() const { return _shader; }

    // 参数设置
    void set_shader_parameter(const std::string& name, const std::variant<float, Vector2, Vector3, Vector4, Color, Ref<Texture2D>>& value);

    std::variant<float, Vector2, Vector3, Vector4, Color, Ref<Texture2D>> get_shader_parameter(const std::string& name) const;

    // 渲染状态
    void set_blend_mode(BlendMode mode) { _blend_mode = mode; }
    BlendMode get_blend_mode() const { return _blend_mode; }

    void set_depth_test(DepthTest mode) { _depth_test = mode; }
    DepthTest get_depth_test() const { return _depth_test; }

    void set_cull_mode(int mode) { _cull_mode = mode; }
    int get_cull_mode() const { return _cull_mode; }

    // 标志
    void set_flag(MaterialFlag flag, bool enabled) {
        if (enabled) _flags |= (1 << (int)flag);
        else _flags &= ~(1 << (int)flag);
    }
    bool get_flag(MaterialFlag flag) const { return (_flags & (1 << (int)flag)) != 0; }

    // 渲染顺序
    void set_render_priority(int priority) { _render_priority = priority; }
    int get_render_priority() const { return _render_priority; }

    // 颜色
    void set_albedo_color(const Color& color) { _albedo_color = color; }
    Color get_albedo_color() const { return _albedo_color; }

    // 物理材质属性
    void set_metallic(float metallic) { _metallic = metallic; }
    float get_metallic() const { return _metallic; }

    void set_roughness(float roughness) { _roughness = roughness; }
    float get_roughness() const { return _roughness; }

    void set_sss_strength(float strength) { _sss_strength = strength; }
    float get_sss_strength() const { return _sss_strength; }

    // 透明度
    void set_transparency(float transparency) { _transparency = transparency; }
    float get_transparency() const { return _transparency; }

    // 自发光
    void set_emission_color(const Color& color) { _emission_color = color; }
    Color get_emission_color() const { return _emission_color; }

    void set_emission_energy(float energy) { _emission_energy = energy; }
    float get_emission_energy() const { return _emission_energy; }

    // 贴图
    void set_albedo_texture(Ref<Texture2D> tex) { _albedo_texture = tex; }
    Ref<Texture2D> get_albedo_texture() const { return _albedo_texture; }

    void set_normal_texture(Ref<Texture2D> tex) { _normal_texture = tex; }
    Ref<Texture2D> get_normal_texture() const { return _normal_texture; }

    void set_roughness_texture(Ref<Texture2D> tex) { _roughness_texture = tex; }
    Ref<Texture2D> get_roughness_texture() const { return _roughness_texture; }

    void set_metallic_texture(Ref<Texture2D> tex) { _metallic_texture = tex; }
    Ref<Texture2D> get_metallic_texture() const { return _metallic_texture; }

    void set_ao_texture(Ref<Texture2D> tex) { _ao_texture = tex; }
    Ref<Texture2D> get_ao_texture() const { return _ao_texture; }

    void set_emission_texture(Ref<Texture2D> tex) { _emission_texture = tex; }
    Ref<Texture2D> get_emission_texture() const { return _emission_texture; }

    // 克隆
    Ref<Material> duplicate() const;

protected:
    void _copy_from(const Material* other);

private:
    Ref<Shader> _shader;

    // 渲染状态
    BlendMode _blend_mode = BlendMode::DISABLED;
    DepthTest _depth_test = DepthTest::LESS;
    int _cull_mode = 1;  // 1 = back, 0 = front, -1 = off
    int _flags = 0;
    int _render_priority = 0;

    // 颜色和属性
    Color _albedo_color = Color(1, 1, 1, 1);
    float _metallic = 0.0f;
    float _roughness = 0.5f;
    float _sss_strength = 0.0f;
    float _transparency = 1.0f;

    Color _emission_color = Color::BLACK();
    float _emission_energy = 0.0f;

    // 贴图
    Ref<Texture2D> _albedo_texture;
    Ref<Texture2D> _normal_texture;
    Ref<Texture2D> _roughness_texture;
    Ref<Texture2D> _metallic_texture;
    Ref<Texture2D> _ao_texture;
    Ref<Texture2D> _emission_texture;

    // 自定义参数
    std::unordered_map<std::string, MaterialParam> _params;
};

} // namespace MyEngine
