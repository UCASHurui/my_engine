#pragma once

#include "Node3D.h"
#include "core/math/Color.h"

namespace MyEngine {

// 光源类型
enum class LightType {
    DIRECTIONAL,  // 平行光
    POINT,        // 点光源
    SPOT,         // 聚光灯
    OMNI          // 全向光
};

// 阴影类型
enum class ShadowType {
    NONE,
    HARD,
    SOFT
};

// 3D 光源
class Light3D : public Node3D {
public:
    Light3D();
    ~Light3D() override;

    const char* get_class_name() const override { return "Light3D"; }

    // 光源类型
    void set_light_type(LightType type) { _light_type = type; }
    LightType get_light_type() const { return _light_type; }

    // 颜色和强度
    void set_color(const Color& color) { _color = color; }
    Color get_color() const { return _color; }

    void set_energy(float energy) { _energy = energy; }
    float get_energy() const { return _energy; }

    void set_indirect_energy(float energy) { _indirect_energy = energy; }
    float get_indirect_energy() const { return _indirect_energy; }

    // 衰减
    void set_attenuation(float attenuation) { _attenuation = attenuation; }
    float get_attenuation() const { return _attenuation; }

    void set_range(float range) { _range = range; }
    float get_range() const { return _range; }

    // 聚光灯参数
    void set_spot_angle(float angle_degrees) { _spot_angle = angle_degrees; }
    float get_spot_angle() const { return _spot_angle; }

    void set_spot_attenuation(float attenuation) { _spot_attenuation = attenuation; }
    float get_spot_attenuation() const { return _spot_attenuation; }

    // 阴影
    void set_shadow_mode(ShadowType type) { _shadow_type = type; }
    ShadowType get_shadow_mode() const { return _shadow_type; }

    void set_shadow_bias(float bias) { _shadow_bias = bias; }
    float get_shadow_bias() const { return _shadow_bias; }

    void set_shadow_normal_bias(float bias) { _shadow_normal_bias = bias; }
    float get_shadow_normal_bias() const { return _shadow_normal_bias; }

    void set_shadow_blur(float blur) { _shadow_blur = blur; }
    float get_shadow_blur() const { return _shadow_blur; }

    // 烘焙
    void set_bake_mode(int mode) { _bake_mode = mode; }
    int get_bake_mode() const { return _bake_mode; }

    // 最大距离（用于视锥剔除）
    void set_max_distance(float distance) { _max_distance = distance; }
    float get_max_distance() const { return _max_distance; }

    // 是否启用
    void set_enabled(bool enabled) { _enabled = enabled; }
    bool is_enabled() const { return _enabled; }

protected:
    LightType _light_type = LightType::DIRECTIONAL;

    Color _color = Color(1, 1, 1, 1);
    float _energy = 1.0f;
    float _indirect_energy = 0.0f;

    float _attenuation = 1.0f;
    float _range = 0.0f;  // 0 = 无穷远

    float _spot_angle = 45.0f;
    float _spot_attenuation = 1.0f;

    ShadowType _shadow_type = ShadowType::NONE;
    float _shadow_bias = 0.0f;
    float _shadow_normal_bias = 0.0f;
    float _shadow_blur = 0.0f;

    int _bake_mode = 0;  // 0 = Dynamic, 1 = Static, 2 = Mixed
    float _max_distance = 0.0f;

    bool _enabled = true;
};

} // namespace MyEngine
