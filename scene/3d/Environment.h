#pragma once

#include "math/Color.h"
#include "math/Vector3.h"
#include "object/RefCounted.h"
#include "variant/Variant.h"
#include <cstdint>

namespace MyEngine {

// 环境背景类型
enum class BackgroundType {
    COLOR,
    SKY,
    SKYBOX,
    CANVAS,
    CUSTOM
};

// 雾类型
enum class FogType {
    NONE,
    EXP,
    EXP2,
    LINEAR
};

// 环境设置
class Environment : public RefCounted {
public:
    Environment();
    ~Environment() override;

    // 背景
    void set_background_type(BackgroundType type) { _background_type = type; }
    BackgroundType get_background_type() const { return _background_type; }

    void set_background_color(const Color& color) { _background_color = color; }
    Color get_background_color() const { return _background_color; }

    void set_sky_orientation(const Vector3& orientation) { _sky_orientation = orientation; }
    Vector3 get_sky_orientation() const { return _sky_orientation; }

    // 环境光
    void set_ambient_light_color(const Color& color) { _ambient_color = color; }
    Color get_ambient_light_color() const { return _ambient_color; }

    void set_ambient_light_energy(float energy) { _ambient_energy = energy; }
    float get_ambient_light_energy() const { return _ambient_energy; }

    // 雾
    void set_fog_type(FogType type) { _fog_type = type; }
    FogType get_fog_type() const { return _fog_type; }

    void set_fog_color(const Color& color) { _fog_color = color; }
    Color get_fog_color() const { return _fog_color; }

    void set_fog_sun_color(const Color& color) { _fog_sun_color = color; }
    Color get_fog_sun_color() const { return _fog_sun_color; }

    void set_fog_density(float density) { _fog_density = density; }
    float get_fog_density() const { return _fog_density; }

    void set_fog_depth_begin(float begin) { _fog_depth_begin = begin; }
    float get_fog_depth_begin() const { return _fog_depth_begin; }

    void set_fog_depth_end(float end) { _fog_depth_end = end; }
    float get_fog_depth_end() const { return _fog_depth_end; }

    void set_fog_sky_affect(float affect) { _fog_sky_affect = affect; }
    float get_fog_sky_affect() const { return _fog_sky_affect; }

    // 色调映射
    void set_tone_map_type(int type) { _tone_map_type = type; }
    int get_tone_map_type() const { return _tone_map_type; }

    void set_tone_map_exposure(float exposure) { _tone_map_exposure = exposure; }
    float get_tone_map_exposure() const { return _tone_map_exposure; }

    void set_tone_map_white(float white) { _tone_map_white = white; }
    float get_tone_map_white() const { return _tone_map_white; }

    // 调整
    void set_brightness(float brightness) { _brightness = brightness; }
    float get_brightness() const { return _brightness; }

    void set_contrast(float contrast) { _contrast = contrast; }
    float get_contrast() const { return _contrast; }

    void set_saturation(float saturation) { _saturation = saturation; }
    float get_saturation() const { return _saturation; }

    // 屏幕空间特效
    void set_ssao_enabled(bool enabled) { _ssao_enabled = enabled; }
    bool is_ssao_enabled() const { return _ssao_enabled; }

    void set_glow_enabled(bool enabled) { _glow_enabled = enabled; }
    bool is_glow_enabled() const { return _glow_enabled; }

    void set_dof_blur_size(float size) { _dof_blur_size = size; }
    float get_dof_blur_size() const { return _dof_blur_size; }

    void set_dof_distance_begin(float begin) { _dof_distance_begin = begin; }
    float get_dof_distance_begin() const { return _dof_distance_begin; }

    void set_dof_distance_end(float end) { _dof_distance_end = end; }
    float get_dof_distance_end() const { return _dof_distance_end; }

    // 创建默认环境
    static Ref<Environment> create_default();

private:
    BackgroundType _background_type = BackgroundType::COLOR;
    Color _background_color = Color(0.1f, 0.1f, 0.1f, 1.0f);
    Vector3 _sky_orientation;

    Color _ambient_color = Color(0.2f, 0.2f, 0.2f, 1.0f);
    float _ambient_energy = 0.5f;

    FogType _fog_type = FogType::NONE;
    Color _fog_color = Color(0.5f, 0.6f, 0.7f, 1.0f);
    Color _fog_sun_color = Color(0.8f, 0.7f, 0.6f, 1.0f);
    float _fog_density = 0.01f;
    float _fog_depth_begin = 0.0f;
    float _fog_depth_end = 100.0f;
    float _fog_sky_affect = 0.0f;

    int _tone_map_type = 0;
    float _tone_map_exposure = 1.0f;
    float _tone_map_white = 1.0f;

    float _brightness = 0.0f;
    float _contrast = 1.0f;
    float _saturation = 1.0f;

    bool _ssao_enabled = false;
    bool _glow_enabled = false;

    float _dof_blur_size = 0.0f;
    float _dof_distance_begin = 10.0f;
    float _dof_distance_end = 20.0f;
};

} // namespace MyEngine
