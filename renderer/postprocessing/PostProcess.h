#pragma once

#include "core/math/Color.h"
#include "core/math/Vector2.h"
#include <string>
#include <vector>
#include <memory>

namespace MyEngine {

class Texture2D;
class RenderTarget;

// 后处理效果类型
enum class EffectType {
    NONE = 0,
    SSAO,           // 屏幕空间环境光遮蔽
    BLOOM,          // 泛光
    DOF,            // 景深
    MOTION_BLUR,    // 动态模糊
    FXAA,           // 快速近似抗锯齿
    SMAA,           // 形态学抗锯齿
    CHROMATIC_ABERRATION, // 色差
    GRAIN,          // 胶片颗粒
    VIGNETTE,       // 暗角
    COLOR_CORRECTION, // 颜色校正
    LUT,            // LUT 颜色查找表
    TONE_MAPPING,   // 色调映射
    FOG,            // 距离雾
    VOLUME_FOG,     // 体积雾
    LIGHT_SHAFTS,   // 光轴
    SCREEN_SPACE_REFLECTION, // 屏幕空间反射
    MAX_EFFECTS
};

// 后处理效果基类
class PostEffect {
public:
    PostEffect() = default;
    virtual ~PostEffect() = default;

    virtual EffectType get_type() const = 0;
    virtual bool is_enabled() const { return _enabled; }
    virtual void set_enabled(bool enabled) { _enabled = enabled; }

    virtual void set_intensity(float intensity) { _intensity = intensity; }
    virtual float get_intensity() const { return _intensity; }

    // 处理帧
    virtual void process(Texture2D* input, Texture2D* output, float delta) = 0;

protected:
    bool _enabled = true;
    float _intensity = 1.0f;
};

// SSAO 效果
class SSAOEffect : public PostEffect {
public:
    EffectType get_type() const override { return EffectType::SSAO; }

    void set_radius(float radius) { _radius = radius; }
    float get_radius() const { return _radius; }

    void set_bias(float bias) { _bias = bias; }
    float get_bias() const { return _bias; }

    void set_power(float power) { _power = power; }
    float get_power() const { return _power; }

    void process(Texture2D* input, Texture2D* output, float delta) override;

private:
    float _radius = 0.5f;
    float _bias = 0.025f;
    float _power = 2.0f;
};

// Bloom 效果
class BloomEffect : public PostEffect {
public:
    EffectType get_type() const override { return EffectType::BLOOM; }

    void set_threshold(float threshold) { _threshold = threshold; }
    float get_threshold() const { return _threshold; }

    void set_soft_threshold(float threshold) { _soft_threshold = threshold; }
    float get_soft_threshold() const { return _soft_threshold; }

    void process(Texture2D* input, Texture2D* output, float delta) override;

private:
    float _threshold = 0.8f;
    float _soft_threshold = 0.7f;
    int _iterations = 4;
    float _diffusion = 0.6f;
};

// DOF 效果
class DOFEffect : public PostEffect {
public:
    EffectType get_type() const override { return EffectType::DOF; }

    void set_focus_distance(float dist) { _focus_distance = dist; }
    float get_focus_distance() const { return _focus_distance; }

    void set_focus_range(float range) { _focus_range = range; }
    float get_focus_range() const { return _focus_range; }

    void set_blur_size(float size) { _blur_size = size; }
    float get_blur_size() const { return _blur_size; }

    void process(Texture2D* input, Texture2D* output, float delta) override;

private:
    float _focus_distance = 10.0f;
    float _focus_range = 2.0f;
    float _blur_size = 1.0f;
    bool _high_quality = false;
};

// 色调映射效果
class ToneMappingEffect : public PostEffect {
public:
    EffectType get_type() const override { return EffectType::TONE_MAPPING; }

    void set_exposure(float exposure) { _exposure = exposure; }
    float get_exposure() const { return _exposure; }

    void set_white(float white) { _white = white; }
    float get_white() const { return _white; }

    void set_tone_map_type(int type) { _tone_map_type = type; }
    int get_tone_map_type() const { return _tone_map_type; }

    void process(Texture2D* input, Texture2D* output, float delta) override;

private:
    float _exposure = 1.0f;
    float _white = 1.0f;
    int _tone_map_type = 0;  // 0=ACES, 1=Reinhard, 2=Filmic
};

// 暗角效果
class VignetteEffect : public PostEffect {
public:
    EffectType get_type() const override { return EffectType::VIGNETTE; }

    void set_center(const Vector2& center) { _center = center; }
    Vector2 get_center() const { return _center; }

    void set_sharpness(float sharpness) { _sharpness = sharpness; }
    float get_sharpness() const { return _sharpness; }

    void set_darkness(float darkness) { _darkness = darkness; }
    float get_darkness() const { return _darkness; }

    void process(Texture2D* input, Texture2D* output, float delta) override;

private:
    Vector2 _center = Vector2(0.5f, 0.5f);
    float _sharpness = 0.5f;
    float _darkness = 0.5f;
};

// 后处理链
class PostProcess {
public:
    PostProcess();
    ~PostProcess();

    // 添加效果
    void add_effect(std::unique_ptr<PostEffect> effect);

    // 移除效果
    void remove_effect(EffectType type);
    void clear_effects();

    // 获取效果
    PostEffect* get_effect(EffectType type) const;

    // 处理帧
    void process(Texture2D* input, Texture2D* output, float delta);

    // 设置强度
    void set_effect_intensity(EffectType type, float intensity);

    // 质量设置
    void set_quality(int quality) { _quality = quality; }
    int get_quality() const { return _quality; }

    // 启用/禁用
    void set_enabled(bool enabled) { _enabled = enabled; }
    bool is_enabled() const { return _enabled; }

private:
    std::vector<std::unique_ptr<PostEffect>> _effects;
    std::unique_ptr<RenderTarget> _temp_target;
    int _quality = 1;  // 0=low, 1=medium, 2=high
    bool _enabled = true;
};

} // namespace MyEngine
