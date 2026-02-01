#include "PostProcess.h"
#include "scene/resources/Texture2D.h"
#include "renderer/RenderTarget.h"

namespace MyEngine {

void SSAOEffect::process(Texture2D* input, Texture2D* output, float delta) {
    (void)input; (void)output; (void)delta;
    // 简化实现
}

void BloomEffect::process(Texture2D* input, Texture2D* output, float delta) {
    (void)input; (void)output; (void)delta;
    // 简化实现
}

void DOFEffect::process(Texture2D* input, Texture2D* output, float delta) {
    (void)input; (void)output; (void)delta;
    // 简化实现
}

void ToneMappingEffect::process(Texture2D* input, Texture2D* output, float delta) {
    (void)input; (void)output; (void)delta;
    // 简化实现
}

void VignetteEffect::process(Texture2D* input, Texture2D* output, float delta) {
    (void)input; (void)output; (void)delta;
    // 简化实现
}

PostProcess::PostProcess() = default;

PostProcess::~PostProcess() = default;

void PostProcess::add_effect(std::unique_ptr<PostEffect> effect) {
    _effects.push_back(std::move(effect));
}

void PostProcess::remove_effect(EffectType type) {
    _effects.erase(
        std::remove_if(_effects.begin(), _effects.end(),
            [type](const auto& e) { return e->get_type() == type; }),
        _effects.end()
    );
}

void PostProcess::clear_effects() {
    _effects.clear();
}

PostEffect* PostProcess::get_effect(EffectType type) const {
    for (const auto& e : _effects) {
        if (e->get_type() == type) return e.get();
    }
    return nullptr;
}

void PostProcess::process(Texture2D* input, Texture2D* output, float delta) {
    if (!_enabled) return;

    Texture2D* current_input = input;
    Texture2D* temp_output = nullptr;

    for (const auto& effect : _effects) {
        if (!effect->is_enabled()) continue;

        if (temp_output) {
            effect->process(current_input, temp_output, delta);
            current_input = temp_output;
        } else {
            effect->process(current_input, output, delta);
        }
    }
}

void PostProcess::set_effect_intensity(EffectType type, float intensity) {
    PostEffect* effect = get_effect(type);
    if (effect) {
        effect->set_intensity(intensity);
    }
}

} // namespace MyEngine
