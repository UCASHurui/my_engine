#include "Sprite2D.h"

namespace MyEngine {

Sprite2D::Sprite2D() {}

Sprite2D::~Sprite2D() {}

void Sprite2D::set_texture(Texture2D* texture) {
    _texture = texture;
}

void Sprite2D::set_region(const Rect2& region) {
    _region = region;
    _region_enabled = true;
}

void Sprite2D::set_frame(int frame) {
    int max_frame = _hframes * _vframes;
    _frame = (frame + max_frame) % max_frame;
}

void Sprite2D::set_frame_coords(const Vector2& coords) {
    set_frame((int)coords.y * _hframes + (int)coords.x);
}

Vector2 Sprite2D::get_frame_coords() const {
    int max_frame = _hframes * _vframes;
    int frame = (_frame + max_frame) % max_frame;
    return Vector2(frame % _hframes, frame / _hframes);
}

void Sprite2D::_draw() {
    if (_texture.is_null()) return;

    Vector2 size;
    if (_region_enabled) {
        size = _region.size;
    } else {
        size = Vector2(1, 1); // TODO: 获取纹理大小
    }

    Transform2D t = get_global_transform();
    Color mod = _modulate * _global_modulate;

    // TODO: 实际绘制
    (void)t;
    (void)mod;
}

// AnimatedSprite2D

AnimatedSprite2D::AnimatedSprite2D() {}

AnimatedSprite2D::~AnimatedSprite2D() {}

void AnimatedSprite2D::play(const String& animation) {
    if (!animation.empty()) {
        _current_animation = animation;
    }
    if (_animations.contains(_current_animation)) {
        _playing = true;
        _frame_index = 0;
        _time = 0.0f;
    }
}

void AnimatedSprite2D::pause() {
    _playing = false;
}

void AnimatedSprite2D::stop() {
    _playing = false;
    _frame_index = 0;
}

void AnimatedSprite2D::add_frame(const String& animation, Texture2D* texture, float delay) {
    Animation anim;
    Frame frame;
    frame.texture = texture;
    frame.delay = delay;
    anim.frames.push_back(frame);
    _animations.insert(animation, anim);
}

void AnimatedSprite2D::add_frame_by_filename(const String& animation, const String& path, float delay) {
    // 加载纹理并添加帧
    (void)animation; (void)path; (void)delay;
}

void AnimatedSprite2D::clear_animations() {
    _animations.clear();
}

Vector<String> AnimatedSprite2D::get_animations() const {
    Vector<String> result;
    for (const auto& kv : _animations) {
        result.push_back(kv.key);
    }
    return result;
}

void AnimatedSprite2D::_process(float delta) {
    if (!_playing) return;
    if (_current_animation.empty()) return;

    Animation* anim = _get_animation(_current_animation);
    if (!anim || anim->frames.empty()) return;

    _time += delta * _animation_speed;

    while (_time >= anim->frames[_frame_index].delay) {
        _time -= anim->frames[_frame_index].delay;
        _frame_index++;
        if (_frame_index >= (int)anim->frames.size()) {
            _frame_index = 0;
            // 发送动画结束信号
        }
    }

    set_frame(_frame_index);
}

AnimatedSprite2D::Animation* AnimatedSprite2D::_get_animation(const String& name) {
    auto it = _animations.find(name);
    if (it != _animations.end()) {
        return &it->value;
    }
    return nullptr;
}

} // namespace MyEngine
