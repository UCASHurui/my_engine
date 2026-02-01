#include "AnimationPlayer.h"

namespace MyEngine {

AnimationPlayer::AnimationPlayer() = default;
AnimationPlayer::~AnimationPlayer() = default;

void AnimationPlayer::add_animation(const std::string& name, Ref<Animation> anim) {
    _animations[name] = anim;
}

void AnimationPlayer::remove_animation(const std::string& name) {
    _animations.erase(name);
}

void AnimationPlayer::rename_animation(const std::string& old_name, const std::string& new_name) {
    auto it = _animations.find(old_name);
    if (it != _animations.end()) {
        _animations[new_name] = it->second;
        _animations.erase(it);
    }
}

Ref<Animation> AnimationPlayer::get_animation(const std::string& name) const {
    auto it = _animations.find(name);
    return it != _animations.end() ? it->second : nullptr;
}

std::vector<std::string> AnimationPlayer::get_animation_names() const {
    std::vector<std::string> names;
    for (const auto& [name, anim] : _animations) {
        names.push_back(name);
    }
    return names;
}

bool AnimationPlayer::has_animation(const std::string& name) const {
    return _animations.find(name) != _animations.end();
}

void AnimationPlayer::play(const std::string& name, bool from_end) {
    _from_end = from_end;
    _play_animation(name);
}

void AnimationPlayer::play_backwards(const std::string& name) {
    _speed_scale = -1.0f;
    _play_animation(name);
}

void AnimationPlayer::play_new(const std::string& name, float from_pos) {
    stop();
    _play_animation(name);
    _current_position = from_pos;
}

void AnimationPlayer::queue(const std::string& name) {
    _animation_queued.push_back(name);
}

void AnimationPlayer::clear_queue() {
    _animation_queued.clear();
}

void AnimationPlayer::_play_animation(const std::string& name) {
    auto it = _animations.find(name);
    if (it != _animations.end()) {
        _current_animation = it->second;
        _current_animation_name = name;
        _playing = true;
        _paused = false;
        _current_position = _from_end ? _current_animation->get_length() : 0.0f;
        _speed_scale = 1.0f;
    }
}

void AnimationPlayer::stop() {
    _playing = false;
    _paused = false;
    _current_position = 0.0f;
}

void AnimationPlayer::pause() {
    _paused = true;
}

void AnimationPlayer::seek(float position, bool update) {
    _current_position = std::max(0.0f, std::min(position, get_current_animation_length()));
}

void AnimationPlayer::seek_end(float offset) {
    if (_current_animation) {
        _current_position = _current_animation->get_length() + offset;
    }
}

void AnimationPlayer::animation_finished(const std::string& anim_name) {
    (void)anim_name;
    // 检查队列
    if (!_animation_queued.empty()) {
        std::string next = _animation_queued.front();
        _animation_queued.erase(_animation_queued.begin());
        play(next);
    }
}

void AnimationPlayer::update(float delta) {
    if (!_playing || _paused) return;

    _current_position += delta * _speed_scale;

    Ref<Animation> anim = _current_animation;
    if (!anim) return;

    float length = anim->get_length();
    bool finished = false;

    if (_speed_scale > 0) {
        if (_current_position >= length) {
            finished = true;
            _current_position = length;
        }
    } else {
        if (_current_position <= 0) {
            finished = true;
            _current_position = 0;
        }
    }

    if (finished) {
        if (anim->get_loop_mode() == LoopMode::LOOP_ENABLED ||
            anim->get_loop_mode() == LoopMode::LOOP_PINGPONG_ENABLED) {
            if (_speed_scale < 0) {
                _current_position = length;
                _speed_scale = std::abs(_speed_scale);
            } else {
                _current_position = 0;
            }
        } else {
            _playing = false;
            animation_finished(_current_animation_name);
        }
    }
}

} // namespace MyEngine
