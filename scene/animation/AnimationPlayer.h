#pragma once

#include "object/RefCounted.h"
#include "Animation.h"
#include <unordered_map>

namespace MyEngine {

// 动画播放器节点
class AnimationPlayer : public RefCounted {
public:
    AnimationPlayer();
    ~AnimationPlayer() override;

    const char* get_class_name() const override { return "AnimationPlayer"; }

    // 动画管理
    void add_animation(const std::string& name, Ref<Animation> anim);
    void remove_animation(const std::string& name);
    void rename_animation(const std::string& old_name, const std::string& new_name);

    Ref<Animation> get_animation(const std::string& name) const;
    std::vector<std::string> get_animation_names() const;
    bool has_animation(const std::string& name) const;

    // 当前动画
    void play(const std::string& name, bool from_end = false);
    void play_backwards(const std::string& name);
    void play_new(const std::string& name, float from_pos = 0.0f);
    void queue(const std::string& name);
    void clear_queue();

    void stop();
    void pause();

    bool is_playing() const { return _playing; }
    bool is_paused() const { return _paused; }
    const std::string& get_current_animation() const { return _current_animation_name; }

    // 位置控制
    void seek(float position, bool update = true);
    void seek_end(float offset = 0.0f);

    float get_current_animation_position() const { return _current_position; }
    float get_current_animation_length() const { return _current_animation ? _current_animation->get_length() : 0.0f; }

    // 播放速度
    void set_speed_scale(float scale) { _speed_scale = scale; }
    float get_speed_scale() const { return _speed_scale; }

    // 自动播放
    void set_autoplay(const std::string& name) { _autoplay = name; }
    std::string get_autoplay() const { return _autoplay; }

    // 动画混合
    void set_blend_time(float time) { _blend_time = time; }
    float get_blend_time() const { return _blend_time; }

    // 回调
    void animation_finished(const std::string& anim_name);

    // 音频同步
    void set_audio_track_enabled(bool enabled) { _audio_track_enabled = enabled; }
    bool is_audio_track_enabled() const { return _audio_track_enabled; }

    void set_audio_track_pitch_scale(float scale) { _audio_track_pitch_scale = scale; }
    float get_audio_track_pitch_scale() const { return _audio_track_pitch_scale; }

    // 更新
    void update(float delta);

private:
    std::unordered_map<std::string, Ref<Animation>> _animations;
    std::vector<std::string> _animation_queued;
    std::string _current_animation_name;
    std::string _autoplay;

    Ref<Animation> _current_animation;

    bool _playing = false;
    bool _paused = false;
    float _current_position = 0.0f;
    float _speed_scale = 1.0f;
    float _blend_time = 0.0f;

    bool _audio_track_enabled = false;
    float _audio_track_pitch_scale = 1.0f;

    bool _from_end = false;

    void _play_animation(const std::string& name);
    void _advance(float delta);
};

} // namespace MyEngine
