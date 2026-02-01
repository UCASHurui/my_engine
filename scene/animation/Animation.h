#pragma once

#include "core/object/Resource.h"
#include "AnimationTrack.h"
#include <unordered_map>

namespace MyEngine {

// 动画循环模式
enum class LoopMode {
    LOOP_DISABLED,
    LOOP_ENABLED,
    LOOP_PINGPONG,
    LOOP_PINGPONG_ENABLED
};

// 动画资源
class Animation : public Resource {
public:
    Animation();
    ~Animation() override;

    const char* get_class_name() const override { return "Animation"; }

    // 时长
    void set_length(float length) { _length = length; }
    float get_length() const { return _length; }

    // 循环模式
    void set_loop_mode(LoopMode mode) { _loop_mode = mode; }
    LoopMode get_loop_mode() const { return _loop_mode; }

    // BPM 和节拍
    void set_bpm(float bpm) { _bpm = bpm; }
    float get_bpm() const { return _bpm; }

    void set_beats_per_bar(int beats) { _beats_per_bar = beats; }
    int get_beats_per_bar() const { return _beats_per_bar; }

    void set_cue_points(const std::vector<std::string>& cues) { _cue_points = cues; }
    const std::vector<std::string>& get_cue_points() const { return _cue_points; }

    void add_cue_point(const std::string& cue, float time) {
        _cue_points.push_back(cue);
        _cue_times[cue] = time;
    }

    float get_cue_time(const std::string& cue) const {
        auto it = _cue_times.find(cue);
        return it != _cue_times.end() ? it->second : -1.0f;
    }

    // 轨道管理
    int get_track_count() const { return _tracks.size(); }

    int add_track(Ref<AnimationTrack> track);
    void remove_track(int idx);
    void remove_track(const std::string& path);

    Ref<AnimationTrack> get_track(int idx) const { return idx < (int)_tracks.size() ? _tracks[idx] : nullptr; }
    Ref<AnimationTrack> get_track(const std::string& path) const;

    int find_track(const std::string& path) const;

    // 复制
    Ref<Animation> duplicate() const;

    // 获取所有轨道
    const std::vector<Ref<AnimationTrack>>& get_tracks() const { return _tracks; }

private:
    float _length = 1.0f;
    LoopMode _loop_mode = LoopMode::LOOP_ENABLED;
    float _bpm = 60.0f;
    int _beats_per_bar = 4;
    std::vector<std::string> _cue_points;
    std::unordered_map<std::string, float> _cue_times;
    std::vector<Ref<AnimationTrack>> _tracks;
};

} // namespace MyEngine
