#pragma once

#include "core/object/Resource.h"
#include "core/math/Vector3.h"
#include <string>
#include <vector>

namespace MyEngine {

// 音频格式
enum class AudioFormat {
    UNKNOWN,
    WAV,
    OGG,
    MP3
};

// 音频流
class AudioStream : public Resource {
public:
    AudioStream();
    ~AudioStream() override;

    const char* get_class_name() const override { return "AudioStream"; }

    // 格式
    void set_format(AudioFormat format) { _format = format; }
    AudioFormat get_format() const { return _format; }

    // 采样率
    void set_mix_rate(int rate) { _mix_rate = rate; }
    int get_mix_rate() const { return _mix_rate; }

    // 声道数
    void set_channels(int channels) { _channels = channels; }
    int get_channels() const { return _channels; }

    // 时长
    void set_length(float length) { _length = length; }
    float get_length() const { return _length; }

    // 数据
    void set_data(const std::vector<float>& data) { _data = data; }
    const std::vector<float>& get_data() const { return _data; }

    // 加载/保存
    bool load(const std::string& path);
    bool save(const std::string& path);

private:
    AudioFormat _format = AudioFormat::UNKNOWN;
    int _mix_rate = 44100;
    int _channels = 2;
    float _length = 0.0f;
    std::vector<float> _data;
};

// 音频流播放器
class AudioStreamPlayer : public RefCounted {
public:
    AudioStreamPlayer();
    ~AudioStreamPlayer() override;

    const char* get_class_name() const override { return "AudioStreamPlayer"; }

    // 流设置
    void set_stream(Ref<AudioStream> stream) { _stream = stream; }
    Ref<AudioStream> get_stream() const { return _stream; }

    // 播放控制
    void play();
    void stop();
    void pause();

    bool is_playing() const { return _playing; }
    bool is_paused() const { return _paused; }

    // 位置
    void set_position(float position) { _position = position; }
    float get_position() const { return _position; }

    // 音量
    void set_volume(float volume) { _volume = volume; }
    float get_volume() const { return _volume; }

    void set_volume_db(float db) { _volume = std::pow(10.0f, db / 20.0f); }
    float get_volume_db() const { return 20.0f * std::log10(_volume); }

    // 音调
    void set_pitch_scale(float scale) { _pitch_scale = scale; }
    float get_pitch_scale() const { return _pitch_scale; }

    // 3D 空间音频
    void set_unit_size(float size) { _unit_size = size; }
    float get_unit_size() const { return _unit_size; }

    void set_max_distance(float distance) { _max_distance = distance; }
    float get_max_distance() const { return _max_distance; }

    void set_attenuation_model(int model) { _attenuation_model = model; }
    int get_attenuation_model() const { return _attenuation_model; }

    void set_position_3d(const Vector3& pos) { _position_3d = pos; }
    Vector3 get_position_3d() const { return _position_3d; }

    void set_listener_attenuation(float attenuation) { _listener_attenuation = attenuation; }
    float get_listener_attenuation() const { return _listener_attenuation; }

    // 循环
    void set_loop(bool loop) { _loop = loop; }
    bool is_looping() const { return _loop; }

    void set_autoplay(bool autoplay) { _autoplay = autoplay; }
    bool is_autoplay() const { return _autoplay; }

    // 随机播放
    void set_random_pitch_factor(float factor) { _random_pitch_factor = factor; }
    float get_random_pitch_factor() const { return _random_pitch_factor; }

    // 更新
    void update(float delta);

private:
    Ref<AudioStream> _stream;
    bool _playing = false;
    bool _paused = false;
    float _position = 0.0f;
    float _volume = 1.0f;
    float _pitch_scale = 1.0f;
    float _unit_size = 1.0f;
    float _max_distance = 100.0f;
    int _attenuation_model = 0;
    Vector3 _position_3d;
    float _listener_attenuation = 1.0f;
    bool _loop = false;
    bool _autoplay = false;
    float _random_pitch_factor = 0.0f;
    float _start_pitch = 1.0f;
};

} // namespace MyEngine
