#pragma once

#include "object/RefCounted.h"
#include "math/Vector3.h"
#include "AudioStream.h"
#include <vector>
#include <mutex>

namespace MyEngine {

// 音频服务器
class AudioServer : public RefCounted {
public:
    AudioServer();
    ~AudioServer() override;

    const char* get_class_name() const override { return "AudioServer"; }

    // 单例
    static AudioServer* get() { return _singleton; }

    // 初始化/清理
    void initialize();
    void shutdown();

    // 主音量
    void set_volume(float volume) { _volume = volume; }
    float get_volume() const { return _volume; }

    // 混音器
    void add_stream_player(AudioStreamPlayer* player);
    void remove_stream_player(AudioStreamPlayer* player);

    // 录音
    void start_input_capturing();
    void stop_input_capturing();
    bool is_input_capturing() const { return _input_capturing; }

    // 3D 监听器
    void set_listener_position(const Vector3& pos) { _listener_position = pos; }
    Vector3 get_listener_position() const { return _listener_position; }

    void set_listener_velocity(const Vector3& vel) { _listener_velocity = vel; }
    Vector3 get_listener_velocity() const { return _listener_velocity; }

    void set_listener_transform(const Vector3& pos, const Vector3& forward, const Vector3& up);
    void get_listener_transform(Vector3& pos, Vector3& forward, Vector3& up) const;

    // 更新
    void update(float delta);

    // 效果
    void set_effect_enabled(int effect_id, bool enabled);
    bool is_effect_enabled(int effect_id) const;

    // 录音数据
    const std::vector<float>& get_input_buffer() const { return _input_buffer; }

private:
    static AudioServer* _singleton;

    float _volume = 1.0f;
    bool _initialized = false;
    bool _input_capturing = false;
    std::mutex _mutex;

    Vector3 _listener_position;
    Vector3 _listener_velocity;
    Vector3 _listener_forward;
    Vector3 _listener_up;

    std::vector<AudioStreamPlayer*> _players;
    std::vector<float> _input_buffer;

    // 效果
    bool _effect_enabled[8] = {false};
};

} // namespace MyEngine
