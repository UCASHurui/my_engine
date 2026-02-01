#include "AudioServer.h"

namespace MyEngine {

AudioServer* AudioServer::_singleton = nullptr;

AudioServer::AudioServer() {
    _singleton = this;
}

AudioServer::~AudioServer() {
    if (_singleton == this) {
        _singleton = nullptr;
    }
}

void AudioServer::initialize() {
    _initialized = true;
}

void AudioServer::shutdown() {
    _initialized = false;
}

void AudioServer::add_stream_player(AudioStreamPlayer* player) {
    std::lock_guard<std::mutex> lock(_mutex);
    _players.push_back(player);
}

void AudioServer::remove_stream_player(AudioStreamPlayer* player) {
    std::lock_guard<std::mutex> lock(_mutex);
    _players.erase(
        std::remove(_players.begin(), _players.end(), player),
        _players.end()
    );
}

void AudioServer::start_input_capturing() {
    _input_capturing = true;
}

void AudioServer::stop_input_capturing() {
    _input_capturing = false;
}

void AudioServer::set_listener_transform(const Vector3& pos, const Vector3& forward, const Vector3& up) {
    _listener_position = pos;
    _listener_forward = forward;
    _listener_up = up;
}

void AudioServer::get_listener_transform(Vector3& pos, Vector3& forward, Vector3& up) const {
    pos = _listener_position;
    forward = _listener_forward;
    up = _listener_up;
}

void AudioServer::update(float delta) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto* player : _players) {
        player->update(delta);
    }
}

void AudioServer::set_effect_enabled(int effect_id, bool enabled) {
    if (effect_id >= 0 && effect_id < 8) {
        _effect_enabled[effect_id] = enabled;
    }
}

bool AudioServer::is_effect_enabled(int effect_id) const {
    if (effect_id >= 0 && effect_id < 8) {
        return _effect_enabled[effect_id];
    }
    return false;
}

} // namespace MyEngine
