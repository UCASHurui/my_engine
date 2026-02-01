#include "AudioStream.h"

namespace MyEngine {

AudioStream::AudioStream() = default;
AudioStream::~AudioStream() = default;

bool AudioStream::load(const std::string& path) {
    (void)path;
    return false;
}

bool AudioStream::save(const std::string& path) {
    (void)path;
    return false;
}

AudioStreamPlayer::AudioStreamPlayer() = default;
AudioStreamPlayer::~AudioStreamPlayer() = default;

void AudioStreamPlayer::play() {
    if (_stream) {
        _playing = true;
        _paused = false;
        _start_pitch = _pitch_scale * (1.0f + _random_pitch_factor * 0.1f);
    }
}

void AudioStreamPlayer::stop() {
    _playing = false;
    _paused = false;
    _position = 0.0f;
}

void AudioStreamPlayer::pause() {
    _paused = true;
}

void AudioStreamPlayer::update(float delta) {
    if (!_playing || _paused || !_stream) return;

    _position += delta * _start_pitch;

    if (_position >= _stream->get_length()) {
        if (_loop) {
            _position = 0.0f;
        } else {
            _playing = false;
            _position = 0.0f;
        }
    }
}

} // namespace MyEngine
