#include "Animation.h"

namespace MyEngine {

Animation::Animation() = default;
Animation::~Animation() = default;

int Animation::add_track(Ref<AnimationTrack> track) {
    _tracks.push_back(track);
    return _tracks.size() - 1;
}

void Animation::remove_track(int idx) {
    if (idx >= 0 && idx < (int)_tracks.size()) {
        _tracks.erase(_tracks.begin() + idx);
    }
}

void Animation::remove_track(const std::string& path) {
    for (auto it = _tracks.begin(); it != _tracks.end(); ) {
        if ((*it)->get_path() == path) {
            it = _tracks.erase(it);
        } else {
            ++it;
        }
    }
}

Ref<AnimationTrack> Animation::get_track(const std::string& path) const {
    for (const auto& track : _tracks) {
        if (track->get_path() == path) {
            return track;
        }
    }
    return nullptr;
}

int Animation::find_track(const std::string& path) const {
    for (int i = 0; i < (int)_tracks.size(); i++) {
        if (_tracks[i]->get_path() == path) {
            return i;
        }
    }
    return -1;
}

Ref<Animation> Animation::duplicate() const {
    Ref<Animation> anim = new Animation();
    anim->_length = _length;
    anim->_loop_mode = _loop_mode;
    anim->_bpm = _bpm;
    anim->_beats_per_bar = _beats_per_bar;
    anim->_cue_points = _cue_points;
    anim->_cue_times = _cue_times;

    for (const auto& track : _tracks) {
        Ref<AnimationTrack> new_track = new AnimationTrack();
        new_track->set_track_type(track->get_track_type());
        new_track->set_path(track->get_path());
        new_track->set_interpolation_mode(track->get_interpolation_mode());
        // 复制关键帧...
        anim->add_track(new_track);
    }

    return anim;
}

} // namespace MyEngine
