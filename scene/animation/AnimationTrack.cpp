#include "AnimationTrack.h"

namespace MyEngine {

AnimationTrack::AnimationTrack() = default;
AnimationTrack::~AnimationTrack() = default;

int AnimationTrack::get_key_index(float time, bool is_end) const {
    int count = 0;
    switch (_type) {
        case TrackType::TRANSFORM:
        case TrackType::POSITION:
        case TrackType::ROTATION:
        case TrackType::SCALE:
            count = _transform_keys.size();
            break;
        case TrackType::FLOAT:
        case TrackType::VALUE:
            count = _float_keys.size();
            break;
        case TrackType::COLOR:
            count = _color_keys.size();
            break;
        case TrackType::BOOL:
            count = _bool_keys.size();
            break;
        default:
            return 0;
    }

    if (count == 0) return 0;

    int idx = 0;
    for (int i = 0; i < count; i++) {
        float t;
        switch (_type) {
            case TrackType::TRANSFORM:
            case TrackType::POSITION:
            case TrackType::ROTATION:
            case TrackType::SCALE:
                t = _transform_keys[i].time;
                break;
            case TrackType::FLOAT:
            case TrackType::VALUE:
                t = _float_keys[i].time;
                break;
            case TrackType::COLOR:
                t = _color_keys[i].time;
                break;
            case TrackType::BOOL:
                t = _bool_keys[i].time;
                break;
            default:
                t = 0;
        }

        if (is_end) {
            if (t > time) {
                idx = i;
                break;
            }
        } else {
            if (t >= time) {
                idx = i;
                break;
            }
        }
    }

    return idx;
}

} // namespace MyEngine
