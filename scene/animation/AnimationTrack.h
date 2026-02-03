#pragma once

#include "object/Resource.h"
#include "math/Vector3.h"
#include "math/Transform.h"
#include "math/Color.h"
#include <string>
#include <vector>

namespace MyEngine {

// 轨道类型
enum class TrackType {
    POSITION,
    ROTATION,
    SCALE,
    TRANSFORM,
    VALUE,
    BOOL,
    FLOAT,
    COLOR,
    CALLABLE
};

// 插值模式
enum class InterpolationMode {
    HOLD,
    LINEAR,
    CUBIC,
    NEAREST
};

// 轨道关键帧
struct TransformKey {
    float time = 0.0f;
    Transform3D value;
    float transition = 0.5f;
};

struct FloatKey {
    float time = 0.0f;
    float value = 0.0f;
    float transition = 0.5f;
};

struct ColorKey {
    float time = 0.0f;
    Color value;
    float transition = 0.5f;
};

struct BoolKey {
    float time = 0.0f;
    bool value = false;
};

// 动画轨道
class AnimationTrack : public RefCounted {
public:
    AnimationTrack();
    ~AnimationTrack() override;

    const char* get_class_name() const override { return "AnimationTrack"; }

    // 轨道类型
    void set_track_type(TrackType type) { _type = type; }
    TrackType get_track_type() const { return _type; }

    // 路径
    void set_path(const std::string& path) { _path = path; }
    std::string get_path() const { return _path; }

    // 插值模式
    void set_interpolation_mode(InterpolationMode mode) { _interpolation = mode; }
    InterpolationMode get_interpolation_mode() const { return _interpolation; }

    // 关键帧
    void add_transform_key(const TransformKey& key) { _transform_keys.push_back(key); }
    void add_float_key(const FloatKey& key) { _float_keys.push_back(key); }
    void add_color_key(const ColorKey& key) { _color_keys.push_back(key); }
    void add_bool_key(const BoolKey& key) { _bool_keys.push_back(key); }

    const std::vector<TransformKey>& get_transform_keys() const { return _transform_keys; }
    const std::vector<FloatKey>& get_float_keys() const { return _float_keys; }
    const std::vector<ColorKey>& get_color_keys() const { return _color_keys; }
    const std::vector<BoolKey>& get_bool_keys() const { return _bool_keys; }

    // 获取关键帧索引
    int get_key_index(float time, bool is_end = false) const;

    // 清空
    void clear() {
        _transform_keys.clear();
        _float_keys.clear();
        _color_keys.clear();
        _bool_keys.clear();
    }

private:
    TrackType _type = TrackType::POSITION;
    std::string _path;
    InterpolationMode _interpolation = InterpolationMode::LINEAR;

    std::vector<TransformKey> _transform_keys;
    std::vector<FloatKey> _float_keys;
    std::vector<ColorKey> _color_keys;
    std::vector<BoolKey> _bool_keys;
};

} // namespace MyEngine
