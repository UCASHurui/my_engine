#pragma once

#include "Node2D.h"
#include "containers/HashMap.h"
#include "Texture2D.h"

namespace MyEngine {

// 精灵节点 - 显示 2D 图像
class Sprite2D : public Node2D {

public:
    Sprite2D();
    virtual ~Sprite2D();

    virtual const char* get_class_name() const override { return "Sprite2D"; }

    // 纹理
    void set_texture(Texture2D* texture);
    Texture2D* get_texture() const { return _texture.get(); }

    // 区域
    void set_region(const Rect2& region);
    Rect2 get_region() const { return _region; }
    void set_region_enabled(bool enabled) { _region_enabled = enabled; }
    bool is_region_enabled() const { return _region_enabled; }

    // 翻转
    void set_flip_h(bool flip) { _flip_h = flip; }
    bool is_flipped_h() const { return _flip_h; }
    void set_flip_v(bool flip) { _flip_v = flip; }
    bool is_flipped_v() const { return _flip_v; }

    // 动画
    void set_hframes(int frames) { _hframes = frames; }
    int get_hframes() const { return _hframes; }
    void set_vframes(int frames) { _vframes = frames; }
    int get_vframes() const { return _vframes; }
    void set_frame(int frame);
    int get_frame() const { return _frame; }
    void set_frame_coords(const Vector2& coords);
    Vector2 get_frame_coords() const;

    // 裁剪
    void set_clip_above(bool clip) { _clip_above = clip; }
    bool is_clip_above() const { return _clip_above; }

protected:
    virtual void _draw() override;

private:
    Ref<Texture2D> _texture;
    Rect2 _region;
    bool _region_enabled = false;
    bool _flip_h = false;
    bool _flip_v = false;
    int _hframes = 1;
    int _vframes = 1;
    int _frame = 0;
    bool _clip_above = false;
};

// 动画精灵
class AnimatedSprite2D : public Sprite2D {

public:
    AnimatedSprite2D();
    virtual ~AnimatedSprite2D();

    virtual const char* get_class_name() const override { return "AnimatedSprite2D"; }

    // 动画
    void play(const String& animation = "");
    void pause();
    void stop();
    bool is_playing() const { return _playing; }

    void set_current_animation(const String& anim) { _current_animation = anim; }
    String get_current_animation() const { return _current_animation; }

    void set_animation_speed(float speed) { _animation_speed = speed; }
    float get_animation_speed() const { return _animation_speed; }

    void add_frame(const String& animation, Texture2D* texture, float delay = 1.0f);
    void add_frame_by_filename(const String& animation, const String& path, float delay = 1.0f);
    void clear_animations();
    Vector<String> get_animations() const;

    void set_autoplay(const String& anim) { _autoplay = anim; }
    String get_autoplay() const { return _autoplay; }

protected:
    virtual void _process(float delta) override;

private:
    struct Frame {
        Ref<Texture2D> texture;
        float delay;
    };

    struct Animation {
        Vector<Frame> frames;
        float speed_scale = 1.0f;
    };

    HashMap<String, Animation> _animations;
    String _current_animation;
    String _autoplay;
    bool _playing = false;
    float _animation_speed = 1.0f;
    float _time = 0.0f;
    int _frame_index = 0;

    Animation* _get_animation(const String& name);
};

} // namespace MyEngine
