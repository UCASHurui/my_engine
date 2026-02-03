#pragma once

#include "Node.h"
#include "math/Rect2.h"

namespace MyEngine {

class RenderTarget;

// 视口 - 渲染的目标区域
class Viewport : public Node {
    DECLARE_CLASS(Viewport, Node)

public:
    Viewport();
    virtual ~Viewport();

    virtual const char* get_class_name() const override { return "Viewport"; }

    // 尺寸
    void set_size(const Vector2& size);
    Vector2 get_size() const { return _size; }
    void set_size_override(const Vector2& size);
    Vector2 get_size_override() const { return _size_override; }

    // 渲染目标
    void set_render_target(RenderTarget* rt);
    RenderTarget* get_render_target() const { return _render_target.get(); }

    // 相机
    void set_camera(Node* camera);
    Node* get_camera() const { return _camera.get(); }

    // 场景
    void set_world_3d(class World3D* world);
    World3D* get_world_3d() const { return _world_3d.get(); }
    void set_world_2d(class World2D* world);
    World2D* get_world_2d() const { return _world_2d.get(); }

    // 模式
    void set_update_mode(UpdateMode mode);
    UpdateMode get_update_mode() const { return _update_mode; }
    void set_clear_mode(ClearMode mode);
    ClearMode get_clear_mode() const { return _clear_mode; }

    // GUI
    void set_gui_input_to_foreground(bool enable);
    bool is_gui_input_to_foreground() const { return _gui_input_to_foreground; }

    // 嵌入
    void set_embed_subwindow(bool embed);
    bool is_embed_subwindow() const { return _embed_subwindow; }

    // 截图
    void set_screenshot_as_texture(bool enable);
    bool is_screenshot_as_texture() const { return _screenshot_as_texture; }

    // 监听
    void set_as_audio_listener(bool enable);
    bool is_audio_listener() const { return _audio_listener; }

    // 过渡
    void set_transition(Transition* transition);
    Transition* get_transition() const { return _transition.get(); }

    // 场景实例化
    Node* instance(const String& path);
    Node* instance_attach_to_node(Node* parent, const String& path);

protected:
    virtual void _input(const InputEvent& event);

private:
    Vector2 _size;
    Vector2 _size_override;
    Ref<RenderTarget> _render_target;
    Ref<Node> _camera;
    Ref<World3D> _world_3d;
    Ref<World2D> _world_2d;

    enum UpdateMode {
        UPDATE_WHEN_PARENT_VISIBLE,
        UPDATE_WHEN_PARENT_PROCESS,
        UPDATE_ALWAYS,
        UPDATE_DISABLED
    } _update_mode = UPDATE_ALWAYS;

    enum ClearMode {
        CLEAR_MODE_ALWAYS,
        CLEAR_MODE_NEVER,
        CLEAR_MODE_ONCE
    } _clear_mode = CLEAR_MODE_ALWAYS;

    bool _gui_input_to_foreground = true;
    bool _embed_subwindow = false;
    bool _screenshot_as_texture = false;
    bool _audio_listener = false;
    Ref<Transition> _transition;
};

// 更新模式
enum class UpdateMode {
    WHEN_PARENT_VISIBLE,
    WHEN_PARENT_PROCESS,
    ALWAYS,
    DISABLED
};

// 清除模式
enum class ClearMode {
    ALWAYS,
    NEVER,
    ONCE
};

// 过渡
class Transition {
public:
    virtual ~Transition() = default;
    virtual void prepare() = 0;
    virtual void transition(float t) = 0;
    virtual bool is_done() = 0;
};

} // namespace MyEngine
