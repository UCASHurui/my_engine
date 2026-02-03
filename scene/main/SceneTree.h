#pragma once

#include "object/Object.h"
#include "os/Input.h"
#include "math/Vector2.h"
#include "Node.h"
#include <functional>

namespace MyEngine {

// 场景树 - 管理所有场景节点
class SceneTree : public Node {

public:
    SceneTree();
    virtual ~SceneTree();

    virtual const char* get_class_name() const override { return "SceneTree"; }

    // 根节点
    void set_root(Node* root);
    Node* get_root() const { return _root.get(); }

    // 暂停
    void set_pause(bool pause);
    bool is_paused() const { return _paused; }

    // 帧循环
    void main_loop();
    void quit();

    // 通知
    void add_idle_callback(std::function<void()> callback);

    // 时间
    float get_delta_time() const { return _delta_time; }
    float get_fixed_delta_time() const { return _fixed_delta_time; }
    void set_time_scale(float scale) { _time_scale = scale; }
    float get_time_scale() const { return _time_scale; }

    // 网络
    void set_network_peer(class NetworkPeer* peer);
    NetworkPeer* get_network_peer() const { return _network_peer; }

    // 输入模式
    void set_input_mode(InputMode mode) { _input_mode = mode; }
    InputMode get_input_mode() const { return _input_mode; }

    // 触摸/鼠标模式
    void set_use_input_unhandled(bool use) { _use_input_unhandled = use; }
    bool is_using_input_unhandled() const { return _use_input_unhandled; }

    // 单例
    static SceneTree* get_singleton() { return _singleton; }

private:
    static SceneTree* _singleton;

    Ref<Node> _root;
    bool _paused = false;
    bool _quit = false;
    float _delta_time = 0.016f;
    float _fixed_delta_time = 0.02f;
    float _time_scale = 1.0f;
    double _time_accumulator = 0.0;
    InputMode _input_mode = InputMode::ALL;
    bool _use_input_unhandled = false;
    NetworkPeer* _network_peer = nullptr;

    Vector<std::function<void()>> _idle_callbacks;
    uint64_t _frame_count = 0;
    double _current_time = 0.0;

    void _process_frame();
    void _physics_process_frame();
    void _input_frame();
    void _draw_frame();

    void _notify_nodes(Node* node, int notification);
    void _process_nodes(Node* node, float delta, bool process_enabled);
    void _physics_process_nodes(Node* node, float delta, bool physics_enabled);
};

} // namespace MyEngine
