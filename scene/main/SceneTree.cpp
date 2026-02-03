#include "SceneTree.h"
#include "Node.h"
#include "os/OS.h"

namespace MyEngine {

SceneTree* SceneTree::_singleton = nullptr;

SceneTree::SceneTree() {
    _singleton = this;
    _name = "SceneTree";
}

SceneTree::~SceneTree() {
    if (_root) {
        _root->set_parent(nullptr);
        _root->unreference();
        _root = nullptr;
    }
    _singleton = nullptr;
}

void SceneTree::set_root(Node* root) {
    if (_root) {
        _root->set_parent(nullptr);
        _root->unreference();
    }
    _root = root;
    if (root) {
        root->set_parent(static_cast<Object*>(this));
        root->reference();
        _notify_nodes(root, NodeNotification::ENTER_TREE);
    }
}

void SceneTree::set_pause(bool pause) {
    if (_paused == pause) return;
    _paused = pause;
    // 通知所有节点
    if (_root) {
        _notify_nodes(_root.get(), pause ? NodeNotification::EXIT_TREE : NodeNotification::ENTER_TREE);
    }
}

void SceneTree::main_loop() {
    double last_time = OS::get_seconds();
    double current_time = last_time;

    while (!_quit) {
        current_time = OS::get_seconds();
        double frame_time = current_time - last_time;
        last_time = current_time;

        // 限制最大帧时间，避免卡顿时出问题
        if (frame_time > 0.5) frame_time = 0.5;

        _delta_time = (float)frame_time * _time_scale;
        _frame_count++;

        _process_frame();
        _physics_process_frame();
        _input_frame();
    }
}

void SceneTree::quit() {
    _quit = true;
}

void SceneTree::add_idle_callback(std::function<void()> callback) {
    _idle_callbacks.push_back(callback);
}

void SceneTree::_process_frame() {
    // 执行空闲回调
    for (auto& callback : _idle_callbacks) {
        callback();
    }
    _idle_callbacks.clear();

    // 处理节点更新
    if (_root) {
        bool process_enabled = !_paused;
        _process_nodes(_root.get(), _delta_time, process_enabled);
    }
}

void SceneTree::_physics_process_frame() {
    _time_accumulator += _delta_time;
    while (_time_accumulator >= _fixed_delta_time) {
        if (_root) {
            bool physics_enabled = !_paused;
            _physics_process_nodes(_root.get(), _fixed_delta_time, physics_enabled);
        }
        _time_accumulator -= _fixed_delta_time;
    }
}

void SceneTree::_input_frame() {
    // 输入处理
    Input::poll_events();

    if (_root) {
        // 传递输入事件到节点
    }
}

void SceneTree::_notify_nodes(Node* node, int notification) {
    if (!node) return;
    node->_notification(notification);

    for (Object* child_obj : node->get_children()) {
        Node* child = static_cast<Node*>(child_obj);
        _notify_nodes(child, notification);
    }
}

void SceneTree::_process_nodes(Node* node, float delta, bool process_enabled) {
    if (!node) return;

    if (process_enabled && node->is_processing()) {
        node->_process(delta);
    }

    for (Object* child_obj : node->get_children()) {
        Node* child = static_cast<Node*>(child_obj);
        _process_nodes(child, delta, process_enabled);
    }
}

void SceneTree::_physics_process_nodes(Node* node, float delta, bool physics_enabled) {
    if (!node) return;

    if (physics_enabled && node->is_physics_processing()) {
        node->_physics_process(delta);
    }

    for (Object* child_obj : node->get_children()) {
        Node* child = static_cast<Node*>(child_obj);
        _physics_process_nodes(child, delta, physics_enabled);
    }
}

} // namespace MyEngine
