#include "Node.h"
#include "SceneTree.h"

namespace MyEngine {

Node::Node() : _name("Node") {}

Node::~Node() {
    // 清理子节点
    while (!_children.empty()) {
        Node* child = _children.back();
        child->_parent = nullptr;
        _children.pop_back();
        child->unreference();
    }

    // 从父节点移除
    if (_parent) {
        _parent->remove_child(this);
    }
}

void Node::add_child(Node* child) {
    if (!child) return;

    if (child->_parent) {
        child->_parent->remove_child(child);
    }

    child->_parent = this;
    _children.push_back(child);
    _on_child_added(child);
}

void Node::remove_child(Node* child) {
    if (!child) return;

    for (size_t i = 0; i < _children.size(); i++) {
        if (_children[i] == child) {
            _children.erase(i);
            child->_parent = nullptr;
            _on_child_removed(child);
            return;
        }
    }
}

void Node::remove_child_at(int index) {
    if (index >= 0 && index < (int)_children.size()) {
        Node* child = _children[index];
        _children.erase(index);
        child->_parent = nullptr;
        _on_child_removed(child);
    }
}

void Node::remove_all_children() {
    while (!_children.empty()) {
        remove_child(_children[0]);
    }
}

Node* Node::get_child(int index) const {
    if (index >= 0 && index < (int)_children.size()) {
        return _children[index];
    }
    return nullptr;
}

Node* Node::find_child(const String& name, bool recursive) const {
    for (Node* child : _children) {
        if (child->get_name() == name) {
            return child;
        }
        if (recursive) {
            Node* found = child->find_child(name, true);
            if (found) return found;
        }
    }
    return nullptr;
}

Vector<Node*> Node::find_children(const String& name, bool recursive) const {
    Vector<Node*> result;
    for (Node* child : _children) {
        if (child->get_name() == name) {
            result.push_back(child);
        }
        if (recursive) {
            Vector<Node*> sub = child->find_children(name, true);
            for (Node* c : sub) {
                result.push_back(c);
            }
        }
    }
    return result;
}

void Node::add_to_group(const String& group) {
    if (!is_in_group(group)) {
        _groups.push_back(group);
    }
}

void Node::remove_from_group(const String& group) {
    for (size_t i = 0; i < _groups.size(); i++) {
        if (_groups[i] == group) {
            _groups.erase(i);
            return;
        }
    }
}

bool Node::is_in_group(const String& group) const {
    for (const String& g : _groups) {
        if (g == group) return true;
    }
    return false;
}

Transform2D Node::get_global_transform_2d() const {
    Transform2D global = _transform_2d;
    if (_parent) {
        global = _parent->get_global_transform_2d() * global;
    }
    return global;
}

Transform3D Node::get_global_transform_3d() const {
    Transform3D global = _transform_3d;
    if (_parent) {
        global = _parent->get_global_transform_3d() * global;
    }
    return global;
}

int Node::get_depth() const {
    int depth = 0;
    Node* p = _parent;
    while (p) {
        depth++;
        p = p->_parent;
    }
    return depth;
}

Node* Node::get_root() const {
    Node* p = const_cast<Node*>(this);
    while (p->_parent) {
        p = p->_parent;
    }
    return p;
}

void Node::set_process(bool enabled) {
    _processing = enabled;
}

void Node::set_physics_process(bool enabled) {
    _physics_processing = enabled;
}

void Node::_ready() {}

void Node::_process(float delta) {
    (void)delta;
}

void Node::_physics_process(float delta) {
    (void)delta;
}

void Node::_enter_tree() {}

void Node::_exit_tree() {}

void Node::_notification(int notification) {
    switch (notification) {
        case NodeNotification::READY:
            _ready();
            break;
        case NodeNotification::PROCESS:
            _process(0.016f); // TODO: 传入实际 delta time
            break;
    }
}

void Node::_on_child_added(Node* child) {
    (void)child;
}

void Node::_on_child_removed(Node* child) {
    (void)child;
}

void Node::_on_parent_changed(Node* old_parent) {
    (void)old_parent;
}

void Node::_update_scene_tree() {
    for (Node* child : _children) {
        child->_scene_tree = _scene_tree;
        child->_update_scene_tree();
    }
}

} // namespace MyEngine
