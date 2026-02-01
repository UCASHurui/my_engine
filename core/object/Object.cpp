#include "Object.h"
#include "ClassDB.h"

namespace MyEngine {

uint32_t Object::_next_instance_id = 0;

Object::Object() : _instance_id(++_next_instance_id) {}

Object::~Object() {
    // 从父节点移除
    if (_parent) {
        _parent->remove_child(this);
    }

    // 删除所有子节点
    while (!_children.empty()) {
        Object* child = _children.back();
        child->_parent = nullptr;
        _children.pop_back();
        child->unreference();
    }
}

void Object::set_parent(Object* parent) {
    if (_parent == parent) return;

    Object* old_parent = _parent;
    _parent = parent;

    if (old_parent) {
        old_parent->_on_child_removed(this);
    }
    if (parent) {
        parent->_on_child_added(this);
    }

    _on_parent_changed(old_parent);
}

void Object::add_child(Object* child) {
    if (!child) return;
    if (child->_parent == this) return;

    child->set_parent(this);
    _children.push_back(child);
}

void Object::remove_child(Object* child) {
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

void Object::emit_signal(const String& signal, const Variant& arg1, const Variant& arg2) {
    (void)signal; (void)arg1; (void)arg2;
}

} // namespace MyEngine
