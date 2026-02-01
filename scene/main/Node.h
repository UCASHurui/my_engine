#pragma once

#include "core/object/Object.h"
#include "core/math/Transform.h"

namespace MyEngine {

class SceneTree;

// 场景节点 - 引擎中最基本的组成单元
class Node : public Object {

public:
    Node();
    virtual ~Node();

    // 类名
    virtual const char* get_class_name() const override { return "Node"; }

    // 名称
    const String& get_name() const { return _name; }
    void set_name(const String& name) { _name = name; }

    // 场景树
    SceneTree* get_scene_tree() const { return _scene_tree; }

    // 父子关系
    void add_child(Node* child);
    void remove_child(Node* child);
    void remove_child_at(int index);
    void remove_all_children();
    int get_child_count() const { return (int)_children.size(); }
    Node* get_child(int index) const;
    Node* find_child(const String& name, bool recursive = true) const;
    Vector<Node*> find_children(const String& name, bool recursive = true) const;

    // 组
    void add_to_group(const String& group);
    void remove_from_group(const String& group);
    bool is_in_group(const String& group) const;
    Vector<Node*> get_group(const String& group) const;

    // 变换 (本地空间)
    const Transform2D& get_transform_2d() const { return _transform_2d; }
    void set_transform_2d(const Transform2D& t) { _transform_2d = t; }
    const Transform3D& get_transform_3d() const { return _transform_3d; }
    void set_transform_3d(const Transform3D& t) { _transform_3d = t; }

    // 世界变换 (只读)
    Transform2D get_global_transform_2d() const;
    Transform3D get_global_transform_3d() const;
    float get_global_rotation() const { return get_global_transform_2d().get_rotation(); }
    Vector2 get_global_scale() const { return get_global_transform_2d().get_scale(); }
    Vector3 get_global_scale_3d() const { return get_global_transform_3d().get_scale(); }

    // 层级
    int get_depth() const;
    Node* get_root() const;

    // 通知
    void set_process(bool enabled);
    bool is_processing() const { return _processing; }
    void set_physics_process(bool enabled);
    bool is_physics_processing() const { return _physics_processing; }

    // 生命周期
    virtual void _ready();
    virtual void _process(float delta);
    virtual void _physics_process(float delta);
    virtual void _enter_tree();
    virtual void _exit_tree();
    virtual void _notification(int notification);

    // 消息
    virtual void _on_child_added(Node* child);
    virtual void _on_child_removed(Node* child);
    virtual void _on_parent_changed(Node* old_parent);

protected:
    String _name;
    SceneTree* _scene_tree = nullptr;
    Node* _parent = nullptr;
    Vector<Node*> _children;
    Vector<String> _groups;

    Transform2D _transform_2d;
    Transform3D _transform_3d;

    bool _processing = false;
    bool _physics_processing = false;
    bool _ready_called = false;

private:
    void _update_scene_tree();
};

} // namespace MyEngine

// 通知类型
namespace NodeNotification {
    enum {
        ENTER_TREE = 10,
        EXIT_TREE = 11,
        READY = 12,
        PROCESS = 13,
        PHYSICS_PROCESS = 14,
        PARENT_CHANGED = 15,
        CHILD_ADDED = 16,
        CHILD_REMOVED = 17
    };
}
