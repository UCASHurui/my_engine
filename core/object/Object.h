#pragma once

#include "core/variant/Variant.h"
#include "core/containers/String.h"
#include "core/containers/Vector.h"
#include <functional>
#include <vector>

namespace MyEngine {

// 对象标志位
enum class ObjectFlag : uint32_t {
    NONE = 0,
    DELETE_DISABLED = 1 << 0,
    BLOCK_NOTIFICATION = 1 << 1,
    REQUIRE_PROCESSING = 1 << 2
};

// 对象基类 - 所有引擎对象的父类
class Object {
public:
    Object();
    virtual ~Object();

    // 类信息
    virtual const char* get_class_name() const { return "Object"; }
    virtual bool is_class(const char* name) const { return strcmp(name, "Object") == 0; }

    // 标识
    uint32_t get_instance_id() const { return _instance_id; }
    const String& get_name() const { return _name; }
    void set_name(const String& name) { _name = name; }

    // 父子关系
    Object* get_parent() const { return _parent; }
    void set_parent(Object* parent);
    void add_child(Object* child);
    void remove_child(Object* child);
    Vector<Object*>& get_children() { return _children; }
    const Vector<Object*>& get_children() const { return _children; }

    // 标志位
    void set_flag(ObjectFlag flag) { _flags |= (uint32_t)flag; }
    void clear_flag(ObjectFlag flag) { _flags &= ~(uint32_t)flag; }
    bool has_flag(ObjectFlag flag) const { return (_flags & (uint32_t)flag) != 0; }

    // 引用计数
    int get_reference_count() const { return _ref_count; }
    void reference() { _ref_count++; }
    void unreference() {
        _ref_count--;
        if (_ref_count <= 0) {
            delete this;
        }
    }

    // 生命周期
    virtual void _ready() {}
    virtual void _process(float delta) { (void)delta; }
    virtual void _notification(int notification) { (void)notification; }

    // 消息
    virtual void emit_signal(const String& signal, const Variant& arg1 = Variant::NIL,
                            const Variant& arg2 = Variant::NIL);

protected:
    virtual void _on_parent_changed(Object* old_parent) { (void)old_parent; }
    virtual void _on_child_added(Object* child) { (void)child; }
    virtual void _on_child_removed(Object* child) { (void)child; }

private:
    static uint32_t _next_instance_id;

    uint32_t _instance_id;
    String _name;
    Object* _parent = nullptr;
    Vector<Object*> _children;
    uint32_t _flags = 0;
    int _ref_count = 0;
};

// 引用计数智能指针
template<typename T>
class Ref {
public:
    Ref() : _ptr(nullptr) {}
    explicit Ref(T* ptr) : _ptr(ptr) {
        if (_ptr) _ptr->reference();
    }
    Ref(const Ref& other) : _ptr(other._ptr) {
        if (_ptr) _ptr->reference();
    }
    Ref(Ref&& other) noexcept : _ptr(other._ptr) {
        other._ptr = nullptr;
    }
    ~Ref() {
        if (_ptr) _ptr->unreference();
    }

    Ref& operator=(T* ptr) {
        if (_ptr) _ptr->unreference();
        _ptr = ptr;
        if (_ptr) _ptr->reference();
        return *this;
    }
    Ref& operator=(const Ref& other) {
        if (_ptr) _ptr->unreference();
        _ptr = other._ptr;
        if (_ptr) _ptr->reference();
        return *this;
    }
    Ref& operator=(Ref&& other) noexcept {
        if (_ptr) _ptr->unreference();
        _ptr = other._ptr;
        other._ptr = nullptr;
        return *this;
    }

    T* operator->() const { return _ptr; }
    T& operator*() const { return *_ptr; }
    operator bool() const { return _ptr != nullptr; }
    T* get() const { return _ptr; }
    bool is_null() const { return _ptr == nullptr; }

    void detach() {
        if (_ptr) {
            _ptr->unreference();
            _ptr = nullptr;
        }
    }

private:
    T* _ptr;
};

} // namespace MyEngine
