#pragma once

#include "core/containers/String.h"
#include "core/containers/HashMap.h"
#include "core/variant/Variant.h"
#include <functional>
#include <vector>

namespace MyEngine {

class Object;

// 类信息
struct ClassInfo {
    String class_name;
    String parent_class;
    Object* (*creator)();
    HashMap<String, std::function<void(Object*)>> setters;
    HashMap<String, std::function<Variant(const Object*)>> getters;
    std::vector<String> method_names;
    HashMap<String, std::function<Variant(Object*, const Vector<Variant>&)>> methods;
};

// 类注册表 - 运行时类注册与反射
class ClassDB {
public:
    static ClassDB& get_instance();

    // 注册类
    template<typename T>
    void register_class(const String& class_name, const String& parent_class = "");

    // 注册属性
    template<typename T, typename V>
    void register_property(const String& class_name, const String& prop_name,
                           V (T::* getter)() const, void (T::* setter)(V));

    // 注册方法
    template<typename T, typename R, typename... Args>
    void register_method(const String& class_name, const String& method_name,
                         R (T::* method)(Args...));

    // 查询
    const ClassInfo* get_class(const String& class_name) const;
    bool class_exists(const String& class_name) const;
    bool is_parent_class(const String& child, const String& parent) const;

    // 创建实例
    Object* instantiate(const String& class_name);

    // 获取所有类
    std::vector<String> get_class_list() const;

private:
    ClassDB() = default;
    ~ClassDB() = default;

    HashMap<String, ClassInfo> _classes;
};

} // namespace MyEngine

#include "ClassDB.inl"
