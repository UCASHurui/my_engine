#pragma once

#include "ClassDB.h"
#include "core/variant/Variant.h"

namespace MyEngine {

template<typename T>
void ClassDB::register_class(const String& class_name, const String& parent_class) {
    ClassInfo info;
    info.class_name = class_name;
    info.parent_class = parent_class;
    info.creator = []() -> Object* { return new T(); };
    _classes.insert(class_name, info);
}

template<typename T, typename V>
void ClassDB::register_property(const String& class_name, const String& prop_name,
                                V (T::* getter)() const, void (T::* setter)(V)) {
    auto it = _classes.find(class_name);
    if (it != _classes.end()) {
        ClassInfo& info = (*it).value;
        info.setters.insert(prop_name, [setter](Object* obj) {
            T* t = static_cast<T*>(obj);
            (t->*setter)(V());
        });
        info.getters.insert(prop_name, [getter](const Object* obj) -> Variant {
            const T* t = static_cast<const T*>(obj);
            return Variant((t->*getter)());
        });
    }
}

template<typename T, typename R, typename... Args>
void ClassDB::register_method(const String& class_name, const String& method_name,
                              R (T::* method)(Args...)) {
    auto it = _classes.find(class_name);
    if (it != _classes.end()) {
        ClassInfo& info = (*it).value;
        info.method_names.push_back(method_name);
        info.methods.insert(method_name, [method](Object* obj, const Vector<Variant>& args) -> Variant {
            T* t = static_cast<T*>(obj);
            (void)args;
            if constexpr (sizeof...(Args) == 0) {
                if constexpr (std::is_void_v<R>) {
                    (t->*method)();
                    return Variant();
                } else {
                    return Variant((t->*method)());
                }
            } else {
                // Simplified: no args conversion for now
                return Variant();
            }
        });
    }
}

} // namespace MyEngine
