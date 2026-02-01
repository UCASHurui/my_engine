#pragma once

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

namespace MyEngine {

// Lua 绑定工具
class LuaBridge {
public:
    static void initialize(lua_State* L);
    static void shutdown(lua_State* L);

    // 注册 C++ 函数到 Lua
    template<typename... Args>
    static void register_function(const String& name, std::function<void(Args...)> func);

    // 注册类
    template<typename T>
    static void register_class(const String& name);

    // 注册属性
    template<typename T, typename V>
    static void register_property(const String& class_name, const String& prop_name,
                                  V (T::* getter)(), void (T::* setter)(V));

    // 调用 Lua 函数
    static bool call(const String& func_name, int nargs = 0, int nresults = 0);

    // 栈操作
    static void push_variant(lua_State* L, const Variant& v);
    static Variant to_variant(lua_State* L, int index);

    // 错误处理
    static int lua_error_handler(lua_State* L);

private:
    static lua_State* _state;
};

} // namespace MyEngine

#include "LuaBridge.inl"
