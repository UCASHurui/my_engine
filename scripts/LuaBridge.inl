#pragma once

#include "LuaBridge.h"

namespace MyEngine {

template<typename... Args>
void LuaBridge::register_function(const String& name, std::function<void(Args...)> func) {
    if (!_state) return;

    lua_pushlightuserdata(_state, new std::function<void(Args...)>(func));
    lua_pushcclosure(_state, [](lua_State* L) -> int {
        std::function<void(Args...)>* func =
            reinterpret_cast<std::function<void(Args...)*>(lua_touserdata(L, lua_upvalueindex(1)));

        if constexpr (sizeof...(Args) == 0) {
            (*func)();
        } else if constexpr (sizeof...(Args) == 1) {
            (*func)(to_variant(L, 1).get_value<Args>());
        } else if constexpr (sizeof...(Args) == 2) {
            (*func)(to_variant(L, 1).get_value<Args>(),
                    to_variant(L, 2).get_value<Args>());
        }
        return 0;
    }, 1);

    lua_setglobal(_state, name.c_str());
}

template<typename T>
void LuaBridge::register_class(const String& name) {
    if (!_state) return;

    // 创建元表
    luaL_newmetatable(_state, name.c_str());

    // 设置 __index
    lua_pushvalue(_state, -1);
    lua_setfield(_state, -2, "__index");

    // 注册到全局
    lua_setglobal(_state, name.c_str());
}

} // namespace MyEngine
