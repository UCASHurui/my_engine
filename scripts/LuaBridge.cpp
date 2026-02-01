#include "LuaBridge.h"
#include "core/variant/Variant.h"
#include "core/object/Object.h"
#include <cstdio>

namespace MyEngine {

lua_state* LuaBridge::_state = nullptr;

void LuaBridge::initialize(lua_State* L) {
    _state = L;

    // 打开标准库
    luaL_openlibs(L);

    // 注册基本函数
    lua_register(L, "print", [](lua_State* L) -> int {
        int n = lua_gettop(L);
        for (int i = 1; i <= n; i++) {
            const char* s = luaL_tolstring(L, i, nullptr);
            printf("%s", s);
            lua_pop(L, 1);
            if (i < n) printf("\t");
        }
        printf("\n");
        return 0;
    });

    lua_register(L, "type", [](lua_State* L) -> int {
        if (lua_gettop(L) >= 1) {
            // 简化处理
            lua_pushstring(L, "userdata");
        }
        return 1;
    });
}

void LuaBridge::shutdown(lua_State* L) {
    (void)L;
    _state = nullptr;
}

bool LuaBridge::call(const String& func_name, int nargs, int nresults) {
    if (!_state) return false;

    lua_getglobal(_state, func_name.c_str());
    if (!lua_isfunction(_state, -1)) {
        lua_pop(_state, 1);
        return false;
    }

    if (lua_pcall(_state, nargs, nresults, 0) != 0) {
        const char* err = lua_tostring(_state, -1);
        fprintf(stderr, "Lua error: %s\n", err);
        lua_pop(_state, 1);
        return false;
    }
    return true;
}

void LuaBridge::push_variant(lua_State* L, const Variant& v) {
    switch (v.get_type()) {
        case VariantType::NIL:
            lua_pushnil(L);
            break;
        case VariantType::BOOL:
            lua_pushboolean(L, v.as_bool());
            break;
        case VariantType::INT:
            lua_pushinteger(L, v.as_int());
            break;
        case VariantType::FLOAT:
            lua_pushnumber(L, v.as_float());
            break;
        case VariantType::STRING:
            lua_pushstring(L, v.as_string().c_str());
            break;
        default:
            lua_pushnil(L);
            break;
    }
}

Variant LuaBridge::to_variant(lua_State* L, int index) {
    switch (lua_type(L, index)) {
        case LUA_TNIL:
            return Variant::NIL;
        case LUA_TBOOLEAN:
            return Variant(lua_toboolean(L, index) != 0);
        case LUA_TNUMBER:
            if (lua_isinteger(L, index)) {
                return Variant((int32_t)lua_tointeger(L, index));
            }
            return Variant((float)lua_tonumber(L, index));
        case LUA_TSTRING:
            return Variant(lua_tostring(L, index));
        default:
            return Variant::NIL;
    }
}

int LuaBridge::lua_error_handler(lua_State* L) {
    const char* msg = lua_tostring(L, 1);
    if (msg == nullptr) {
        if (luaL_callmeta(L, 1, "__tostring")) {
            return 1;
        }
        msg = "(error object is not a string)";
    }
    luaL_traceback(L, L, msg, 1);
    return 1;
}

} // namespace MyEngine
