#include "LuaVM.h"
#include "core/variant/Variant.h"
#include "core/os/OS.h"
#include <cstdio>

namespace MyEngine {

LuaVM::LuaVM() {}

LuaVM::~LuaVM() {
    shutdown();
}

bool LuaVM::initialize() {
    _L = luaL_newstate();
    if (!_L) {
        _last_error = "Failed to create Lua state";
        return false;
    }

    // 打开标准库
    luaL_openlibs(_L);
    return true;
}

void LuaVM::shutdown() {
    if (_L) {
        lua_close(_L);
        _L = nullptr;
    }
}

bool LuaVM::do_file(const String& path) {
    if (!_L) return false;

    int status = luaL_dofile(_L, path.c_str());
    if (status != LUA_OK) {
        _last_error = lua_tostring(_L, -1);
        lua_pop(_L, 1);
        return false;
    }
    return true;
}

bool LuaVM::do_string(const String& code) {
    if (!_L) return false;

    int status = luaL_dostring(_L, code.c_str());
    if (status != LUA_OK) {
        _last_error = lua_tostring(_L, -1);
        lua_pop(_L, 1);
        return false;
    }
    return true;
}

bool LuaVM::load_file(const String& path) {
    if (!_L) return false;

    int status = luaL_loadfile(_L, path.c_str());
    if (status != LUA_OK) {
        _last_error = lua_tostring(_L, -1);
        lua_pop(_L, 1);
        return false;
    }
    return true;
}

bool LuaVM::load_string(const String& code) {
    if (!_L) return false;

    int status = luaL_loadstring(_L, code.c_str());
    if (status != LUA_OK) {
        _last_error = lua_tostring(_L, -1);
        lua_pop(_L, 1);
        return false;
    }
    return true;
}

bool LuaVM::call(int nargs, int nresults) {
    if (!_L) return false;

    int status = lua_pcall(_L, nargs, nresults, 0);
    if (status != LUA_OK) {
        _last_error = lua_tostring(_L, -1);
        lua_pop(_L, 1);
        return false;
    }
    return true;
}

void LuaVM::register_function(const String& name, int (*func)(lua_State* L)) {
    if (_L) {
        lua_register(_L, name.c_str(), func);
    }
}

void LuaVM::register_function(const String& name, std::function<int(lua_State*)> func) {
    if (_L) {
        lua_pushlightuserdata(_L, new std::function<int(lua_State*)>(func));
        lua_pushcclosure(_L, [](lua_State* L) -> int {
            auto* func = reinterpret_cast<std::function<int(lua_State*)>*>(lua_touserdata(L, lua_upvalueindex(1)));
            return (*func)(L);
        }, 1);
        lua_setglobal(_L, name.c_str());
    }
}

void LuaVM::register_constant(const String& name, int value) {
    if (_L) {
        lua_pushinteger(_L, value);
        lua_setglobal(_L, name.c_str());
    }
}

void LuaVM::register_constant(const String& name, float value) {
    if (_L) {
        lua_pushnumber(_L, value);
        lua_setglobal(_L, name.c_str());
    }
}

void LuaVM::register_constant(const String& name, const char* value) {
    if (_L) {
        lua_pushstring(_L, value);
        lua_setglobal(_L, name.c_str());
    }
}

void LuaVM::new_table(const String& name) {
    if (_L) {
        lua_newtable(_L);
        lua_setglobal(_L, name.c_str());
    }
}

void LuaVM::set_table_value(const String& table, const String& key, const Variant& value) {
    if (!_L) return;

    lua_getglobal(_L, table.c_str());
    if (lua_istable(_L, -1)) {
        LuaBridge::push_variant(_L, value);
        lua_setfield(_L, -2, key.c_str());
    }
    lua_pop(_L, 1);
}

void LuaVM::set_global(const String& name, const Variant& value) {
    if (_L) {
        LuaBridge::push_variant(_L, value);
        lua_setglobal(_L, name.c_str());
    }
}

Variant LuaVM::get_global(const String& name) {
    if (!_L) return Variant::NIL;

    lua_getglobal(_L, name.c_str());
    Variant result = LuaBridge::to_variant(_L, -1);
    lua_pop(_L, 1);
    return result;
}

int LuaVM::gc(int what, int data) {
    if (_L) {
        return lua_gc(_L, what, data);
    }
    return 0;
}

} // namespace MyEngine
