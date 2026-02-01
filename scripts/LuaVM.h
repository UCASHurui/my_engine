#pragma once

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

namespace MyEngine {

// Lua 虚拟机
class LuaVM {
public:
    LuaVM();
    ~LuaVM();

    // 初始化/销毁
    bool initialize();
    void shutdown();

    // 执行脚本
    bool do_file(const String& path);
    bool do_string(const String& code);

    // 加载脚本
    bool load_file(const String& path);
    bool load_string(const String& code);

    // 调用函数
    bool call(int nargs = 0, int nresults = 0);

    // 注册 C++ 函数
    void register_function(const String& name, int (*func)(lua_State* L));
    void register_function(const String& name, std::function<int(lua_State*)> func);

    // 注册常量
    void register_constant(const String& name, int value);
    void register_constant(const String& name, float value);
    void register_constant(const String& name, const char* value);

    // 表操作
    void new_table(const String& name);
    void set_table_value(const String& table, const String& key, const Variant& value);

    // 全局变量
    void set_global(const String& name, const Variant& value);
    Variant get_global(const String& name);

    // 错误处理
    const char* get_last_error() const { return _last_error.c_str(); }

    // 状态
    lua_State* get_state() { return _L; }
    bool is_valid() const { return _L != nullptr; }

    // 垃圾回收
    int gc(int what, int data = 0);

private:
    lua_State* _L = nullptr;
    String _last_error;
};

} // namespace MyEngine
