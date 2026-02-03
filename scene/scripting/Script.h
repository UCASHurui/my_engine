#pragma once

#include "object/RefCounted.h"
#include "variant/Variant.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

namespace MyEngine {

// 脚本实例
class ScriptInstance : public RefCounted {
public:
    ScriptInstance();
    ~ScriptInstance() override;

    // 执行脚本方法
    bool call_method(const std::string& method, const std::vector<Variant>& args, Variant& result);

    // 获取属性
    bool get_property(const std::string& name, Variant& value);
    bool set_property(const std::string& name, const Variant& value);

    // 获取类型信息
    const std::string& get_script_class_name() const { return _class_name; }

protected:
    std::string _class_name;
    std::unordered_map<std::string, Variant> _properties;
    std::unordered_map<std::string, std::function<Variant(const std::vector<Variant>&)>> _methods;

    void _register_method(const std::string& name,
        std::function<Variant(const std::vector<Variant>&)> func);
    void _set_property(const std::string& name, const Variant& value);
    Variant _get_property(const std::string& name);

    friend class Script;
};

// 脚本基类
class Script : public RefCounted {
public:
    Script();
    ~Script() override;

    const char* get_class_name() const override { return "Script"; }

    // 脚本名称
    void set_name(const std::string& name) { _name = name; }
    std::string get_name() const { return _name; }

    // 继承的类名
    void set_inherits(const std::string& class_name) { _inherits = class_name; }
    std::string get_inherits() const { return _inherits; }

    // 脚本路径
    void set_path(const std::string& path) { _path = path; }
    std::string get_path() const { return _path; }

    // 脚本源
    void set_source(const std::string& source) { _source = source; }
    std::string get_source() const { return _source; }

    // 实例化
    Ref<ScriptInstance> instantiate();

    // 脚本状态
    bool is_valid() const { return _valid; }
    std::string get_error() const { return _error; }

    // 属性定义
    struct PropertyDef {
        std::string name;
        VariantType type = VariantType::NIL;
        Variant default_value;
        int flags = 0;
    };

    void add_property(const PropertyDef& prop);
    const std::vector<PropertyDef>& get_properties() const { return _properties; }

    // 方法定义
    struct MethodDef {
        std::string name;
        std::vector<VariantType> arguments;
        VariantType return_type = VariantType::NIL;
    };

    void add_method(const MethodDef& method);
    const std::vector<MethodDef>& get_methods() const { return _methods; }

    // 编译
    bool compile();
    bool reload();

protected:
    std::string _name;
    std::string _inherits;
    std::string _path;
    std::string _source;
    bool _valid = false;
    std::string _error;

    std::vector<PropertyDef> _properties;
    std::vector<MethodDef> _methods;

    virtual bool _compile_source(const std::string& source);
    virtual Ref<ScriptInstance> _create_instance();

    friend class ScriptLanguage;
};

// 脚本语言
class ScriptLanguage {
public:
    virtual ~ScriptLanguage() = default;

    virtual const char* get_name() const = 0;
    virtual const char* get_extension() const = 0;

    // 初始化/清理
    virtual void initialize() = 0;
    virtual void finalize() = 0;

    // 创建脚本
    virtual Ref<Script> create_script() = 0;

    // 执行代码
    virtual bool execute(const std::string& code, Variant& result) = 0;

    // 调试
    virtual void set_debugger_enabled(bool enabled) = 0;
    virtual bool is_debugger_enabled() const = 0;
};

// 脚本服务器
class ScriptServer : public RefCounted {
public:
    ScriptServer();
    ~ScriptServer() override;

    const char* get_class_name() const override { return "ScriptServer"; }

    static ScriptServer* get() { return _singleton; }

    // 注册语言
    void register_language(ScriptLanguage* language);
    void unregister_language(const std::string& name);

    // 获取语言
    ScriptLanguage* get_language(const std::string& name) const;
    std::vector<std::string> get_language_names() const;

    // 默认语言
    void set_default_language(const std::string& name);
    ScriptLanguage* get_default_language() const;

    // 加载脚本
    Ref<Script> load_script(const std::string& path);

    // 全局变量
    void set_global_variable(const std::string& name, const Variant& value);
    Variant get_global_variable(const std::string& name) const;
    void clear_global_variable(const std::string& name);

private:
    static ScriptServer* _singleton;

    std::unordered_map<std::string, ScriptLanguage*> _languages;
    ScriptLanguage* _default_language = nullptr;
    std::unordered_map<std::string, Variant> _global_variables;
};

} // namespace MyEngine
