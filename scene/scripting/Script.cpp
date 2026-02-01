#include "Script.h"

namespace MyEngine {

ScriptInstance::ScriptInstance() = default;
ScriptInstance::~ScriptInstance() = default;

bool ScriptInstance::call_method(const std::string& method, const std::vector<Variant>& args, Variant& result) {
    auto it = _methods.find(method);
    if (it != _methods.end()) {
        result = it->second(args);
        return true;
    }
    return false;
}

bool ScriptInstance::get_property(const std::string& name, Variant& value) {
    auto it = _properties.find(name);
    if (it != _properties.end()) {
        value = it->second;
        return true;
    }
    return false;
}

bool ScriptInstance::set_property(const std::string& name, const Variant& value) {
    _properties[name] = value;
    return true;
}

void ScriptInstance::_register_method(const std::string& name,
    std::function<Variant(const std::vector<Variant>&)> func) {
    _methods[name] = std::move(func);
}

void ScriptInstance::_set_property(const std::string& name, const Variant& value) {
    _properties[name] = value;
}

Variant ScriptInstance::_get_property(const std::string& name) {
    auto it = _properties.find(name);
    return it != _properties.end() ? it->second : Variant();
}

Script::Script() = default;
Script::~Script() = default;

void Script::add_property(const PropertyDef& prop) {
    _properties.push_back(prop);
}

void Script::add_method(const MethodDef& method) {
    _methods.push_back(method);
}

bool Script::compile() {
    return _compile_source(_source);
}

bool Script::reload() {
    _valid = false;
    return compile();
}

bool Script::_compile_source(const std::string& source) {
    (void)source;
    _valid = true;
    return true;
}

Ref<ScriptInstance> Script::instantiate() {
    Ref<ScriptInstance> instance = _create_instance();
    if (instance) {
        instance->_class_name = _name;
    }
    return instance;
}

Ref<ScriptInstance> Script::_create_instance() {
    return new ScriptInstance();
}

ScriptServer* ScriptServer::_singleton = nullptr;

ScriptServer::ScriptServer() {
    _singleton = this;
}

ScriptServer::~ScriptServer() {
    if (_singleton == this) {
        _singleton = nullptr;
    }
}

void ScriptServer::register_language(ScriptLanguage* language) {
    _languages[language->get_name()] = language;
}

void ScriptServer::unregister_language(const std::string& name) {
    _languages.erase(name);
}

ScriptLanguage* ScriptServer::get_language(const std::string& name) const {
    auto it = _languages.find(name);
    return it != _languages.end() ? it->second : nullptr;
}

std::vector<std::string> ScriptServer::get_language_names() const {
    std::vector<std::string> names;
    for (const auto& [name, lang] : _languages) {
        names.push_back(name);
    }
    return names;
}

void ScriptServer::set_default_language(const std::string& name) {
    _default_language = get_language(name);
}

ScriptLanguage* ScriptServer::get_default_language() const {
    return _default_language;
}

Ref<Script> ScriptServer::load_script(const std::string& path) {
    if (!_default_language) return nullptr;

    Ref<Script> script = _default_language->create_script();
    if (script) {
        script->set_path(path);
        // 加载文件内容...
        if (!script->compile()) {
            return nullptr;
        }
    }
    return script;
}

void ScriptServer::set_global_variable(const std::string& name, const Variant& value) {
    _global_variables[name] = value;
}

Variant ScriptServer::get_global_variable(const std::string& name) const {
    auto it = _global_variables.find(name);
    return it != _global_variables.end() ? it->second : Variant();
}

void ScriptServer::clear_global_variable(const std::string& name) {
    _global_variables.erase(name);
}

} // namespace MyEngine
