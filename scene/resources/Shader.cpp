#include "Shader.h"
#include "renderer/RenderDevice.h"

namespace MyEngine {

Shader* Shader::_current_shader = nullptr;

Shader::Shader() = default;

Shader::~Shader() {
    if (_program != 0) {
        // RenderDevice::get()->shader_free(_program);
    }
}

bool Shader::compile() {
    // 简化实现：标记为已编译
    // 实际编译逻辑在 RenderDevice 中
    _compiled = true;

    // 解析 uniform 定义
    _reflect_uniforms();

    return _compiled;
}

void Shader::_reflect_uniforms() {
    // 简化实现：从源码中解析 uniform 定义
    // 格式: uniform float name = value;
    //       uniform vec3 name;
    //       uniform sampler2D name;

    auto parse_uniforms = [](const std::string& source) {
        std::unordered_map<std::string, UniformDef> uniforms;

        size_t pos = 0;
        while ((pos = source.find("uniform", pos)) != std::string::npos) {
            size_t semicolon = source.find(";", pos);
            if (semicolon == std::string::npos) break;

            std::string decl = source.substr(pos, semicolon - pos + 1);

            // 解析类型和名称
            size_t type_start = decl.find("uniform ") + 8;
            size_t name_start = decl.find_first_of(" \t\n", type_start);
            std::string type_str = decl.substr(type_start, name_start - type_start);

            size_t name_end = decl.find_first_of(" \t\n[]=", name_start);
            std::string name = decl.substr(name_start, name_end - name_start);

            UniformType utype = UniformType::NONE;
            if (type_str == "float") utype = UniformType::FLOAT;
            else if (type_str == "vec2") utype = UniformType::VEC2;
            else if (type_str == "vec3") utype = UniformType::VEC3;
            else if (type_str == "vec4") utype = UniformType::VEC4;
            else if (type_str == "int" || type_str == "bool") utype = UniformType::INT;
            else if (type_str == "mat4") utype = UniformType::MAT4;
            else if (type_str == "sampler2D") utype = UniformType::SAMPLER2D;
            else if (type_str == "samplerCube") utype = UniformType::SAMPLER_CUBE;

            if (utype != UniformType::NONE && !name.empty()) {
                UniformDef def;
                def.name = name;
                def.type = utype;
                uniforms[name] = def;
            }

            pos = semicolon;
        }

        return uniforms;
    };

    _uniforms = parse_uniforms(_vertex_source);
    auto frag_uniforms = parse_uniforms(_fragment_source);
    _uniforms.insert(frag_uniforms.begin(), frag_uniforms.end());
}

int Shader::get_uniform_location(const std::string& name) {
    auto it = _uniforms.find(name);
    if (it != _uniforms.end()) {
        return it->second.location;
    }
    return -1;
}

void Shader::set_uniform(int location, float value) {
    (void)location; (void)value;
}

void Shader::set_uniform(int location, const Vector2& value) {
    (void)location; (void)value;
}

void Shader::set_uniform(int location, const Vector3& value) {
    (void)location; (void)value;
}

void Shader::set_uniform(int location, const Vector4& value) {
    (void)location; (void)value;
}

void Shader::set_uniform(int location, int value) {
    (void)location; (void)value;
}

void Shader::set_uniform(int location, bool value) {
    (void)location; (void)value;
}

void Shader::set_uniform(int location, const Matrix4& value) {
    (void)location; (void)value;
}

void Shader::set_uniform_array(int location, int count, const float* values) {
    (void)location; (void)count; (void)values;
}

void Shader::set_uniform_matrix(int location, const Matrix4& value) {
    set_uniform(location, value);
}

std::vector<ShaderParam> Shader::get_params() const {
    std::vector<ShaderParam> params;
    for (const auto& [name, def] : _uniforms) {
        params.push_back({name, def.type});
    }
    return params;
}

void Shader::_bind() {
    if (_current_shader != this) {
        _current_shader = this;
        // RenderDevice::get()->shader_bind(_program);
    }
}

void Shader::_unbind() {
    if (_current_shader != nullptr) {
        // RenderDevice::get()->shader_bind(0);
        _current_shader = nullptr;
    }
}

} // namespace MyEngine
