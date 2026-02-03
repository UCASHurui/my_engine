#pragma once

#include "object/Resource.h"
#include "math/Color.h"
#include "math/Vector2.h"
#include "math/Vector3.h"
#include "math/Matrix4.h"
#include <string>
#include <unordered_map>
#include <variant>

namespace MyEngine {

// 着色器类型
enum class ShaderType {
    VERTEX,
    FRAGMENT,
    GEOMETRY,
    COMPUTE
};

// Uniform 类型
enum class UniformType {
    NONE,
    FLOAT,
    VEC2,
    VEC3,
    VEC4,
    INT,
    IVEC2,
    IVEC3,
    IVEC4,
    BOOL,
    BVEC2,
    BVEC3,
    BVEC4,
    MAT2,
    MAT3,
    MAT4,
    SAMPLER2D,
    SAMPLER_CUBE,
    ARRAY,
    STRUCT
};

// Uniform 定义
struct UniformDef {
    std::string name;
    UniformType type;
    int array_size = 0;
    int location = -1;
    std::variant<float, Vector2, Vector3, Vector4, int, bool, Matrix4> default_value;
};

// 着色器参数
struct ShaderParam {
    std::string name;
    UniformType type;
    int count = 1;
};

// 着色器
class Shader : public Resource {
public:
    Shader();
    ~Shader() override;

    const char* get_class_name() const override { return "Shader"; }

    // 源码设置
    void set_vertex_source(const std::string& source) { _vertex_source = source; }
    void set_fragment_source(const std::string& source) { _fragment_source = source; }
    void set_geometry_source(const std::string& source) { _geometry_source = source; }
    void set_compute_source(const std::string& source) { _compute_source = source; }

    std::string get_vertex_source() const { return _vertex_source; }
    std::string get_fragment_source() const { return _fragment_source; }
    std::string get_geometry_source() const { return _geometry_source; }
    std::string get_compute_source() const { return _compute_source; }

    // 编译
    bool compile();
    bool is_compiled() const { return _compiled; }
    std::string get_compile_error() const { return _compile_error; }

    // Uniform 管理
    int get_uniform_location(const std::string& name);
    void set_uniform(int location, float value);
    void set_uniform(int location, const Vector2& value);
    void set_uniform(int location, const Vector3& value);
    void set_uniform(int location, const Vector4& value);
    void set_uniform(int location, int value);
    void set_uniform(int location, bool value);
    void set_uniform(int location, const Matrix4& value);
    void set_uniform_array(int location, int count, const float* values);
    void set_uniform_matrix(int location, const Matrix4& value);

    // 参数查询
    std::vector<ShaderParam> get_params() const;

    // 内置 Uniform
    static constexpr const char* MODEL_MATRIX = "model";
    static constexpr const char* VIEW_MATRIX = "view";
    static constexpr const char* PROJECTION_MATRIX = "projection";
    static constexpr const char* MODEL_VIEW_MATRIX = "model_view";
    static constexpr const char* VIEW_PROJECTION_MATRIX = "view_projection";
    static constexpr const char* MODEL_VIEW_PROJECTION_MATRIX = "model_view_projection";
    static constexpr const char* NORMAL_MATRIX = "normal_matrix";
    static constexpr const char* CAMERA_POSITION = "camera_position";
    static constexpr const char* TIME = "time";
    static constexpr const char* SCREEN_SIZE = "screen_size";
    static constexpr const char* ALBEDO = "albedo";
    static constexpr const char* METALLIC = "metallic";
    static constexpr const char* ROUGHNESS = "roughness";
    static constexpr const char* AO = "ao";

protected:
    bool _compile_source(ShaderType type, const std::string& source);
    void _reflect_uniforms();

private:
    std::string _vertex_source;
    std::string _fragment_source;
    std::string _geometry_source;
    std::string _compute_source;

    bool _compiled = false;
    std::string _compile_error;

    // 平台特定的着色器句柄
    uint32_t _vertex_handle = 0;
    uint32_t _fragment_handle = 0;
    uint32_t _geometry_handle = 0;
    uint32_t _compute_handle = 0;
    uint32_t _program = 0;

    std::unordered_map<std::string, UniformDef> _uniforms;

    static Shader* _current_shader;
    static Shader* get_current() { return _current_shader; }
    void _bind();
    void _unbind();

    friend class RenderDevice;
};

} // namespace MyEngine
