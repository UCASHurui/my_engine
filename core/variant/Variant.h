#pragma once

#include "containers/String.h"
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace MyEngine {

// 变体类型枚举
enum class VariantType : uint8_t {
    NIL = 0,
    BOOL,
    INT,
    FLOAT,
    STRING,
    VECTOR2,
    VECTOR3,
    VECTOR4,
    MATRIX4,
    TRANSFORM2D,
    TRANSFORM3D,
    COLOR,
    RECT2,
    QUAT,
    PLANE,
    AABB,
    POOL_BYTE_ARRAY,
    POOL_INT_ARRAY,
    POOL_REAL_ARRAY,
    POOL_STRING_ARRAY,
    POOL_VECTOR2_ARRAY,
    POOL_VECTOR3_ARRAY,
    POOL_COLOR_ARRAY,
    OBJECT,
    CALLABLE,
    SIGNAL,
    DICTIONARY,
    ARRAY,
    MAX
};

// Variant 联合体 - 支持多种类型的变体值
class Variant {
public:
    static const Variant NIL;

    // 构造函数
    Variant() : _type(VariantType::NIL) {}
    Variant(std::nullptr_t) : _type(VariantType::NIL) {}

    Variant(bool v) : _type(VariantType::BOOL) { _data.b = v; }
    Variant(int32_t v) : _type(VariantType::INT) { _data.i = v; }
    Variant(int64_t v) : _type(VariantType::INT) { _data.i = (int32_t)v; }
    Variant(float v) : _type(VariantType::FLOAT) { _data.f = v; }
    Variant(double v) : _type(VariantType::FLOAT) { _data.f = (float)v; }
    Variant(const String& v) : _type(VariantType::STRING) { _data.str = new String(v); }
    Variant(const char* v) : _type(VariantType::STRING) { _data.str = new String(v); }

    // 复制构造
    Variant(const Variant& other) { _copy(other); }

    // 移动构造
    Variant(Variant&& other) noexcept { _move(other); }

    // 析构
    ~Variant() { _destroy(); }

    // 赋值操作符
    Variant& operator=(const Variant& other) {
        if (this != &other) {
            _destroy();
            _copy(other);
        }
        return *this;
    }

    Variant& operator=(Variant&& other) noexcept {
        if (this != &other) {
            _destroy();
            _move(other);
        }
        return *this;
    }

    Variant& operator=(std::nullptr_t) { _destroy(); _type = VariantType::NIL; return *this; }
    Variant& operator=(bool v) { _destroy(); _type = VariantType::BOOL; _data.b = v; return *this; }
    Variant& operator=(int32_t v) { _destroy(); _type = VariantType::INT; _data.i = v; return *this; }
    Variant& operator=(float v) { _destroy(); _type = VariantType::FLOAT; _data.f = v; return *this; }
    Variant& operator=(const String& v) { _destroy(); _type = VariantType::STRING; _data.str = new String(v); return *this; }
    Variant& operator=(const char* v) { _destroy(); _type = VariantType::STRING; _data.str = new String(v); return *this; }

    // 类型查询
    VariantType get_type() const { return _type; }
    const char* get_type_name() const;

    bool is_nil() const { return _type == VariantType::NIL; }
    bool is_numeric() const { return _type == VariantType::INT || _type == VariantType::FLOAT; }
    bool is_string() const { return _type == VariantType::STRING; }
    bool is_array() const { return _type == VariantType::ARRAY; }
    bool is_dictionary() const { return _type == VariantType::DICTIONARY; }

    // 值获取
    bool as_bool() const { return _get_as<bool>(VariantType::BOOL, _data.b); }
    int32_t as_int() const { return _get_as<int32_t>(VariantType::INT, _data.i); }
    float as_float() const { return _get_as<float>(VariantType::FLOAT, _data.f); }
    String as_string() const;

    // 模板化值获取
    template<typename T>
    T get_value() const;

    // 比较操作符
    bool operator==(const Variant& other) const { return _compare(other); }
    bool operator!=(const Variant& other) const { return !_compare(other); }
    bool operator<(const Variant& other) const;

    // 序列化
    String serialize() const;
    static Variant deserialize(const String& str);

private:
    union VariantData {
        bool b;
        int32_t i;
        float f;
        String* str;
        void* ptr;
    };

    VariantType _type = VariantType::NIL;
    VariantData _data{};

    void _copy(const Variant& other);
    void _move(Variant& other);
    void _destroy();

    template<typename T>
    T _get_as(VariantType expected, T default_value) const {
        if (_type == expected) return _data.b;
        return default_value;
    }

    bool _compare(const Variant& other) const;
};

// 模板特化
template<>
inline bool Variant::get_value<bool>() const { return as_bool(); }

template<>
inline int32_t Variant::get_value<int32_t>() const { return as_int(); }

template<>
inline float Variant::get_value<float>() const { return as_float(); }

template<>
inline String Variant::get_value<String>() const { return as_string(); }

} // namespace MyEngine
