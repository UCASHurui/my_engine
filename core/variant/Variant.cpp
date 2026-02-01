#include "Variant.h"
#include "core/containers/String.h"
#include <string>

namespace MyEngine {

const Variant Variant::NIL;

const char* Variant::get_type_name() const {
    static const char* names[] = {
        "NIL", "BOOL", "INT", "FLOAT", "STRING", "VECTOR2", "VECTOR3",
        "VECTOR4", "MATRIX4", "TRANSFORM2D", "TRANSFORM3D", "COLOR",
        "RECT2", "QUAT", "PLANE", "AABB", "POOL_BYTE_ARRAY", "POOL_INT_ARRAY",
        "POOL_REAL_ARRAY", "POOL_STRING_ARRAY", "POOL_VECTOR2_ARRAY",
        "POOL_VECTOR3_ARRAY", "POOL_COLOR_ARRAY", "OBJECT", "CALLABLE",
        "SIGNAL", "DICTIONARY", "ARRAY"
    };
    return names[(int)_type];
}

String Variant::as_string() const {
    switch (_type) {
        case VariantType::NIL:
            return "";
        case VariantType::BOOL:
            return _data.b ? "true" : "false";
        case VariantType::INT:
            return String::format("%d", _data.i);
        case VariantType::FLOAT:
            return String::format("%f", _data.f);
        case VariantType::STRING:
            return *_data.str;
        default:
            return "";
    }
}

void Variant::_copy(const Variant& other) {
    _type = other._type;
    switch (_type) {
        case VariantType::STRING:
            _data.str = new String(*other._data.str);
            break;
        default:
            _data = other._data;
            break;
    }
}

void Variant::_move(Variant& other) {
    _type = other._type;
    _data = other._data;
    other._type = VariantType::NIL;
    other._data.b = false;
}

void Variant::_destroy() {
    if (_type == VariantType::STRING && _data.str) {
        delete _data.str;
        _data.str = nullptr;
    }
}

bool Variant::_compare(const Variant& other) const {
    if (_type != other._type) return false;

    switch (_type) {
        case VariantType::NIL:
            return true;
        case VariantType::BOOL:
            return _data.b == other._data.b;
        case VariantType::INT:
            return _data.i == other._data.i;
        case VariantType::FLOAT:
            return _data.f == other._data.f;
        case VariantType::STRING:
            return *_data.str == *other._data.str;
        default:
            return false;
    }
}

bool Variant::operator<(const Variant& other) const {
    if (_type == VariantType::INT && other._type == VariantType::INT) {
        return _data.i < other._data.i;
    }
    if (_type == VariantType::FLOAT && other._type == VariantType::FLOAT) {
        return _data.f < other._data.f;
    }
    if (_type == VariantType::STRING && other._type == VariantType::STRING) {
        return *_data.str < *other._data.str;
    }
    return false;
}

String Variant::serialize() const {
    switch (_type) {
        case VariantType::NIL:
            return "nil";
        case VariantType::BOOL:
            return _data.b ? "true" : "false";
        case VariantType::INT:
            return String::format("%d", _data.i);
        case VariantType::FLOAT:
            return String::format("%f", _data.f);
        case VariantType::STRING:
            return String("\"") + *_data.str + "\"";
        default:
            return "\"\""; // 简化处理
    }
}

Variant Variant::deserialize(const String& str) {
    if (str == "nil") return Variant::NIL;
    if (str == "true") return Variant(true);
    if (str == "false") return Variant(false);

    // 检查是否为字符串
    if (str.length() >= 2 && str[0] == '"' && str[str.length() - 1] == '"') {
        return Variant(str.substr(1, str.length() - 2));
    }

    // 检查是否为数字
    if (str.find(".") != String::npos) {
        return Variant(std::stof(str.c_str()));
    }
    return Variant(std::stoi(str.c_str()));
}

} // namespace MyEngine
