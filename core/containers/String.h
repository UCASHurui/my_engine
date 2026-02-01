#pragma once

#include <cstddef>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstdarg>

namespace MyEngine {

// 简单字符串类
class String {
public:
    String() : _data(nullptr), _length(0) {
        _data = new char[1];
        _data[0] = '\0';
    }

    String(const char* str) {
        _length = strlen(str);
        _data = new char[_length + 1];
        memcpy(_data, str, _length + 1);
    }

    String(const String& other) {
        _length = other._length;
        _data = new char[_length + 1];
        memcpy(_data, other._data, _length + 1);
    }

    String(String&& other) noexcept : _data(other._data), _length(other._length) {
        other._data = new char[1];
        other._data[0] = '\0';
        other._length = 0;
    }

    String(size_t count, char ch) {
        _length = count;
        _data = new char[_length + 1];
        memset(_data, ch, count);
        _data[count] = '\0';
    }

    ~String() {
        delete[] _data;
    }

    // 赋值
    String& operator=(const char* str) {
        delete[] _data;
        _length = strlen(str);
        _data = new char[_length + 1];
        memcpy(_data, str, _length + 1);
        return *this;
    }

    String& operator=(const String& other) {
        if (this != &other) {
            delete[] _data;
            _length = other._length;
            _data = new char[_length + 1];
            memcpy(_data, other._data, _length + 1);
        }
        return *this;
    }

    String& operator=(String&& other) noexcept {
        if (this != &other) {
            delete[] _data;
            _data = other._data;
            _length = other._length;
            other._data = new char[1];
            other._data[0] = '\0';
            other._length = 0;
        }
        return *this;
    }

    // 访问
    char& operator[](size_t index) { return _data[index]; }
    const char& operator[](size_t index) const { return _data[index]; }

    const char* c_str() const { return _data ? _data : ""; }
    const char* data() const { return _data ? _data : ""; }

    size_t length() const { return _length; }
    bool empty() const { return _length == 0; }

    // 修改
    void clear() {
        delete[] _data;
        _data = new char[1];
        _data[0] = '\0';
        _length = 0;
    }

    void resize(size_t new_size, char fill = '\0') {
        if (new_size == _length) return;
        char* new_data = new char[new_size + 1];
        size_t copy_len = (_length < new_size) ? _length : new_size;
        if (_data) memcpy(new_data, _data, copy_len);
        if (new_size > _length) {
            memset(new_data + _length, fill, new_size - _length);
        }
        new_data[new_size] = '\0';
        delete[] _data;
        _data = new_data;
        _length = new_size;
    }

    // 拼接
    String operator+(const String& other) const {
        String result;
        result._length = _length + other._length;
        result._data = new char[result._length + 1];
        memcpy(result._data, _data, _length);
        memcpy(result._data + _length, other._data, other._length + 1);
        return result;
    }

    String operator+(const char* other) const {
        return *this + String(other);
    }

    String& operator+=(const String& other) {
        return *this = *this + other;
    }

    String& operator+=(char c) {
        char tmp[2] = {c, '\0'};
        return *this = *this + tmp;
    }

    // 比较
    bool operator==(const String& other) const {
        return _length == other._length && memcmp(_data, other._data, _length) == 0;
    }
    bool operator!=(const String& other) const { return !(*this == other); }
    bool operator<(const String& other) const { return strcmp(_data, other._data) < 0; }
    bool operator<=(const String& other) const { return strcmp(_data, other._data) <= 0; }
    bool operator>(const String& other) const { return strcmp(_data, other._data) > 0; }
    bool operator>=(const String& other) const { return strcmp(_data, other._data) >= 0; }

    // 子串
    String substr(size_t start, size_t len = (size_t)-1) const {
        if (start > _length) return String();
        size_t actual_len = (len == (size_t)-1 || start + len > _length) ? _length - start : len;
        String result;
        result._length = actual_len;
        result._data = new char[actual_len + 1];
        memcpy(result._data, _data + start, actual_len);
        result._data[actual_len] = '\0';
        return result;
    }

    size_t find(const String& str, size_t start = 0) const {
        if (str._length == 0) return start;
        for (size_t i = start; i <= _length - str._length; i++) {
            if (memcmp(_data + i, str._data, str._length) == 0) {
                return i;
            }
        }
        return npos;
    }

    size_t find_last_of(const char* chars, size_t start = 0) const {
        if (_length == 0 || start >= _length) return npos;
        for (size_t i = _length - 1; i >= start; i--) {
            for (const char* p = chars; *p; p++) {
                if (_data[i] == *p) return i;
            }
        }
        return npos;
    }

    size_t find_first_of(const char* chars, size_t start = 0) const {
        for (size_t i = start; i < _length; i++) {
            for (const char* p = chars; *p; p++) {
                if (_data[i] == *p) return i;
            }
        }
        return npos;
    }

    bool starts_with(const String& prefix) const {
        if (prefix._length > _length) return false;
        return memcmp(_data, prefix._data, prefix._length) == 0;
    }

    bool ends_with(const String& suffix) const {
        if (suffix._length > _length) return false;
        return memcmp(_data + _length - suffix._length, suffix._data, suffix._length) == 0;
    }

    // 大小写
    String to_lower() const {
        String result = *this;
        for (size_t i = 0; i < result._length; i++) {
            result._data[i] = (char)tolower(result._data[i]);
        }
        return result;
    }

    String to_upper() const {
        String result = *this;
        for (size_t i = 0; i < result._length; i++) {
            result._data[i] = (char)toupper(result._data[i]);
        }
        return result;
    }

    // 转换
    int to_int() const {
        return (int)strtol(_data, nullptr, 10);
    }
    float to_float() const {
        return strtof(_data, nullptr);
    }

    // 格式化
    static String format(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        va_list args_copy;
        va_copy(args_copy, args);
        int len = vsnprintf(nullptr, 0, fmt, args_copy);
        va_end(args_copy);

        if (len <= 0) return String();

        String result;
        result._length = len;
        result._data = new char[len + 1];
        vsnprintf(result._data, len + 1, fmt, args);
        va_end(args);
        return result;
    }

    // 迭代器
    const char* begin() const { return _data; }
    const char* end() const { return _data + _length; }

    static constexpr size_t npos = (size_t)-1;

private:
    char* _data;
    size_t _length;
};

// 便利函数
inline String operator+(const char* a, const String& b) {
    return String(a) + b;
}

// 简单的字符串哈希函数
inline size_t hash_string(const String& s) {
    const char* p = s.c_str();
    size_t h = 0;
    while (*p) {
        h = h * 31 + *p++;
    }
    return h;
}

} // namespace MyEngine
