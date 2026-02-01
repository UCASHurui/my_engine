#pragma once

#include <cstdlib>
#include <new>
#include <algorithm>

namespace MyEngine {

// 动态数组
template<typename T>
class Vector {
public:
    Vector() : _data(nullptr), _size(0), _capacity(0) {}

    explicit Vector(size_t size) : _size(size), _capacity(size) {
        _data = _allocate(_capacity);
        for (size_t i = 0; i < _size; i++) {
            new (&_data[i]) T();
        }
    }

    Vector(size_t size, const T& value) : _size(size), _capacity(size) {
        _data = _allocate(_capacity);
        for (size_t i = 0; i < _size; i++) {
            new (&_data[i]) T(value);
        }
    }

    Vector(std::initializer_list<T> list) : _size(list.size()), _capacity(list.size()) {
        _data = _allocate(_capacity);
        size_t i = 0;
        for (const auto& item : list) {
            new (&_data[i++]) T(item);
        }
    }

    ~Vector() {
        _destroy();
    }

    // 复制构造
    Vector(const Vector& other) : _size(other._size), _capacity(other._size) {
        _data = _allocate(_capacity);
        for (size_t i = 0; i < _size; i++) {
            new (&_data[i]) T(other._data[i]);
        }
    }

    // 移动构造
    Vector(Vector&& other) noexcept : _data(other._data), _size(other._size), _capacity(other._capacity) {
        other._data = nullptr;
        other._size = 0;
        other._capacity = 0;
    }

    // 赋值
    Vector& operator=(const Vector& other) {
        if (this != &other) {
            _destroy();
            _size = other._size;
            _capacity = other._size;
            _data = _allocate(_capacity);
            for (size_t i = 0; i < _size; i++) {
                new (&_data[i]) T(other._data[i]);
            }
        }
        return *this;
    }

    Vector& operator=(Vector&& other) noexcept {
        if (this != &other) {
            _destroy();
            _data = other._data;
            _size = other._size;
            _capacity = other._capacity;
            other._data = nullptr;
            other._size = 0;
            other._capacity = 0;
        }
        return *this;
    }

    // 访问
    T& operator[](size_t index) { return _data[index]; }
    const T& operator[](size_t index) const { return _data[index]; }

    T& at(size_t index) {
        if (index >= _size) return _data[0];
        return _data[index];
    }
    const T& at(size_t index) const {
        if (index >= _size) return _data[0];
        return _data[index];
    }

    T& front() { return _data[0]; }
    const T& front() const { return _data[0]; }
    T& back() { return _data[_size - 1]; }
    const T& back() const { return _data[_size - 1]; }

    T* data() { return _data; }
    const T* data() const { return _data; }

    // 容量
    size_t size() const { return _size; }
    size_t capacity() const { return _capacity; }
    bool empty() const { return _size == 0; }
    void reserve(size_t new_cap) {
        if (new_cap > _capacity) {
            _reallocate(new_cap);
        }
    }

    // 修改
    void clear() {
        for (size_t i = 0; i < _size; i++) {
            _data[i].~T();
        }
        _size = 0;
    }

    void push_back(const T& value) {
        if (_size >= _capacity) {
            _reallocate(_capacity == 0 ? 4 : _capacity * 2);
        }
        new (&_data[_size++]) T(value);
    }

    void push_back(T&& value) {
        if (_size >= _capacity) {
            _reallocate(_capacity == 0 ? 4 : _capacity * 2);
        }
        new (&_data[_size++]) T(std::move(value));
    }

    void pop_back() {
        if (_size > 0) {
            _data[--_size].~T();
        }
    }

    void resize(size_t new_size, const T& value = T()) {
        if (new_size < _size) {
            for (size_t i = new_size; i < _size; i++) {
                _data[i].~T();
            }
        } else if (new_size > _size) {
            if (new_size > _capacity) {
                _reallocate(new_size);
            }
            for (size_t i = _size; i < new_size; i++) {
                new (&_data[i]) T(value);
            }
        }
        _size = new_size;
    }

    void erase(size_t index) {
        if (index >= _size) return;
        _data[index].~T();
        for (size_t i = index; i < _size - 1; i++) {
            new (&_data[i]) T(std::move(_data[i + 1]));
        }
        _size--;
    }

    void erase(size_t first, size_t last) {
        if (first >= last || last > _size) return;
        for (size_t i = first; i < last; i++) {
            _data[i].~T();
        }
        for (size_t i = first; i < _size - (last - first); i++) {
            new (&_data[i]) T(std::move(_data[i + (last - first)]));
        }
        _size -= (last - first);
    }

    // 迭代器
    T* begin() { return _data; }
    T* end() { return _data + _size; }
    const T* begin() const { return _data; }
    const T* end() const { return _data + _size; }

    // 查找
    size_t find(const T& value) const {
        for (size_t i = 0; i < _size; i++) {
            if (_data[i] == value) return i;
        }
        return npos;
    }

    bool contains(const T& value) const {
        return find(value) != npos;
    }

    static constexpr size_t npos = static_cast<size_t>(-1);

private:
    T* _data;
    size_t _size;
    size_t _capacity;

    T* _allocate(size_t count) {
        if (count == 0) return nullptr;
        return reinterpret_cast<T*>(::operator new(count * sizeof(T)));
    }

    void _reallocate(size_t new_cap) {
        T* new_data = _allocate(new_cap);
        for (size_t i = 0; i < _size; i++) {
            new (&new_data[i]) T(std::move(_data[i]));
        }
        ::operator delete(_data);
        _data = new_data;
        _capacity = new_cap;
    }

    void _destroy() {
        if (_data) {
            for (size_t i = 0; i < _size; i++) {
                _data[i].~T();
            }
            ::operator delete(_data);
            _data = nullptr;
        }
        _size = 0;
        _capacity = 0;
    }
};

} // namespace MyEngine
