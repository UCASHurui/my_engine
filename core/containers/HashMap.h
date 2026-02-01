#pragma once

#include "Vector.h"
#include "String.h"
#include <functional>

namespace MyEngine {

// 简单哈希表 - 简化版本
template<typename K, typename V>
class HashMap {
public:
    struct Entry {
        K key;
        V value;
        bool occupied = false;
    };

    HashMap(size_t initial_capacity = 16) : _capacity(initial_capacity) {
        _data = new Entry[_capacity];
    }

    ~HashMap() {
        delete[] _data;
    }

    // 复制/移动
    HashMap(const HashMap& other) : _size(other._size), _capacity(other._capacity) {
        _data = new Entry[_capacity];
        for (size_t i = 0; i < _capacity; i++) {
            _data[i] = other._data[i];
        }
    }

    HashMap(HashMap&& other) noexcept : _data(other._data), _size(other._size), _capacity(other._capacity) {
        other._data = nullptr;
        other._size = 0;
        other._capacity = 0;
    }

    HashMap& operator=(const HashMap& other) {
        if (this != &other) {
            delete[] _data;
            _size = other._size;
            _capacity = other._capacity;
            _data = new Entry[_capacity];
            for (size_t i = 0; i < _capacity; i++) {
                _data[i] = other._data[i];
            }
        }
        return *this;
    }

    HashMap& operator=(HashMap&& other) noexcept {
        if (this != &other) {
            delete[] _data;
            _data = other._data;
            _size = other._size;
            _capacity = other._capacity;
            other._data = nullptr;
            other._size = 0;
            other._capacity = 0;
        }
        return *this;
    }

    // 查找 - 非 const 版本
    V& at(const K& key) {
        size_t idx = _find_index(key);
        if (idx == npos) {
            _rehash_if_needed();
            idx = _insert(key, V());
        }
        return _data[idx].value;
    }

    const V& at(const K& key) const {
        size_t idx = _find_index(key);
        if (idx == npos) {
            static V dummy;
            return dummy;
        }
        return _data[idx].value;
    }

    bool contains(const K& key) const {
        return _find_index(key) != npos;
    }

    // 查找 - 返回迭代器
    struct Iterator {
        Entry* entry;
        Entry* end;

        Iterator& operator++() {
            do {
                entry++;
            } while (entry < end && !entry->occupied);
            return *this;
        }

        bool operator==(const Iterator& other) const { return entry == other.entry; }
        bool operator!=(const Iterator& other) const { return entry != other.entry; }

        Entry& operator*() { return *entry; }
        const Entry& operator*() const { return *entry; }
        Entry* operator->() { return entry; }
    };

    Iterator begin() {
        Iterator it{_data, _data + _capacity};
        if (_capacity > 0 && !_data[0].occupied) {
            ++it;
        }
        return it;
    }

    Iterator end() {
        return Iterator{_data + _capacity, _data + _capacity};
    }

    Iterator find(const K& key) {
        size_t idx = _find_index(key);
        if (idx == npos) return end();
        return Iterator{_data + idx, _data + _capacity};
    }

    // Const 版本
    struct ConstIterator {
        const Entry* entry;
        const Entry* end;

        ConstIterator& operator++() {
            do {
                entry++;
            } while (entry < end && !entry->occupied);
            return *this;
        }

        bool operator==(const ConstIterator& other) const { return entry == other.entry; }
        bool operator!=(const ConstIterator& other) const { return entry != other.entry; }

        const Entry& operator*() const { return *entry; }
        const Entry* operator->() const { return entry; }
    };

    ConstIterator begin() const {
        ConstIterator it{_data, _data + _capacity};
        if (_capacity > 0 && !_data[0].occupied) {
            ++it;
        }
        return it;
    }

    ConstIterator end() const {
        return ConstIterator{_data + _capacity, _data + _capacity};
    }

    ConstIterator find(const K& key) const {
        size_t idx = _find_index(key);
        if (idx == npos) return end();
        return ConstIterator{_data + idx, _data + _capacity};
    }

    // 修改
    void insert(const K& key, const V& value) {
        if (contains(key)) return;
        _rehash_if_needed();
        _insert(key, value);
    }

    bool erase(const K& key) {
        size_t idx = _find_index(key);
        if (idx == npos) return false;
        _data[idx].occupied = false;
        _size--;
        return true;
    }

    void clear() {
        for (size_t i = 0; i < _capacity; i++) {
            _data[i].occupied = false;
        }
        _size = 0;
    }

    // 查询
    size_t size() const { return _size; }
    bool empty() const { return _size == 0; }

    static constexpr size_t npos = (size_t)-1;

private:
    Entry* _data;
    size_t _size = 0;
    size_t _capacity;

    // String 专用哈希
    static size_t _hash_string(const String& key) {
        return hash_string(key);
    }

    // 默认哈希 - 委托给 std::hash
    template<typename T>
    static size_t _hash_default(const T& key) {
        return std::hash<T>{}(key);
    }

    // 哈希分发
    static size_t _hash(const String& key) {
        return _hash_string(key);
    }

    template<typename T>
    static size_t _hash(const T& key) {
        return _hash_default(key);
    }

    size_t _find_index(const K& key) const {
        size_t start = _hash(key) % _capacity;
        for (size_t i = 0; i < _capacity; i++) {
            size_t idx = (start + i) % _capacity;
            if (!_data[idx].occupied) {
                return npos;
            }
            if (_data[idx].key == key) {
                return idx;
            }
        }
        return npos;
    }

    void _insert(const K& key, const V& value) {
        size_t start = _hash(key) % _capacity;
        for (size_t i = 0; i < _capacity; i++) {
            size_t idx = (start + i) % _capacity;
            if (!_data[idx].occupied) {
                _data[idx].key = key;
                _data[idx].value = value;
                _data[idx].occupied = true;
                _size++;
                return;
            }
        }
    }

    void _rehash_if_needed() {
        float load_factor = static_cast<float>(_size) / _capacity;
        if (load_factor > 0.7f) {
            _rehash(_capacity * 2);
        }
    }

    void _rehash(size_t new_cap) {
        Entry* old_data = _data;
        size_t old_cap = _capacity;

        _capacity = new_cap;
        _data = new Entry[_capacity];
        _size = 0;

        for (size_t i = 0; i < old_cap; i++) {
            if (old_data[i].occupied) {
                _insert(old_data[i].key, old_data[i].value);
            }
        }

        delete[] old_data;
    }
};

} // namespace MyEngine
