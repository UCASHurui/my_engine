#include "ClassDB.h"
#include "Object.h"

namespace MyEngine {

ClassDB& ClassDB::get_instance() {
    static ClassDB instance;
    return instance;
}

const ClassInfo* ClassDB::get_class(const String& class_name) const {
    auto it = _classes.find(class_name);
    if (it != _classes.end()) {
        return &it->value;
    }
    return nullptr;
}

bool ClassDB::class_exists(const String& class_name) const {
    return _classes.find(class_name) != _classes.end();
}

bool ClassDB::is_parent_class(const String& child, const String& parent) const {
    if (child == parent) return true;

    auto it = _classes.find(child);
    if (it != _classes.end()) {
        String current = it->value.parent_class;
        while (!current.empty()) {
            if (current == parent) return true;
            auto pit = _classes.find(current);
            if (pit != _classes.end()) {
                current = pit->value.parent_class;
            } else {
                break;
            }
        }
    }
    return false;
}

Object* ClassDB::instantiate(const String& class_name) {
    auto it = _classes.find(class_name);
    if (it != _classes.end() && it->value.creator) {
        return it->value.creator();
    }
    return nullptr;
}

std::vector<String> ClassDB::get_class_list() const {
    std::vector<String> result;
    for (const auto& kv : _classes) {
        result.push_back(kv.key);
    }
    return result;
}

} // namespace MyEngine
