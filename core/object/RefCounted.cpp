#include "RefCounted.h"

namespace MyEngine {

RefCounted::RefCounted() : _ref_count(0) {}

RefCounted::~RefCounted() {
    // 确保引用计数为0
}

void RefCounted::reference() {
    _ref_count++;
}

void RefCounted::unreference() {
    _ref_count--;
    if (_ref_count <= 0) {
        delete this;
    }
}

} // namespace MyEngine
