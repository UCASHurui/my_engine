#pragma once

#include "Object.h"

namespace MyEngine {

// 引用计数基类
class RefCounted : public Object {
public:
    RefCounted();
    virtual ~RefCounted();

    void reference();
    void unreference();
    bool is_reference_counted() const { return true; }

    int get_reference_count() const { return _ref_count; }

protected:
    int _ref_count = 0;
};

} // namespace MyEngine
