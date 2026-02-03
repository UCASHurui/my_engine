#pragma once

#include "object/RefCounted.h"
#include "containers/String.h"

namespace MyEngine {

class Material : public RefCounted {
public:
    Material();
    virtual ~Material();

    virtual const char* get_class_name() const override { return "Material"; }
};

} // namespace MyEngine
