#pragma once

#include "core/object/RefCounted.h"
#include "core/containers/String.h"

namespace MyEngine {

class Material : public RefCounted {
public:
    Material();
    virtual ~Material();

    virtual const char* get_class_name() const override { return "Material"; }
};

} // namespace MyEngine
