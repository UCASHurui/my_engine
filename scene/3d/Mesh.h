#pragma once

#include "core/object/RefCounted.h"
#include "core/containers/String.h"

namespace MyEngine {

class Mesh : public RefCounted {
public:
    Mesh();
    virtual ~Mesh();

    virtual const char* get_class_name() const override { return "Mesh"; }
};

} // namespace MyEngine
