#pragma once

#include "object/RefCounted.h"
#include "containers/String.h"

namespace MyEngine {

class Mesh : public RefCounted {
public:
    Mesh();
    virtual ~Mesh();

    virtual const char* get_class_name() const override { return "Mesh"; }
};

} // namespace MyEngine
