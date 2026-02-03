#pragma once

#include "object/RefCounted.h"
#include "containers/String.h"

namespace MyEngine {

class Texture2D : public RefCounted {
public:
    Texture2D();
    virtual ~Texture2D();

    virtual const char* get_class_name() const override { return "Texture2D"; }

    int get_width() const { return _width; }
    int get_height() const { return _height; }

protected:
    int _width = 0;
    int _height = 0;
};

} // namespace MyEngine
