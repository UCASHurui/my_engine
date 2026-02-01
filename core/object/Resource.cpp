#include "Resource.h"
#include "core/os/OS.h"

namespace MyEngine {

Resource::Resource() = default;

Resource::~Resource() = default;

bool Resource::_load(const std::string& path) {
    (void)path;
    return true;
}

void Resource::_unload() {
    _loaded = false;
}

} // namespace MyEngine
