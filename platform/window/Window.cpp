#include "os/Window.h"

namespace MyEngine {

Window* Window::create(const Config& config) {
    return _create_platform(config);
}

Window* Window::create(int width, int height, const String& title) {
    Config config;
    config.width = width;
    config.height = height;
    config.title = title;
    return create(config);
}

void Window::destroy(Window* window) {
    if (window) {
        window->_term();
        delete window;
    }
}

} // namespace MyEngine
