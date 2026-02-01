#include <iostream>
#include "core/os/OS.h"
#include "core/os/Window.h"
#include "core/os/Input.h"

using namespace MyEngine;

int main(int argc, char* argv[]) {
    std::cout << "MyEngine v0.1.0" << std::endl;

    // 初始化平台层
    OS::initialize();
    Window::Config config;
    config.width = 1280;
    config.height = 720;
    config.title = "MyEngine";
    Window* window = Window::create(config);
    Input::initialize();

    // 主循环
    while (!window->should_close()) {
        Input::poll_events();
        window->swap_buffers();
    }

    // 清理
    Input::shutdown();
    Window::destroy(window);
    OS::shutdown();

    return 0;
}
