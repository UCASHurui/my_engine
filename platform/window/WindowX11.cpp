#include "core/os/Window.h"

namespace MyEngine {

// 简单的 X11 窗口存根实现（开发阶段使用）
class X11Window : public Window {
public:
    X11Window(const Config& config) : _config(config) {}

    bool _init() override { return true; }
    void _term() override {}

    int get_width() const override { return _config.width; }
    int get_height() const override { return _config.height; }
    String get_title() const override { return _config.title; }
    void set_title(const String& title) override { (void)title; }
    WindowMode get_mode() const override { return _config.mode; }
    void set_mode(WindowMode mode) override { (void)mode; }

    Vector2 get_position() const override { return Vector2(0, 0); }
    void set_position(const Vector2& pos) override { (void)pos; }
    void center() override {}

    bool should_close() const override { return _should_close; }
    bool is_focused() const override { return true; }
    bool is_minimized() const override { return false; }
    bool is_maximized() const override { return false; }

    void* get_native_handle() const override { return nullptr; }
    void* get_native_display() const override { return nullptr; }

    void swap_buffers() override {}

    bool poll_event(WindowEvent& event) override {
        (void)event;
        return false;
    }

    void set_mouse_position(const Vector2& pos) override { (void)pos; }
    void set_mouse_cursor_visible(bool visible) override { (void)visible; }
    void set_mouse_lock(bool locked) override { (void)locked; }
    bool is_mouse_locked() const override { return false; }

    void capture_screenshot() override {}

private:
    Config _config;
    bool _should_close = false;
};

// 工厂方法
Window* Window::_create_platform(const Config& config) {
    return new X11Window(config);
}

} // namespace MyEngine
