#pragma once

#include "core/math/Vector2.h"
#include "core/containers/String.h"

namespace MyEngine {

// 窗口模式
enum class WindowMode {
    WINDOWED,
    FULLSCREEN,
    BORDERLESS,
    MAXIMIZED
};

// 垂直同步
enum class VSyncMode {
    DISABLED,
    ENABLED,
    ADAPTIVE,
    MAILBOX
};

// 窗口事件
struct WindowEvent {
    enum Type {
        NONE,
        CLOSE,
        RESIZE,
        FOCUS_GAINED,
        FOCUS_LOST,
        KEY_PRESSED,
        KEY_RELEASED,
        MOUSE_ENTER,
        MOUSE_LEAVE
    } type = NONE;

    int width = 0;
    int height = 0;
    int key = 0;
};

// 窗口类
class Window {
public:
    struct Config {
        int width = 1280;
        int height = 720;
        String title = "Window";
        WindowMode mode = WindowMode::WINDOWED;
        VSyncMode vsync = VSyncMode::ENABLED;
        bool resizable = true;
        bool transparent = false;
        bool always_on_top = false;
    };

    static Window* create(const Config& config);
    static Window* create(int width, int height, const String& title);
    static void destroy(Window* window);

    virtual ~Window() = default;

    // 属性
    virtual int get_width() const = 0;
    virtual int get_height() const = 0;
    virtual Vector2 get_size() const { return Vector2(get_width(), get_height()); }
    virtual String get_title() const = 0;
    virtual void set_title(const String& title) = 0;
    virtual WindowMode get_mode() const = 0;
    virtual void set_mode(WindowMode mode) = 0;

    // 位置
    virtual Vector2 get_position() const = 0;
    virtual void set_position(const Vector2& pos) = 0;
    virtual void center() = 0;

    // 状态
    virtual bool should_close() const = 0;
    virtual bool is_focused() const = 0;
    virtual bool is_minimized() const = 0;
    virtual bool is_maximized() const = 0;

    // 渲染上下文
    virtual void* get_native_handle() const = 0;
    virtual void* get_native_display() const = 0;

    // 交换链
    virtual void swap_buffers() = 0;

    // 事件
    virtual bool poll_event(WindowEvent& event) = 0;

    // 鼠标
    virtual void set_mouse_position(const Vector2& pos) = 0;
    virtual void set_mouse_cursor_visible(bool visible) = 0;
    virtual void set_mouse_lock(bool locked) = 0;
    virtual bool is_mouse_locked() const = 0;

    // 截图
    virtual void capture_screenshot() = 0;

protected:
    static Window* _create_platform(const Config& config);

    // 子类实现
    virtual bool _init() = 0;
    virtual void _term() = 0;
};

} // namespace MyEngine
