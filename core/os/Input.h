#pragma once

#include "containers/String.h"
#include "math/Vector2.h"
#include "containers/Vector.h"
#include <set>
#include <map>
#include <vector>
#include <cstdint>

namespace MyEngine {

// 按键代码
enum class Key : uint16_t {
    NONE = 0,
    SPACE = ' ',
    APOSTROPHE = '\'',
    COMMA = ',',
    MINUS = '-',
    PERIOD = '.',
    SLASH = '/',
    NUM_0 = '0',
    NUM_1 = '1',
    NUM_2 = '2',
    NUM_3 = '3',
    NUM_4 = '4',
    NUM_5 = '5',
    NUM_6 = '6',
    NUM_7 = '7',
    NUM_8 = '8',
    NUM_9 = '9',
    SEMICOLON = ';',
    EQUAL = '=',
    A = 'A',
    B = 'B',
    C = 'C',
    D = 'D',
    E = 'E',
    F = 'F',
    G = 'G',
    H = 'H',
    I = 'I',
    J = 'J',
    K = 'K',
    L = 'L',
    M = 'M',
    N = 'N',
    O = 'O',
    P = 'P',
    Q = 'Q',
    R = 'R',
    S = 'S',
    T = 'T',
    U = 'U',
    V = 'V',
    W = 'W',
    X = 'X',
    Y = 'Y',
    Z = 'Z',
    LEFT_BRACKET = '[',
    BACKSLASH = '\\',
    RIGHT_BRACKET = ']',
    GRAVE_ACCENT = '`',
    ESCAPE = 256,
    ENTER,
    TAB,
    BACKSPACE,
    INSERT,
    DELETE,
    RIGHT,
    LEFT,
    DOWN,
    UP,
    PAGE_UP,
    PAGE_DOWN,
    HOME,
    END,
    CAPS_LOCK,
    SCROLL_LOCK,
    NUM_LOCK,
    PRINT_SCREEN,
    PAUSE,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
    F11, F12, F13, F14, F15, F16, F17, F18, F19, F20,
    F21, F22, F23, F24,
    KP_0, KP_1, KP_2, KP_3, KP_4, KP_5, KP_6, KP_7, KP_8, KP_9,
    KP_DECIMAL,
    KP_DIVIDE,
    KP_MULTIPLY,
    KP_SUBTRACT,
    KP_ADD,
    KP_ENTER,
    KP_EQUAL,
    LEFT_SHIFT,
    LEFT_CONTROL,
    LEFT_ALT,
    LEFT_SUPER,
    RIGHT_SHIFT,
    RIGHT_CONTROL,
    RIGHT_ALT,
    RIGHT_SUPER,
    MENU,
    MAX
};

// 鼠标按钮
enum class MouseButton : uint8_t {
    NONE = 0,
    LEFT = 1,
    RIGHT = 2,
    MIDDLE = 3,
    BUTTON_4 = 4,
    BUTTON_5 = 5,
    BUTTON_6 = 6,
    BUTTON_7 = 7,
    BUTTON_8 = 8,
    MAX = 8
};

// 输入模式
enum class InputMode {
    KEYBOARD,
    MOUSE,
    JOYSTICK,
    TOUCH,
    ALL
};

// 输入事件
struct InputEvent {
    enum Type {
        NONE,
        KEY_PRESSED,
        KEY_RELEASED,
        MOUSE_MOVED,
        MOUSE_BUTTON_PRESSED,
        MOUSE_BUTTON_RELEASED,
        MOUSE_WHEEL,
        TOUCH_BEGAN,
        TOUCH_MOVED,
        TOUCH_ENDED,
        JOYSTICK_BUTTON_PRESSED,
        JOYSTICK_BUTTON_RELEASED,
        JOYSTICK_AXIS
    } type = NONE;

    Key key = Key::NONE;
    MouseButton mouse_button = MouseButton::NONE;
    Vector2 mouse_position;
    Vector2 mouse_delta;
    float mouse_wheel = 0.0f;

    int touch_id = -1;
    Vector2 touch_position;

    int joystick_id = 0;
    uint8_t joystick_button = 0;
    uint8_t joystick_axis = 0;
    float joystick_axis_value = 0.0f;

    bool is_action(const String& action) const;
};

// 输入服务器
class Input {
public:
    static void initialize();
    static void shutdown();
    static void poll_events();

    // 键盘
    static bool is_key_pressed(Key key);
    static bool is_key_just_pressed(Key key);
    static bool is_key_just_released(Key key);

    // 鼠标
    static Vector2 get_mouse_position();
    static Vector2 get_mouse_delta();
    static bool is_mouse_button_pressed(MouseButton button);
    static float get_mouse_wheel();

    // 触摸
    static Vector2 get_touch_position(int id);
    static bool is_touch_active(int id);

    // 手柄
    static int get_connected_joystick_count();
    static bool is_joystick_button_pressed(int joy_id, uint8_t button);
    static float get_joystick_axis(int joy_id, uint8_t axis);

    // 动作映射
    static void action_bind(const String& action, Key key);
    static void action_bind(const String& action, MouseButton button);
    static bool is_action_pressed(const String& action);
    static bool is_action_just_pressed(const String& action);

    // 事件队列
    static void push_event(const InputEvent& event);
    static Vector<InputEvent>& get_event_queue() { return _event_queue; }

    // 鼠标锁定
    static void set_mouse_mode(bool locked);
    static bool is_mouse_locked() { return _mouse_locked; }

    // 虚拟键盘 (移动端)
    static void show_virtual_keyboard();
    static void hide_virtual_keyboard();

private:
    static std::set<Key> _keys_pressed;
    static std::set<Key> _keys_just_pressed;
    static std::set<Key> _keys_just_released;
    static std::set<MouseButton> _mouse_buttons_pressed;
    static std::set<MouseButton> _mouse_buttons_just_pressed;
    static std::set<MouseButton> _mouse_buttons_just_released;
    static Vector2 _mouse_position;
    static Vector2 _mouse_delta;
    static float _mouse_wheel;
    static bool _mouse_locked;

    static std::map<String, std::vector<Key>> _action_key_map;
    static std::map<String, std::vector<MouseButton>> _action_mouse_map;
    static std::set<String> _actions_just_pressed;

    static Vector<InputEvent> _event_queue;
    static Vector<InputEvent> _input_buffer;

    static void _process_event(const InputEvent& event);
    static void _clear_frame_state();
    static void _update_joystick_state();
};

} // namespace MyEngine
