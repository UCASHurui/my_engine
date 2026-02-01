#include "core/os/Input.h"

namespace MyEngine {

std::set<Key> Input::_keys_pressed;
std::set<Key> Input::_keys_just_pressed;
std::set<Key> Input::_keys_just_released;
std::set<MouseButton> Input::_mouse_buttons_pressed;
std::set<MouseButton> Input::_mouse_buttons_just_pressed;
std::set<MouseButton> Input::_mouse_buttons_just_released;
Vector2 Input::_mouse_position;
Vector2 Input::_mouse_delta;
float Input::_mouse_wheel = 0.0f;
bool Input::_mouse_locked = false;

Vector<InputEvent> Input::_event_queue;
Vector<InputEvent> Input::_input_buffer;

std::map<String, std::vector<Key>> Input::_action_key_map;
std::map<String, std::vector<MouseButton>> Input::_action_mouse_map;
std::set<String> Input::_actions_just_pressed;

void Input::initialize() {
    _keys_pressed.clear();
    _keys_just_pressed.clear();
    _mouse_buttons_pressed.clear();
    _event_queue.clear();
}

void Input::shutdown() {
    _keys_pressed.clear();
    _mouse_buttons_pressed.clear();
    _event_queue.clear();
}

void Input::poll_events() {
    _clear_frame_state();
    _input_buffer = _event_queue;
    _event_queue.clear();

    for (const auto& event : _input_buffer) {
        _process_event(event);
    }
}

bool Input::is_key_pressed(Key key) {
    return _keys_pressed.find(key) != _keys_pressed.end();
}

bool Input::is_key_just_pressed(Key key) {
    return _keys_just_pressed.find(key) != _keys_just_pressed.end();
}

bool Input::is_key_just_released(Key key) {
    return _keys_just_released.find(key) != _keys_just_released.end();
}

Vector2 Input::get_mouse_position() {
    return _mouse_position;
}

Vector2 Input::get_mouse_delta() {
    return _mouse_delta;
}

bool Input::is_mouse_button_pressed(MouseButton button) {
    return _mouse_buttons_pressed.find(button) != _mouse_buttons_pressed.end();
}

float Input::get_mouse_wheel() {
    return _mouse_wheel;
}

void Input::push_event(const InputEvent& event) {
    _event_queue.push_back(event);
}

void Input::_clear_frame_state() {
    _keys_just_pressed.clear();
    _keys_just_released.clear();
    _mouse_buttons_just_pressed.clear();
    _mouse_buttons_just_released.clear();
    _mouse_delta = Vector2::ZERO;
    _mouse_wheel = 0.0f;
    _actions_just_pressed.clear();
}

void Input::_process_event(const InputEvent& event) {
    switch (event.type) {
        case InputEvent::KEY_PRESSED:
            if (_keys_pressed.find(event.key) == _keys_pressed.end()) {
                _keys_just_pressed.insert(event.key);
            }
            _keys_pressed.insert(event.key);
            break;
        case InputEvent::KEY_RELEASED:
            _keys_pressed.erase(event.key);
            _keys_just_released.insert(event.key);
            break;
        case InputEvent::MOUSE_BUTTON_PRESSED:
            if (_mouse_buttons_pressed.find(event.mouse_button) == _mouse_buttons_pressed.end()) {
                _mouse_buttons_just_pressed.insert(event.mouse_button);
            }
            _mouse_buttons_pressed.insert(event.mouse_button);
            break;
        case InputEvent::MOUSE_BUTTON_RELEASED:
            _mouse_buttons_pressed.erase(event.mouse_button);
            _mouse_buttons_just_released.insert(event.mouse_button);
            break;
        case InputEvent::MOUSE_MOVED:
            _mouse_delta = event.mouse_delta;
            _mouse_position = event.mouse_position;
            break;
        case InputEvent::MOUSE_WHEEL:
            _mouse_wheel = event.mouse_wheel;
            break;
        default:
            break;
    }
}

void Input::action_bind(const String& action, Key key) {
    _action_key_map[action].push_back(key);
}

void Input::action_bind(const String& action, MouseButton button) {
    _action_mouse_map[action].push_back(button);
}

bool Input::is_action_pressed(const String& action) {
    auto it_key = _action_key_map.find(action);
    if (it_key != _action_key_map.end()) {
        for (Key key : it_key->second) {
            if (is_key_pressed(key)) return true;
        }
    }
    auto it_mouse = _action_mouse_map.find(action);
    if (it_mouse != _action_mouse_map.end()) {
        for (MouseButton btn : it_mouse->second) {
            if (is_mouse_button_pressed(btn)) return true;
        }
    }
    return false;
}

bool Input::is_action_just_pressed(const String& action) {
    return _actions_just_pressed.find(action) != _actions_just_pressed.end();
}

} // namespace MyEngine
