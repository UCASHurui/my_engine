#include "editor/ImGuiPlatform.h"
#include "os/Input.h"

namespace MyEngine::Editor {

ImGuiPlatform::ImGuiPlatform() {
}

ImGuiPlatform::~ImGuiPlatform() {
    shutdown();
}

bool ImGuiPlatform::initialize(Window* window) {
    _window = window;
    return true;
}

void ImGuiPlatform::shutdown() {
}

void ImGuiPlatform::on_mouse_move(float x, float y) {
    (void)x;
    (void)y;
}

void ImGuiPlatform::on_mouse_button(int button, bool pressed) {
    (void)button;
    (void)pressed;
}

void ImGuiPlatform::on_mouse_wheel(float delta) {
    (void)delta;
}

void ImGuiPlatform::on_key_pressed(int key_code, bool pressed, bool repeat) {
    (void)key_code;
    (void)pressed;
    (void)repeat;
}

void ImGuiPlatform::on_char_input(unsigned int c) {
    (void)c;
}

void ImGuiPlatform::on_window_resize(int width, int height) {
    (void)width;
    (void)height;
}

void ImGuiPlatform::new_frame(float delta) {
    (void)delta;
}

const char* ImGuiPlatform::get_clipboard_text() {
    return "";
}

void ImGuiPlatform::set_clipboard_text(const char* text) {
    (void)text;
}

} // namespace MyEngine::Editor
