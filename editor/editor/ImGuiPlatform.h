#pragma once

#include "os/Window.h"
#include <imgui.h>

namespace MyEngine::Editor {

/**
 * ImGui platform integration for handling input and window events
 */
class ImGuiPlatform {
public:
    ImGuiPlatform();
    ~ImGuiPlatform();

    // Initialize with a window
    bool initialize(Window* window);
    void shutdown();

    // Handle input events
    void on_mouse_move(float x, float y);
    void on_mouse_button(int button, bool pressed);
    void on_mouse_wheel(float delta);
    void on_key_pressed(int key_code, bool pressed, bool repeat);
    void on_char_input(unsigned int c);
    void on_window_resize(int width, int height);

    // Update ImGui state
    void new_frame(float delta);

    // Get ImGui context
    ImGuiContext* get_context() const { return _context; }

    // Clipboard
    const char* get_clipboard_text();
    void set_clipboard_text(const char* text);

private:
    ImGuiContext* _context = nullptr;
    Window* _window = nullptr;
    bool _mouse_pressed[5] = {false, false, false, false, false};
};

} // namespace MyEngine::Editor
