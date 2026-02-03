#pragma once

#include <GL/gl.h>
#include <imgui.h>

namespace MyEngine::Editor {

/**
 * ImGui renderer for OpenGL
 * Handles rendering ImGui draw data to an OpenGL framebuffer
 */
class ImGuiRenderer {
public:
    ImGuiRenderer();
    ~ImGuiRenderer();

    // Initialize OpenGL resources
    bool initialize();

    // Shutdown and free resources
    void shutdown();

    // Render ImGui draw data
    void render(ImDrawData* draw_data);

    // Handle window resize
    void on_resize(int width, int height);

private:
    // OpenGL shader program
    GLuint _shader = 0;
    GLuint _vertex_buffer = 0;
    GLuint _index_buffer = 0;
    GLuint _font_texture = 0;

    int _window_width = 0;
    int _window_height = 0;

    // Compile shader program
    bool create_shader_program();

    // Create font texture
    bool create_font_texture();

    // Clean up OpenGL resources
    void cleanup();
};

} // namespace MyEngine::Editor
