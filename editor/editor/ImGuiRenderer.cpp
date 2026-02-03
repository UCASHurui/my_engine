#include "editor/ImGuiRenderer.h"

namespace MyEngine::Editor {

ImGuiRenderer::ImGuiRenderer() {
}

ImGuiRenderer::~ImGuiRenderer() {
    shutdown();
}

bool ImGuiRenderer::initialize() {
    return true;
}

void ImGuiRenderer::shutdown() {
}

void ImGuiRenderer::render(ImDrawData* draw_data) {
    (void)draw_data;
}

void ImGuiRenderer::on_resize(int width, int height) {
    (void)width;
    (void)height;
}

} // namespace MyEngine::Editor
