#include "editor/ScenePanel.h"
#include <imgui.h>

namespace MyEngine::Editor {

ScenePanel::ScenePanel()
    : BasePanel("Scene") {
}

void ScenePanel::on_render() {
    ImGui::Text("Scene Hierarchy");
    ImGui::Separator();

    // Placeholder for scene tree
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No scene loaded");

    if (ImGui::Button("Create Node")) {
        // TODO: Create new node
    }
}

} // namespace MyEngine::Editor
