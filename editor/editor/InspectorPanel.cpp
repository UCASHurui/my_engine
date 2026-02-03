#include "editor/InspectorPanel.h"
#include <imgui.h>

namespace MyEngine::Editor {

InspectorPanel::InspectorPanel()
    : BasePanel("Inspector") {
}

void InspectorPanel::set_target(void* target) {
    _target = target;
}

void InspectorPanel::on_render() {
    ImGui::Text("Inspector");
    ImGui::Separator();

    if (_target == nullptr) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No object selected");
        return;
    }

    // Placeholder property editor
    ImGui::Text("Object: %p", _target);
    ImGui::Spacing();

    // Placeholder properties
    static char name_buffer[256] = "";
    ImGui::InputText("Name", name_buffer, sizeof(name_buffer));

    static float position[3] = {0.0f, 0.0f, 0.0f};
    ImGui::InputFloat3("Position", position);

    static float rotation[3] = {0.0f, 0.0f, 0.0f};
    ImGui::InputFloat3("Rotation", rotation);

    static float scale[3] = {1.0f, 1.0f, 1.0f};
    ImGui::InputFloat3("Scale", scale);
}

} // namespace MyEngine::Editor
