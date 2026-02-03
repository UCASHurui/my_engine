#include "editor/EditorApp.h"
#include "os/OS.h"
#include "PanelManager.h"
#include "ScenePanel.h"
#include "InspectorPanel.h"
#include <iostream>
#include <imgui.h>

namespace MyEngine::Editor {

EditorApp& EditorApp::instance() {
    static EditorApp app;
    return app;
}

EditorApp::EditorApp() {
}

EditorApp::~EditorApp() {
    shutdown();
}

bool EditorApp::initialize(const EditorConfig& config) {
    _config = config;
    _initialized = true;
    _running = true;

    // Initialize default panels
    GetPanelManager().initialize();

    // Create scene panel
    GetPanelManager().add_panel(new ScenePanel());

    // Create inspector panel
    GetPanelManager().add_panel(new InspectorPanel());

    std::cout << "Editor initialized successfully" << std::endl;
    return true;
}

void EditorApp::run() {
    if (!_initialized) {
        std::cerr << "Editor not initialized" << std::endl;
        return;
    }

    _last_time = OS::get_ticks_msec();

    while (_running) {
        // Calculate delta time
        float current_time = OS::get_ticks_msec();
        _delta_time = (current_time - _last_time) / 1000.0f;
        _last_time = current_time;

        // Process input
        process_input();

        // Update
        update(_delta_time);

        // Render
        render();
    }
}

void EditorApp::process_input() {
}

void EditorApp::update(float delta) {
    (void)delta;
    // ImGui new frame
    ImGui::NewFrame();

    // Main menu bar
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Scene", "Ctrl+N")) {
                // TODO: New scene
            }
            if (ImGui::MenuItem("Open Scene", "Ctrl+O")) {
                // TODO: Open scene
            }
            if (ImGui::MenuItem("Save Scene", "Ctrl+S")) {
                // TODO: Save scene
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit", "Alt+F4")) {
                request_exit();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            // Toggle panels
            static bool show_scene = true;
            static bool show_inspector = true;
            ImGui::MenuItem("Scene", nullptr, &show_scene);
            ImGui::MenuItem("Inspector", nullptr, &show_inspector);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Tools")) {
            ImGui::MenuItem("Profiler", nullptr);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            ImGui::MenuItem("About", nullptr);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    // Simple panel layout
    static float scene_width = 250.0f;

    ImGui::SetNextWindowPos(ImVec2(0, 20));
    ImGui::SetNextWindowSize(ImVec2(1600, 880));
    ImGui::Begin("Editor", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    // Scene panel on left
    ImGui::BeginChild("ScenePanel", ImVec2(scene_width, -1), true);
    BasePanel* scene = GetPanelManager().get_panel("Scene");
    if (scene) {
        scene->on_render();
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // Inspector panel on right
    ImGui::BeginChild("InspectorPanel", ImVec2(-1, -1), true);
    BasePanel* inspector = GetPanelManager().get_panel("Inspector");
    if (inspector) {
        inspector->on_render();
    }
    ImGui::EndChild();

    ImGui::End();
}

void EditorApp::render() {
    ImGui::Render();
}

void EditorApp::shutdown() {
    GetPanelManager().shutdown();
    _initialized = false;
    _running = false;
    std::cout << "Editor shutdown complete" << std::endl;
}

void EditorApp::request_exit() {
    _running = false;
}

} // namespace MyEngine::Editor
