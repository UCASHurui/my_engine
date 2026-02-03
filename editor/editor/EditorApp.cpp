#include "editor/EditorApp.h"
#include "os/OS.h"
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

    // Simple menu
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit")) {
                request_exit();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Scene", nullptr);
            ImGui::MenuItem("Inspector", nullptr);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    // Simple window
    ImGui::Begin("Welcome");
    ImGui::Text("MyEngine Editor");
    ImGui::Text("Version 0.1.0");
    ImGui::End();
}

void EditorApp::render() {
    ImGui::Render();
}

void EditorApp::shutdown() {
    _initialized = false;
    _running = false;
    std::cout << "Editor shutdown complete" << std::endl;
}

void EditorApp::request_exit() {
    _running = false;
}

} // namespace MyEngine::Editor
