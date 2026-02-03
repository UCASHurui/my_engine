#pragma once

#include <string>
#include <vector>

namespace MyEngine::Editor {

// Editor configuration
struct EditorConfig {
    // Window settings
    int window_width = 1600;
    int window_height = 900;
    std::string window_title = "MyEngine Editor";

    // UI settings
    float ui_scale = 1.0f;
    bool use_docking = true;
    bool use_floating_panels = false;

    // Theme
    enum class Theme { Light, Dark, Classic };
    Theme theme = Theme::Dark;

    // Layout
    std::vector<std::string> default_panels = {
        "ScenePanel",
        "InspectorPanel",
        "ViewportPanel",
        "ConsolePanel"
    };

    // Paths
    std::string project_dir = "./";
    std::string layout_file = "./editor_layout.ini";
};

// Global config instance
inline EditorConfig& GetConfig() {
    static EditorConfig config;
    return config;
}

} // namespace MyEngine::Editor
