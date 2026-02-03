#include "editor/EditorApp.h"
#include "os/OS.h"
#include <iostream>

using namespace MyEngine;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::cout << "Starting MyEngine Editor..." << std::endl;

    // Create and initialize editor
    Editor::EditorApp& editor = Editor::EditorApp::instance();

    Editor::EditorConfig config;
    config.window_title = "MyEngine Editor";
    config.window_width = 1600;
    config.window_height = 900;

    if (!editor.initialize(config)) {
        std::cerr << "Failed to initialize editor" << std::endl;
        return 1;
    }

    // Run editor
    editor.run();

    // Shutdown
    editor.shutdown();

    std::cout << "Editor exited." << std::endl;
    return 0;
}
