#pragma once

#include "object/Object.h"
#include "EditorConfig.h"
#include <functional>

namespace MyEngine::Editor {

/**
 * Main editor application class
 * Handles initialization, main loop, and shutdown
 */
class EditorApp {

public:
    EditorApp();
    ~EditorApp();

    // Initialize the editor
    bool initialize(const EditorConfig& config = EditorConfig());

    // Main editor loop
    void run();

    // Shutdown the editor
    void shutdown();

    // Request to exit the editor
    void request_exit();

    // Callbacks
    std::function<void()> on_initialize;
    std::function<void(float delta)> on_update;
    std::function<void()> on_shutdown;

    // Static instance
    static EditorApp& instance();

private:
    // Main loop functions
    void process_input();
    void update(float delta);
    void render();

    EditorConfig _config;
    bool _running = false;
    bool _initialized = false;

    float _delta_time = 0.0f;
    float _last_time = 0.0f;
};

// Get global editor instance
inline EditorApp& GetEditor() {
    return EditorApp::instance();
}

} // namespace MyEngine::Editor
