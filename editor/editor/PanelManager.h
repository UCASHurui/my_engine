#pragma once

#include "BasePanel.h"
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>

namespace MyEngine::Editor {

/**
 * Manages all editor panels
 */
class PanelManager : public Object {

public:
    PanelManager();
    ~PanelManager();

    // Initialize/shutdown
    void initialize();
    void shutdown();

    // Panel lifecycle
    BasePanel* create_panel(const String& type);
    void add_panel(BasePanel* panel);
    void destroy_panel(BasePanel* panel);
    void destroy_panel(const String& name);

    // Panel access
    BasePanel* get_panel(const String& name) const;
    std::vector<BasePanel*> get_all_panels() const;

    // Visibility
    void show_panel(const String& name);
    void hide_panel(const String& name);
    void toggle_panel(const String& name);

    // Update and render
    void update(float delta);

    // Layout
    void save_layout(const String& filepath);
    void load_layout(const String& filepath);
    void reset_layout();

    // Docking support
    void begin_docking();
    void end_docking();

private:
    std::vector<std::unique_ptr<BasePanel>> _panels;
    std::map<String, BasePanel*> _panel_by_name;

    bool _docking = false;
};

// Global panel manager instance
inline PanelManager& GetPanelManager() {
    static PanelManager manager;
    return manager;
}

} // namespace MyEngine::Editor
