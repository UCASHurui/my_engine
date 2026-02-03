#include "editor/PanelManager.h"
#include "io/FileAccess.h"
#include <algorithm>

namespace MyEngine::Editor {

PanelManager::PanelManager() {
}

PanelManager::~PanelManager() {
    shutdown();
}

void PanelManager::initialize() {
    // Create default panels
    // Panels will be registered via register_panel_type()
}

void PanelManager::shutdown() {
    for (auto& panel : _panels) {
        if (panel->is_visible()) {
            panel->on_shutdown();
        }
    }
    _panels.clear();
    _panel_by_name.clear();
}

BasePanel* PanelManager::create_panel(const String& type) {
    BasePanel* panel = create_panel(type);
    if (panel) {
        panel->on_init();
        _panels.push_back(std::unique_ptr<BasePanel>(panel));
        _panel_by_name[type] = panel;
    }
    return panel;
}

void PanelManager::destroy_panel(BasePanel* panel) {
    if (!panel) return;

    panel->on_shutdown();

    auto it = std::find_if(_panels.begin(), _panels.end(),
        [panel](const auto& p) { return p.get() == panel; });

    if (it != _panels.end()) {
        _panels.erase(it);
    }

    _panel_by_name.erase(panel->get_name());
}

void PanelManager::destroy_panel(const String& name) {
    BasePanel* panel = get_panel(name);
    if (panel) {
        destroy_panel(panel);
    }
}

BasePanel* PanelManager::get_panel(const String& name) const {
    auto it = _panel_by_name.find(name);
    return it != _panel_by_name.end() ? it->second : nullptr;
}

std::vector<BasePanel*> PanelManager::get_all_panels() const {
    std::vector<BasePanel*> result;
    result.reserve(_panels.size());
    for (const auto& panel : _panels) {
        result.push_back(panel.get());
    }
    return result;
}

void PanelManager::show_panel(const String& name) {
    BasePanel* panel = get_panel(name);
    if (panel) {
        panel->set_visible(true);
    }
}

void PanelManager::hide_panel(const String& name) {
    BasePanel* panel = get_panel(name);
    if (panel) {
        panel->set_visible(false);
    }
}

void PanelManager::toggle_panel(const String& name) {
    BasePanel* panel = get_panel(name);
    if (panel) {
        panel->toggle_visible();
    }
}

void PanelManager::update(float delta) {
    for (const auto& panel : _panels) {
        if (panel->is_visible()) {
            panel->on_update(delta);
        }
    }
}

void PanelManager::render() {
    // Panels will be rendered by the main editor loop
    // using their own ImGui windows/docking
}

void PanelManager::save_layout(const String& filepath) {
    // TODO: Serialize panel positions and states
    (void)filepath;
}

void PanelManager::load_layout(const String& filepath) {
    // TODO: Deserialize panel layout
    (void)filepath;
}

void PanelManager::reset_layout() {
    // TODO: Reset to default layout
}

void PanelManager::begin_docking() {
    _docking = true;
}

void PanelManager::end_docking() {
    _docking = false;
}

} // namespace MyEngine::Editor
