#include "editor/BasePanel.h"
#include <map>

namespace MyEngine::Editor {

// Static registry for panel types
static std::map<String, PanelFactory> _panel_factories;

BasePanel::BasePanel()
    : _name("BasePanel") {
}

BasePanel::BasePanel(const String& name)
    : _name(name) {
    _display_name = name;
}

BasePanel::~BasePanel() {
}

void BasePanel::on_init() {
    _initialized = true;
}

void BasePanel::on_shutdown() {
}

void BasePanel::on_update(float delta) {
    (void)delta;
}

void BasePanel::on_resize(int width, int height) {
    _width = width;
    _height = height;
}

void register_panel_type(const String& type_name, PanelFactory factory) {
    _panel_factories[type_name] = factory;
}

BasePanel* create_panel(const String& type_name) {
    auto it = _panel_factories.find(type_name);
    if (it != _panel_factories.end()) {
        return it->second();
    }
    return nullptr;
}

} // namespace MyEngine::Editor
