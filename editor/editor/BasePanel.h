#pragma once

#include "object/Object.h"
#include "containers/String.h"
#include <functional>

namespace MyEngine::Editor {

/**
 * Base class for all editor panels
 */
class BasePanel : public Object {

public:
    BasePanel();
    BasePanel(const String& name);
    virtual ~BasePanel();

    // Panel lifecycle
    virtual void on_init();
    virtual void on_shutdown();
    virtual void on_update(float delta);
    virtual void on_resize(int width, int height);

    // Panel identification
    const String& get_name() const { return _name; }
    void set_display_name(const String& name) { _display_name = name; }
    const String& get_display_name() const { return _display_name; }

    // Visibility
    bool is_visible() const { return _visible; }
    void set_visible(bool visible) { _visible = visible; }
    void toggle_visible() { _visible = !_visible; }

    // Docking
    bool is_docked() const { return _docked; }
    void set_docked(bool docked) { _docked = docked; }

    // Position and size
    int get_x() const { return _x; }
    int get_y() const { return _y; }
    int get_width() const { return _width; }
    int get_height() const { return _height; }

    // Focus
    bool has_focus() const { return _has_focus; }
    void set_has_focus(bool focus) { _has_focus = focus; }

    // Callback for panel events
    std::function<void(BasePanel*)> on_closed;
    std::function<void(BasePanel*)> on_focus_gained;
    std::function<void(BasePanel*)> on_focus_lost;

    // Public for ImGui access
    bool _visible = true;

public:
    // Override in subclasses to render the panel content
    virtual void on_render() = 0;

    String _name;
    String _display_name;
    bool _docked = true;
    bool _has_focus = false;

    int _x = 0;
    int _y = 0;
    int _width = 300;
    int _height = 400;

    bool _initialized = false;
};

// Panel factory function type
using PanelFactory = std::function<BasePanel*()>;

// Register a panel type
void register_panel_type(const String& type_name, PanelFactory factory);
BasePanel* create_panel(const String& type_name);

} // namespace MyEngine::Editor
