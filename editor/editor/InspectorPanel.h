#pragma once

#include "BasePanel.h"

namespace MyEngine::Editor {

/**
 * Inspector panel for editing selected objects
 */
class InspectorPanel : public BasePanel {
public:
    InspectorPanel();
    virtual ~InspectorPanel() override = default;

    // Set the currently inspected object
    void set_target(void* target);
    void* get_target() const { return _target; }

protected:
    virtual void on_render() override;

private:
    void* _target = nullptr;
};

} // namespace MyEngine::Editor
