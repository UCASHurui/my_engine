#pragma once

#include "BasePanel.h"

namespace MyEngine::Editor {

/**
 * Scene panel showing the scene hierarchy
 */
class ScenePanel : public BasePanel {
public:
    ScenePanel();
    virtual ~ScenePanel() override = default;

protected:
    virtual void on_render() override;
};

} // namespace MyEngine::Editor
