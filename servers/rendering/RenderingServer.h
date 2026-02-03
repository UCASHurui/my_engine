#pragma once

#include "containers/Vector.h"
#include "containers/String.h"
#include "math/Transform.h"
#include "Image.h"

namespace MyEngine {

class Viewport;
class Camera3D;
class Light3D;
class Environment;

// 渲染服务器 - 单例，管理渲染资源
class RenderingServer {
public:
    static RenderingServer& get_singleton();

    // 初始化/销毁
    void initialize();
    void shutdown();

    // 视口
    void viewport_set_size(Viewport* vp, int width, int height);
    void viewport_set_active(Viewport* vp, bool active);
    void viewport_set_render_target(Viewport* vp, class RenderTarget* rt);

    // 相机
    void camera_set_transform(Camera3D* cam, const Transform3D& transform);
    void camera_set_projection(Camera3D* cam, int type, float size_or_fov,
                               float aspect, float near, float far);
    void camera_set_current(Camera3D* cam);

    // 场景
    void scenario_set_environment(class Scenario* scenario, Environment* env);
    void scenario_set_reflection_probe(class Scenario* scenario, class ReflectionProbe* probe);

    // 灯光
    void light_set_transform(Light3D* light, const Transform3D& transform);
    void light_set_param(Light3D* light, int param, float value);
    void light_set_shadow(Light3D* light, bool enabled);

    // 世界
    void attach_instance(class Instance* instance);
    void remove_instance(class Instance* instance);
    void instance_set_transform(Instance* inst, const Transform3D& transform);

    // 绘制
    void draw();
    void draw_viewport(Viewport* vp);

    // 设置
    void set_clear_color(const Color& color);
    Color get_clear_color() const { return _clear_color; }

    // 渲染状态
    void render_set_texture_filter(int mode);
    void render_set_texture_repeat(int mode);

private:
    RenderingServer() = default;
    ~RenderingServer() = default;

    Vector<Viewport*> _active_viewports;
    Camera3D* _current_camera = nullptr;
    Color _clear_color = Color(0.0f, 0.0f, 0.0f, 1.0f);  // BLACK

    // 内部渲染
    void _render_scene();
};

} // namespace MyEngine
