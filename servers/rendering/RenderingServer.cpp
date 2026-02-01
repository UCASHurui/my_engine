#include "RenderingServer.h"
#include "core/os/OS.h"
#include "core/math/Transform.h"

namespace MyEngine {

RenderingServer& RenderingServer::get_singleton() {
    static RenderingServer instance;
    return instance;
}

void RenderingServer::initialize() {
    OS::print("RenderingServer initialized");
}

void RenderingServer::shutdown() {
    OS::print("RenderingServer shutdown");
}

void RenderingServer::viewport_set_size(Viewport* vp, int width, int height) {
    (void)vp; (void)width; (void)height;
}

void RenderingServer::viewport_set_active(Viewport* vp, bool active) {
    (void)vp; (void)active;
}

void RenderingServer::viewport_set_render_target(Viewport* vp, class RenderTarget* rt) {
    (void)vp; (void)rt;
}

void RenderingServer::camera_set_transform(Camera3D* cam, const Transform3D& transform) {
    (void)cam; (void)transform;
}

void RenderingServer::camera_set_projection(Camera3D* cam, int type, float size_or_fov,
                                             float aspect, float near, float far) {
    (void)cam; (void)type; (void)size_or_fov; (void)aspect; (void)near; (void)far;
}

void RenderingServer::camera_set_current(Camera3D* cam) {
    _current_camera = cam;
}

void RenderingServer::scenario_set_environment(class Scenario* scenario, Environment* env) {
    (void)scenario; (void)env;
}

void RenderingServer::scenario_set_reflection_probe(class Scenario* scenario, class ReflectionProbe* probe) {
    (void)scenario; (void)probe;
}

void RenderingServer::light_set_transform(Light3D* light, const Transform3D& transform) {
    (void)light; (void)transform;
}

void RenderingServer::light_set_param(Light3D* light, int param, float value) {
    (void)light; (void)param; (void)value;
}

void RenderingServer::light_set_shadow(Light3D* light, bool enabled) {
    (void)light; (void)enabled;
}

void RenderingServer::attach_instance(class Instance* instance) {
    (void)instance;
}

void RenderingServer::remove_instance(class Instance* instance) {
    (void)instance;
}

void RenderingServer::instance_set_transform(Instance* inst, const Transform3D& transform) {
    (void)inst; (void)transform;
}

void RenderingServer::draw() {
    _render_scene();
}

void RenderingServer::draw_viewport(Viewport* vp) {
    (void)vp;
}

void RenderingServer::render_set_texture_filter(int mode) {
    (void)mode;
}

void RenderingServer::render_set_texture_repeat(int mode) {
    (void)mode;
}

void RenderingServer::_render_scene() {
    // 渲染场景
}

} // namespace MyEngine
