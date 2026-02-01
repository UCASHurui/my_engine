#include "Environment.h"

namespace MyEngine {

Environment::Environment() = default;

Environment::~Environment() = default;

Ref<Environment> Environment::create_default() {
    Ref<Environment> env = new Environment();
    env->set_background_type(BackgroundType::COLOR);
    env->set_background_color(Color(0.1f, 0.1f, 0.15f, 1.0f));
    env->set_ambient_light_color(Color(0.3f, 0.3f, 0.35f, 1.0f));
    env->set_ambient_light_energy(0.5f);
    return env;
}

} // namespace MyEngine
