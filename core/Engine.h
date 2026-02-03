#pragma once

#include "variant/Variant.h"
#include "object/Object.h"

namespace MyEngine {

// 引擎配置
struct EngineConfig {
    int width = 1280;
    int height = 720;
    String title = "MyEngine";
    bool fullscreen = false;
    int target_fps = 60;
};

// 全局配置
extern EngineConfig g_config;

// 引擎状态
enum class EngineState {
    UNINITIALIZED,
    INITIALIZING,
    RUNNING,
    PAUSED,
    SHUTTING_DOWN
};

class Engine {
public:
    static Engine& get_instance();

    // 生命周期
    bool initialize(const EngineConfig& config = EngineConfig());
    void run();
    void shutdown();

    // 状态
    EngineState get_state() const { return _state; }
    float get_delta_time() const { return _delta_time; }
    float get_total_time() const { return _total_time; }

private:
    Engine() = default;
    ~Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void _main_loop();

    EngineState _state = EngineState::UNINITIALIZED;
    float _delta_time = 0.0f;
    float _total_time = 0.0f;
    uint64_t _frame_count = 0;
};

} // namespace MyEngine

// 便捷宏
#define ENGINE MyEngine::Engine::get_instance()
