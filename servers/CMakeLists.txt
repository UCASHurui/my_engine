# Servers 模块
add_library(EngineServers STATIC)
target_sources(EngineServers PRIVATE
    servers/rendering/RenderingServer.cpp
    servers/physics/PhysicsServer.cpp
    servers/audio/AudioServer.cpp
)
target_include_directories(EngineServers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/servers)
target_link_libraries(EngineServers PRIVATE EngineCore EngineRenderer EngineScene)
