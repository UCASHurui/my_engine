# Servers 模块
add_library(EngineServers STATIC)
target_sources(EngineServers PRIVATE
    rendering/RenderingServer.cpp
)
target_include_directories(EngineServers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(EngineServers PRIVATE ${CMAKE_SOURCE_DIR}/renderer)
target_link_libraries(EngineServers PRIVATE EngineCore EngineRenderer EngineScene)
