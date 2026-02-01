# Renderer 模块
add_library(EngineRenderer STATIC)
target_sources(EngineRenderer PRIVATE
    renderer/RenderDevice.cpp
    renderer/Shader.cpp
    renderer/Texture.cpp
    renderer/Mesh.cpp
    renderer/StyleRenderer.cpp
)
target_include_directories(EngineRenderer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/renderer)
target_link_libraries(EngineRenderer PRIVATE OpenGL::GL EngineCore EnginePlatform)
