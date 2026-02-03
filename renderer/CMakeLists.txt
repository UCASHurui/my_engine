# Renderer 模块
find_package(OpenGL REQUIRED)
add_library(EngineRenderer STATIC)
target_sources(EngineRenderer PRIVATE
    RenderDevice.cpp
    RenderTarget.cpp
    opengl/RenderDeviceGL.cpp
    postprocessing/PostProcess.cpp
)
target_include_directories(EngineRenderer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(EngineRenderer PRIVATE ${CMAKE_SOURCE_DIR}/scene)
target_link_libraries(EngineRenderer PRIVATE OpenGL::GL EngineCore EnginePlatform)

# CUDA 子模块 (可选)
add_subdirectory(cuda)
