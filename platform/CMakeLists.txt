# Platform 模块
add_library(EnginePlatform STATIC)
target_sources(EnginePlatform PRIVATE
    platform/window/Window.cpp
    platform/graphics/OpenGL/GLDevice.cpp
    platform/Input.cpp
)
target_include_directories(EnginePlatform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/platform)
target_link_libraries(EnginePlatform PRIVATE OpenGL::GL EngineCore)
