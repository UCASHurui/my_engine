# Platform 模块
find_package(OpenGL REQUIRED)
add_library(EnginePlatform STATIC)
target_sources(EnginePlatform PRIVATE
    window/Window.cpp
    window/WindowX11.cpp
    Input.cpp
)
target_include_directories(EnginePlatform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(EnginePlatform PRIVATE OpenGL::GL EngineCore)
