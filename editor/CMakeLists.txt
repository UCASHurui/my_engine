# Editor module
cmake_minimum_required(VERSION 3.16)

# ImGui fetch (if not found)
find_package(imgui QUIET)
if(NOT imgui_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.90
    )
    FetchContent_MakeAvailable(imgui)
endif()

add_library(EngineEditor STATIC)

# Editor source files
target_sources(EngineEditor PRIVATE
    editor/EditorApp.cpp
    editor/EditorApp.h
    editor/EditorConfig.h
    editor/PanelManager.cpp
    editor/PanelManager.h
    editor/BasePanel.cpp
    editor/BasePanel.h
    editor/ScenePanel.cpp
    editor/ScenePanel.h
    editor/InspectorPanel.cpp
    editor/InspectorPanel.h
    editor/ImGuiRenderer.cpp
    editor/ImGuiRenderer.h
    editor/ImGuiPlatform.cpp
    editor/ImGuiPlatform.h
)

# ImGui sources (use FetchContent result or system)
if(TARGET imgui)
    target_link_libraries(EngineEditor PRIVATE imgui)
else()
    # Manual include and link for imgui from FetchContent
    target_include_directories(EngineEditor PRIVATE ${CMAKE_BINARY_DIR}/_deps/imgui-src)
    target_sources(EngineEditor PRIVATE
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imgui.cpp
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imgui_draw.cpp
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imgui_tables.cpp
        ${CMAKE_BINARY_DIR}/_deps/imgui-src/imgui_widgets.cpp
    )
endif()

target_include_directories(EngineEditor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/editor
)

target_link_libraries(EngineEditor PRIVATE
    EngineCore
    EngineScene
    EngineRenderer
)

# OpenGL for ImGui rendering
find_package(OpenGL REQUIRED)
target_link_libraries(EngineEditor PRIVATE OpenGL::GL)

# Editor executable
add_executable(my_engine_editor editor_main.cpp)
target_link_libraries(my_engine_editor PRIVATE EngineEditor)
if(TARGET imgui)
    target_link_libraries(my_engine_editor PRIVATE imgui)
endif()
target_include_directories(my_engine_editor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/editor
    ${CMAKE_SOURCE_DIR}/core
)
