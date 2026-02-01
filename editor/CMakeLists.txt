# Editor 模块
add_library(EngineEditor STATIC)
target_sources(EngineEditor PRIVATE
    editor/EditorApp.cpp
)
target_include_directories(EngineEditor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/editor)
target_link_libraries(EngineEditor PRIVATE EngineCore EngineScene)
