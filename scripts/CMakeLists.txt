# Scripts 模块 (Lua 绑定)
add_library(EngineScripts STATIC)
target_sources(EngineScripts PRIVATE
    scripts/LuaBridge.cpp
    scripts/LuaVM.cpp
)
target_include_directories(EngineScripts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/scripts)
target_link_libraries(EngineScripts PRIVATE EngineCore EngineScene lua)
