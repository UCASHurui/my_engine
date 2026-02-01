cmake_minimum_required(VERSION 3.16)
project(MyEngine VERSION 0.1.0 LANGUAGES CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
if(MSVC)
    add_compile_options(/W4 /utf-8)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -std=c++20)
endif()

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 子目录
add_subdirectory(core)
add_subdirectory(platform)
add_subdirectory(renderer)
add_subdirectory(scene)
add_subdirectory(servers)
# add_subdirectory(scripts)  # Lua not installed
add_subdirectory(editor)

# 可执行目标
add_executable(my_engine
    main.cpp
)

# 链接库
target_link_libraries(my_engine PRIVATE
    EngineCore
    EnginePlatform
    EngineRenderer
    EngineScene
    EngineServers
)

# 搜索依赖
find_package(OpenGL REQUIRED)

# 安装配置
include(GNUInstallDirs)
install(TARGETS my_engine
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
