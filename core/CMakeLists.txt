# Core 模块
add_library(EngineCore STATIC)
target_sources(EngineCore PRIVATE
    core/object/Object.cpp
    core/object/ClassDB.cpp
    core/object/RefCounted.cpp
    core/object/Signal.cpp
    core/variant/Variant.cpp
    core/math/Vector2.cpp
    core/math/Vector3.cpp
    core/math/Matrix4.cpp
    core/io/ResourceLoader.cpp
    core/io/FileAccess.cpp
    core/os/OS.cpp
)
target_include_directories(EngineCore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
