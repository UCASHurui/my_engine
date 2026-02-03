# Core 模块
add_library(EngineCore STATIC)
target_sources(EngineCore PRIVATE
    object/Object.cpp
    object/ClassDB.cpp
    object/RefCounted.cpp
    object/Signal.cpp
    object/Resource.cpp
    variant/Variant.cpp
    math/Vector2.cpp
    math/Vector3.cpp
    math/Matrix4.cpp
    io/ResourceLoader.cpp
    io/FileAccess.cpp
    os/OS.cpp
)
target_include_directories(EngineCore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
