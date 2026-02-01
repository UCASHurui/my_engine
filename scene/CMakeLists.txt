# Scene 模块
add_library(EngineScene STATIC)
target_sources(EngineScene PRIVATE
    scene/main/Node.cpp
    scene/main/SceneTree.cpp
    scene/main/Viewport.cpp
    scene/2d/Node2D.cpp
    scene/2d/Sprite2D.cpp
    scene/3d/Node3D.cpp
    scene/3d/MeshInstance3D.cpp
)
target_include_directories(EngineScene PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/scene)
target_link_libraries(EngineScene PRIVATE EngineCore EngineRenderer)
