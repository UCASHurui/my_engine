# Scene 模块
add_library(EngineScene STATIC)
target_sources(EngineScene PRIVATE
    main/Node.cpp
    main/SceneTree.cpp
    2d/Node2D.cpp
    2d/Sprite2D.cpp
    3d/Node3D.cpp
    3d/MeshInstance3D.cpp
)
target_include_directories(EngineScene PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(EngineScene PRIVATE ${CMAKE_SOURCE_DIR}/renderer)
target_link_libraries(EngineScene PRIVATE EngineCore EngineRenderer)
