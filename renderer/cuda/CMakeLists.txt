# CUDA 子模块
find_package(CUDA REQUIRED)

# 启用 CUDA 编译选项
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Auto)
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
list(APPEND CUDA_NVCC_FLAGS --use_fast_math -O3)

# 包含目录
target_include_directories(EngineRenderer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# CUDA 源文件列表
set(CUDA_SOURCES
    cuda_runtime_engine.cu
    rt/rt_core.cu
    compute/particle_system.cu
    compute/post_processing.cu
)

# Ray tracing headers
set(RT_HEADERS
    rt/ray_types.h
    rt/bvh_builder.h
    rt/rt_core.h
)

# Compute headers
set(COMPUTE_HEADERS
    compute/particle_system.h
    compute/post_processing.h
)

# 创建 CUDA 库对象（链接到 EngineRenderer）
add_library(EngineCUDA OBJECT ${CUDA_SOURCES} ${RT_HEADERS} ${COMPUTE_HEADERS})
target_link_libraries(EngineCUDA PRIVATE EngineCore CUDA::cuda_driver)
set_target_properties(EngineCUDA PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# CUDA 测试程序
add_executable(test_cuda test_cuda.cpp)
target_include_directories(test_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_cuda PRIVATE ${CUDA_LIBRARIES})
