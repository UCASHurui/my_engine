# CUDA 子模块
find_package(CUDA REQUIRED)

# OpenGL required for interop
find_package(OpenGL REQUIRED)

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
    compute/reduction.cu
    compute/scan.cu
    interop/cuda_texture.cu
    interop/cuda_gl_interop.cpp
    profiler/cuda_profiler.cpp
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
    compute/reduction.h
    compute/scan.h
)

# Profiler headers
set(PROFILER_HEADERS
    profiler/cuda_profiler.h
)

# Interop headers
set(INTEROP_HEADERS
    interop/cuda_texture.h
    interop/cuda_gl_interop.h
)

# 创建 CUDA 库对象（链接到 EngineRenderer）
add_library(EngineCUDA OBJECT ${CUDA_SOURCES} ${RT_HEADERS} ${COMPUTE_HEADERS} ${PROFILER_HEADERS} ${INTEROP_HEADERS})
target_link_libraries(EngineCUDA PRIVATE EngineCore CUDA::cuda_driver OpenGL::GL)
set_target_properties(EngineCUDA PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# CUDA 测试程序
add_executable(test_cuda test_cuda.cpp)
target_include_directories(test_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_cuda PRIVATE ${CUDA_LIBRARIES} OpenGL::GL)

# Interop 测试程序
add_executable(test_interop test_interop.cu)
target_include_directories(test_interop PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_interop PRIVATE ${CUDA_LIBRARIES} OpenGL::GL)

# Primitives 测试程序
add_executable(test_primitives test_primitives.cu)
target_include_directories(test_primitives PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_primitives PRIVATE ${CUDA_LIBRARIES} EngineCUDA)
