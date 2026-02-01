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
    cuda_runtime.cu
)

# 创建 CUDA 库对象（链接到 EngineRenderer）
add_library(EngineCUDA OBJECT ${CUDA_SOURCES})
target_link_libraries(EngineCUDA PRIVATE EngineCore CUDA::cuda_driver)
set_target_properties(EngineCUDA PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# 可选：CUDA 测试程序
if(BUILD_TESTING)
    add_executable(test_cuda test_cuda.cpp)
    target_link_libraries(test_cuda PRIVATE EngineRenderer)
endif()
