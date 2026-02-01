#!/bin/bash
# 独立 CUDA 测试编译脚本

cd /home/douba/Projects/my_engine

# 检查 CUDA
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found. Install nvidia-cuda-toolkit first."
    exit 1
fi

echo "Compiling CUDA test with nvcc..."

nvcc -o cuda_test_bin \
    test_cuda.cpp \
    renderer/cuda/cuda_runtime.cu \
    -I. \
    -dc \
    -O3 \
    --use_fast_math \
    -arch=sm_89

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Run with: ./cuda_test_bin"
else
    echo "Compilation failed!"
    exit 1
fi
