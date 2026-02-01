#include "cuda_runtime_engine.h"
#include "cuda_error.h"
#include <iostream>

namespace MyEngine {

// Device query kernel
__global__ void queryDeviceKernel(int* deviceInfo) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        deviceInfo[0] = 1;
    }
}

// GPU info string
std::string CUDARuntime::getDeviceInfoString() {
    if (!isInitialized()) {
        return "CUDA not initialized";
    }

    std::ostringstream oss;
    auto& prop = getDeviceProperties();

    oss << "Device: " << prop.name << "\n";
    oss << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    oss << "Multiprocessors: " << prop.multiProcessorCount << "\n";
    oss << "Threads per block: " << prop.maxThreadsPerBlock << "\n";
    oss << "Threads per MP: " << prop.maxThreadsPerMultiProcessor << "\n";
    oss << "Registers per block: " << prop.regsPerBlock << "\n";
    oss << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
    oss << "Total global memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
    oss << "Warp size: " << prop.warpSize << "\n";
    oss << "ECC enabled: " << (prop.ECCEnabled ? "Yes" : "No") << "\n";
    oss << "Managed memory: " << (prop.managedMemory ? "Yes" : "No") << "\n";
    oss << "Concurrent kernels: " << (prop.concurrentKernels ? "Yes" : "No") << "\n";

    return oss.str();
}

// Advanced device selection
int CUDARuntime::selectOptimalDevice(int workload_type) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) return -1;

    int bestDevice = 0;
    float bestScore = 0.0f;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        float score = 0.0f;

        switch (workload_type) {
            case WORKLOAD_RENDERING:
                score = prop.totalGlobalMem * 0.4f + prop.multiProcessorCount * 1000.0f;
                break;
            case WORKLOAD_COMPUTE:
                score = prop.multiProcessorCount * 2000.0f + prop.maxThreadsPerMultiProcessor * 0.1f;
                break;
            case WORKLOAD_MEMORY:
                score = prop.totalGlobalMem * 0.6f + prop.memoryBusWidth * 0.5f;
                break;
            default:
                score = prop.multiProcessorCount * 1000.0f;
        }

        if (score > bestScore) {
            bestScore = score;
            bestDevice = i;
        }
    }

    return bestDevice;
}

} // namespace MyEngine
