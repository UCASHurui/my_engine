#pragma once

#include "cuda_runtime.h"
#include <string>
#include <sstream>
#include <iostream>

namespace MyEngine {

// Workload types for device selection
enum WorkloadType {
    WORKLOAD_RENDERING = 0,   // Prioritize VRAM and compute
    WORKLOAD_COMPUTE = 1,      // Prioritize compute power
    WORKLOAD_MEMORY = 2        // Prioritize memory bandwidth
};

// CUDA initialization and management
class CUDARuntime {
public:
    static bool initialize() {
        if (_initialized) return true;

        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);

        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        if (deviceCount == 0) {
            std::cerr << "No CUDA-capable devices found" << std::endl;
            return false;
        }

        // Select best device
        int bestDevice = 0;
        int maxMultiprocessors = 0;

        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            if (prop.multiProcessorCount > maxMultiprocessors) {
                maxMultiprocessors = prop.multiProcessorCount;
                bestDevice = i;
            }
        }

        err = cudaSetDevice(bestDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        _device = bestDevice;
        _initialized = true;

        std::cout << "CUDA initialized: Device " << bestDevice << std::endl;
        return true;
    }

    static void shutdown() {
        if (_initialized) {
            cudaDeviceSynchronize();
            cudaError_t err = cudaDeviceReset();
            if (err != cudaSuccess) {
                std::cerr << "CUDA shutdown error: " << cudaGetErrorString(err) << std::endl;
            }
            _initialized = false;
        }
    }

    static bool isInitialized() { return _initialized; }
    static int getDevice() { return _device; }
    static int getDeviceCount() {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }

    static const cudaDeviceProp& getDeviceProperties() {
        static cudaDeviceProp prop;
        static bool cached = false;
        if (!cached) {
            cudaGetDeviceProperties(&prop, _device);
            cached = true;
        }
        return prop;
    }

    static size_t getTotalMemory() {
        return getDeviceProperties().totalGlobalMem;
    }

    static size_t getAvailableMemory() {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        return free;
    }

    static bool checkError(cudaError_t err, const char* message = nullptr) {
        if (err == cudaSuccess) return true;

        std::cerr << "CUDA Error";
        if (message) std::cerr << " (" << message << ")";
        std::cerr << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    static void checkLastError(const char* message = nullptr) {
        cudaError_t err = cudaGetLastError();
        checkError(err, message);
    }

    static std::string getDeviceName() {
        return std::string(getDeviceProperties().name);
    }

    static int getComputeCapability() {
        auto& prop = getDeviceProperties();
        return prop.major * 10 + prop.minor;
    }

    static bool supportsRTX() {
        return getComputeCapability() >= 80; // Ampere or newer
    }

    static bool supportsManagedMemory() {
        auto& prop = getDeviceProperties();
        return prop.managedMemory == 1;
    }

    static std::string getDeviceInfoString();
    static int selectOptimalDevice(int workload_type = WORKLOAD_RENDERING);

private:
    static bool _initialized;
    static int _device;
};

inline bool CUDARuntime::_initialized = false;
inline int CUDARuntime::_device = 0;

} // namespace MyEngine
