#pragma once

#include "cuda_runtime.h"
#include <cstddef>
#include <vector>
#include <memory>

namespace MyEngine {

// Memory type enumeration
enum class CUDAMemoryType {
    Device,    // Device-only memory (fastest)
    Host,      // Pinned host memory
    Managed    // Unified memory (accessible by CPU and GPU)
};

// RAII wrapper for CUDA memory allocations
class CUDAMemory {
public:
    CUDAMemory() : _ptr(nullptr), _size(0), _type(CUDAMemoryType::Device) {}

    CUDAMemory(void* ptr, size_t size, CUDAMemoryType type)
        : _ptr(ptr), _size(size), _type(type) {}

    ~CUDAMemory() {
        if (_ptr) {
            free();
        }
    }

    // Move semantics
    CUDAMemory(CUDAMemory&& other) noexcept
        : _ptr(other._ptr), _size(other._size), _type(other._type) {
        other._ptr = nullptr;
        other._size = 0;
    }

    CUDAMemory& operator=(CUDAMemory&& other) noexcept {
        if (this != &other) {
            free();
            _ptr = other._ptr;
            _size = other._size;
            _type = other._type;
            other._ptr = nullptr;
            other._size = 0;
        }
        return *this;
    }

    // Copy disabled
    CUDAMemory(const CUDAMemory&) = delete;
    CUDAMemory& operator=(const CUDAMemory&) = delete;

    void* get() const { return _ptr; }
    size_t size() const { return _size; }
    CUDAMemoryType type() const { return _type; }

    explicit operator bool() const { return _ptr != nullptr; }

private:
    void free() {
        if (_type == CUDAMemoryType::Managed) {
            cudaFree(_ptr);
        } else if (_type == CUDAMemoryType::Host) {
            cudaFreeHost(_ptr);
        } else {
            cudaFree(_ptr);
        }
    }

    void* _ptr;
    size_t _size;
    CUDAMemoryType _type;
};

// High-level CUDA memory manager
class CUDAMemoryManager {
public:
    // Allocate device memory
    static CUDAMemory allocDevice(size_t size) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
            return CUDAMemory();
        }
        return CUDAMemory(ptr, size, CUDAMemoryType::Device);
    }

    // Allocate pinned host memory (fast async copy)
    static CUDAMemory allocHost(size_t size) {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, size);
        if (err != cudaSuccess) {
            std::cerr << "CUDA mallocHost failed: " << cudaGetErrorString(err) << std::endl;
            return CUDAMemory();
        }
        return CUDAMemory(ptr, size, CUDAMemoryType::Host);
    }

    // Allocate managed memory (CPU/GPU accessible)
    static CUDAMemory allocManaged(size_t size, cudaMemoryAdvise advise = cudaMemAdviseSetPreferredLocation) {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocManaged(&ptr, size);
        if (err != cudaSuccess) {
            std::cerr << "CUDA mallocManaged failed: " << cudaGetErrorString(err) << std::endl;
            return CUDAMemory();
        }

        // Set memory advice for optimal access
        cudaMemAdvise(ptr, size, advise, CUDARuntime::getDevice());

        return CUDAMemory(ptr, size, CUDAMemoryType::Managed);
    }

    // Prefetch managed memory to device
    static void prefetchToDevice(const void* ptr, size_t size) {
        cudaMemPrefetchAsync(ptr, size, CUDARuntime::getDevice(), 0);
    }

    // Prefetch managed memory to host
    static void prefetchToHost(const void* ptr, size_t size) {
        cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, 0);
    }

    // Synchronous memory copy
    static void memcpy(void* dst, const void* src, size_t size,
                       cudaMemcpyKind kind = cudaMemcpyDefault) {
        cudaError_t err = cudaMemcpy(dst, src, size, kind);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        }
    }

    // Asynchronous memory copy
    static void memcpyAsync(void* dst, const void* src, size_t size,
                            cudaStream_t stream = 0,
                            cudaMemcpyKind kind = cudaMemcpyDefault) {
        cudaError_t err = cudaMemcpyAsync(dst, src, size, kind, stream);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
        }
    }

    // Memory set
    static void memset(void* ptr, int value, size_t size) {
        cudaError_t err = cudaMemset(ptr, value, size);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memset failed: " << cudaGetErrorString(err) << std::endl;
        }
    }

    // Asynchronous memset
    static void memsetAsync(void* ptr, int value, size_t size, cudaStream_t stream = 0) {
        cudaError_t err = cudaMemsetAsync(ptr, value, size, stream);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memsetAsync failed: " << cudaGetErrorString(err) << std::endl;
        }
    }

    // Get memory info
    static void getMemoryInfo(size_t& free_bytes, size_t& total_bytes) {
        cudaMemGetInfo(&free_bytes, &total_bytes);
    }

    static size_t getAvailableMemory() {
        size_t free_bytes, total_bytes;
        getMemoryInfo(free_bytes, total_bytes);
        return free_bytes;
    }

    static size_t getTotalMemory() {
        size_t free_bytes, total_bytes;
        getMemoryInfo(free_bytes, total_bytes);
        return total_bytes;
    }

    // Memory pool for frequent small allocations
    template<typename T>
    static std::vector<T> allocVector(size_t count, CUDAMemoryType type = CUDAMemoryType::Device) {
        std::vector<T> result(count);
        CUDAMemory mem = (type == CUDAMemoryType::Host) ?
            allocHost(count * sizeof(T)) :
            allocDevice(count * sizeof(T));

        if (mem) {
            memcpy(result.data(), mem.get(), count * sizeof(T), cudaMemcpyHostToHost);
        }

        return result;
    }
};

} // namespace MyEngine
