#pragma once

#include "cuda_runtime.h"
#include <vector>
#include <memory>

namespace MyEngine {

// CUDA stream wrapper with RAII
class CUDAStream {
public:
    CUDAStream(unsigned int flags = cudaStreamNonBlocking) {
        cudaError_t err = cudaStreamCreateWithFlags(&_stream, flags);
        if (err != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
            _stream = nullptr;
        }
    }

    ~CUDAStream() {
        if (_stream) {
            cudaStreamDestroy(_stream);
        }
    }

    // Move semantics
    CUDAStream(CUDAStream&& other) noexcept : _stream(other._stream) {
        other._stream = nullptr;
    }

    CUDAStream& operator=(CUDAStream&& other) noexcept {
        if (this != &other) {
            if (_stream) cudaStreamDestroy(_stream);
            _stream = other._stream;
            other._stream = nullptr;
        }
        return *this;
    }

    // Copy disabled
    CUDAStream(const CUDAStream&) = delete;
    CUDAStream& operator=(const CUDAStream&) = delete;

    cudaStream_t get() const { return _stream; }
    operator cudaStream_t() const { return _stream; }
    explicit operator bool() const { return _stream != nullptr; }

    // Wait for all operations to complete
    void synchronize() const {
        if (_stream) {
            cudaStreamSynchronize(_stream);
        }
    }

    // Query if stream is complete
    bool query() const {
        if (!_stream) return true;
        return cudaStreamQuery(_stream) == cudaSuccess;
    }

    // Add callback at stream completion
    void addCallback(cudaStreamCallback_t callback, void* userData) {
        if (_stream) {
            cudaStreamAddCallback(_stream, callback, userData, 0);
        }
    }

private:
    cudaStream_t _stream = nullptr;
};

// Stream pool for multi-stream architecture
class CUDAStreamPool {
public:
    CUDAStreamPool(size_t count = 4) {
        _streams.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            // Create non-blocking streams for overlap
            _streams.emplace_back(cudaStreamNonBlocking);
        }
    }

    ~CUDAStreamPool() = default;

    CUDAStream& getStream(size_t index) {
        return _streams[index % _streams.size()];
    }

    size_t size() const { return _streams.size(); }

    // Synchronize all streams
    void synchronizeAll() {
        for (auto& stream : _streams) {
            stream.synchronize();
        }
    }

    // Wait for a specific stream index
    void waitStream(size_t index) {
        getStream(index).synchronize();
    }

private:
    std::vector<CUDAStream> _streams;
};

// Stream flags enumeration
enum class StreamPriority {
    High = 0,      // Higher priority (lower number)
    Normal = 0,    // Normal priority
    Low = 1        // Lower priority
};

// Create stream with priority
class CUDAPrioritizedStream {
public:
    CUDAPrioritizedStream(StreamPriority priority) {
        int priorityValue = static_cast<int>(priority);
        cudaError_t err = cudaStreamCreateWithPriority(
            &_stream,
            cudaStreamNonBlocking,
            priorityValue
        );
        if (err != cudaSuccess) {
            std::cerr << "Failed to create prioritized stream: "
                      << cudaGetErrorString(err) << std::endl;
            _stream = nullptr;
        }
    }

    ~CUDAPrioritizedStream() {
        if (_stream) {
            cudaStreamDestroy(_stream);
        }
    }

    cudaStream_t get() const { return _stream; }

private:
    cudaStream_t _stream = nullptr;
};

} // namespace MyEngine
