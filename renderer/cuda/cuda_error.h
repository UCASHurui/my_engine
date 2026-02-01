#pragma once

#include "cuda_runtime_engine.h"
#include <string>
#include <sstream>
#include <iostream>

namespace MyEngine {

// Error category for CUDA errors
struct CUDAErrorCategory : std::error_category {
    const char* name() const noexcept override { return "CUDA"; }

    std::string message(int ev) const override {
        cudaError_t err = static_cast<cudaError_t>(ev);
        return cudaGetErrorString(err);
    }
};

// Global error category instance
inline const CUDAErrorCategory& cuda_error_category() {
    static CUDAErrorCategory instance;
    return instance;
}

// Create std::error_code from cudaError_t
inline std::error_code make_cuda_error_code(cudaError_t err) {
    return std::error_code(static_cast<int>(err), cuda_error_category());
}

// Result type for CUDA operations
template<typename T>
struct CUDAResult {
    T value;
    cudaError_t error;

    CUDAResult(T v, cudaError_t e) : value(v), error(e) {}

    bool ok() const { return error == cudaSuccess; }
    explicit operator bool() const { return ok(); }

    T or_default(T default_value) const {
        return ok() ? value : default_value;
    }
};

// Specialization for void
template<>
struct CUDAResult<void> {
    cudaError_t error;

    CUDAResult(cudaError_t e) : error(e) {}

    bool ok() const { return error == cudaSuccess; }
    explicit operator bool() const { return ok(); }
};

// Check macro with error handling
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

// Check but don't throw (returns false on error)
#define CUDA_VERIFY(call) \
    ([&]() -> bool { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
        return true; \
    }())

// Check last error after kernel launch
inline bool checkLastError(const char* kernel_name = nullptr) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error";
        if (kernel_name) std::cerr << " in " << kernel_name;
        std::cerr << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

// Get error string helper
inline const char* getErrorString(cudaError_t err) {
    return cudaGetErrorString(err);
}

// Check device availability
inline bool isDeviceAvailable() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return err == cudaSuccess && deviceCount > 0;
}

// Get last CUDA error as string
inline std::string getLastErrorString() {
    cudaError_t err = cudaGetLastError();
    return std::string(cudaGetErrorString(err));
}

// Error logger for CUDA
class CUDAErrorLogger {
public:
    static void log(cudaError_t err, const char* file, int line, const char* func = nullptr) {
        std::ostringstream oss;
        oss << "CUDA Error [" << cudaGetErrorString(err) << "]"
            << " at " << file << ":" << line;
        if (func) oss << " in " << func;
        std::cerr << oss.str() << std::endl;
    }

    static void logMessage(const char* message, cudaError_t err = cudaSuccess) {
        std::ostringstream oss;
        oss << message;
        if (err != cudaSuccess) {
            oss << ": " << cudaGetErrorString(err);
        }
        std::cerr << oss.str() << std::endl;
    }
};

// Convenience macro for logging errors
#define CUDA_LOG_ERROR(msg) \
    MyEngine::CUDAErrorLogger::logMessage(msg)

#define CUDA_LOG_ERROR_WITH_CODE(msg, code) \
    MyEngine::CUDAErrorLogger::log(code, __FILE__, __LINE__)

// Peek at current synchronization state
inline cudaError_t peekAtSynchronizedState() {
    return cudaPeekAtLastError();
}

} // namespace MyEngine
