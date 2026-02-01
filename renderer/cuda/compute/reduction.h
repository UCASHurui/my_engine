#pragma once

#include "cuda_runtime.h"
#include <functional>

namespace MyEngine {

// Reduction operation types
enum class ReductionOp {
    Sum,
    Min,
    Max,
    Average
};

// Result structure for reduction
template<typename T>
struct ReductionResult {
    T value;
    cudaError_t error;
    float execution_time_ms;

    ReductionResult() : value(T{}), error(cudaSuccess), execution_time_ms(0.0f) {}
    ReductionResult(T v, cudaError_t e, float t) : value(v), error(e), execution_time_ms(t) {}

    bool ok() const { return error == cudaSuccess; }
    explicit operator bool() const { return ok(); }
};

// Parallel reduction manager class
template<typename T>
class CUDAReduction {
public:
    CUDAReduction(ReductionOp operation, size_t max_elements);
    ~CUDAReduction();

    // Move semantics
    CUDAReduction(CUDAReduction&& other) noexcept;
    CUDAReduction& operator=(CUDAReduction&& other) noexcept;

    // Copy disabled
    CUDAReduction(const CUDAReduction&) = delete;
    CUDAReduction& operator=(const CUDAReduction&) = delete;

    // Execute reduction (async)
    cudaError_t reduceAsync(
        const T* d_input,
        T* d_output,
        size_t count,
        cudaStream_t stream = 0
    );

    // Execute reduction (synchronous)
    ReductionResult<T> reduce(
        const T* d_input,
        size_t count,
        cudaStream_t stream = 0
    );

    // Get optimal block count for given data size
    static int getOptimalBlockCount(size_t count, int block_size = 256);

private:
    ReductionOp _operation;
    size_t _max_elements;
    cudaStream_t _stream;
    T* _d_temp_storage;
    size_t _temp_storage_bytes;

    void allocateStorage(size_t max_elements);
    void freeStorage();

    template<typename U>
    __device__ U reduceOp(U a, U b);
};

// Convenience functions for simple reductions (inline implementations)
// These require the full class definition, so they're in reduction.cu
template<typename T>
ReductionResult<T> reduceSum(
    const T* d_input,
    size_t count,
    cudaStream_t stream = 0
);

template<typename T>
ReductionResult<T> reduceMin(
    const T* d_input,
    size_t count,
    cudaStream_t stream = 0
);

template<typename T>
ReductionResult<T> reduceMax(
    const T* d_input,
    size_t count,
    cudaStream_t stream = 0
);

template<typename T>
ReductionResult<T> reduceAverage(
    const T* d_input,
    size_t count,
    cudaStream_t stream = 0
);

} // namespace MyEngine
