#pragma once

#include "cuda_runtime.h"

namespace MyEngine {

// Scan operation types
enum class ScanOp {
    Inclusive,
    Exclusive
};

// Scan result structure
template<typename T>
struct ScanResult {
    T* d_output;
    size_t count;
    cudaError_t error;
    float execution_time_ms;

    ScanResult() : d_output(nullptr), count(0), error(cudaSuccess), execution_time_ms(0.0f) {}
    ~ScanResult() { if (d_output) cudaFree(d_output); }

    // Move semantics
    ScanResult(ScanResult&& other) noexcept;
    ScanResult& operator=(ScanResult&& other) noexcept;

    // Copy disabled
    ScanResult(const ScanResult&) = delete;
    ScanResult& operator=(const ScanResult&) = delete;

    T operator[](size_t idx) const { return d_output[idx]; }
    bool ok() const { return error == cudaSuccess; }
    explicit operator bool() const { return ok(); }
};

// Parallel scan manager class
class CUDAScan {
public:
    // Create scan manager for given max elements
    template<typename T>
    static CUDAScan create(size_t max_elements);

    ~CUDAScan();

    // Move semantics
    CUDAScan(CUDAScan&& other) noexcept;
    CUDAScan& operator=(CUDAScan&& other) noexcept;

    // Copy disabled
    CUDAScan(const CUDAScan&) = delete;
    CUDAScan& operator=(const CUDAScan&) = delete;

    // Inclusive prefix sum: output[i] = sum(input[0..i])
    template<typename T>
    ScanResult<T> scanInclusive(
        const T* d_input,
        size_t count,
        cudaStream_t stream = 0
    );

    // Exclusive prefix sum: output[i] = sum(input[0..i-1])
    template<typename T>
    ScanResult<T> scanExclusive(
        const T* d_input,
        size_t count,
        cudaStream_t stream = 0
    );

    // Compact operation: scatter valid elements based on predicate
    // d_predicate[i] = 1 means keep, 0 means discard
    template<typename T>
    ScanResult<T> compact(
        const T* d_input,
        const int* d_predicate,
        size_t count,
        cudaStream_t stream = 0
    );

    // Get temp storage size needed
    size_t getTempStorageSize() const { return _temp_storage_bytes; }

private:
    CUDAScan();

    void initialize(size_t max_elements, size_t element_size);

    size_t _max_elements;
    size_t _element_size;
    size_t _temp_storage_bytes;
    void* _d_temp_storage;
    cudaStream_t _stream;

    template<typename T>
    void executeScan(const T* d_input, T* d_output, size_t count,
                     bool inclusive, cudaStream_t stream);
};

// Utility function for creating predicate array from condition
inline int* createPredicateFromCondition(
    const float* d_values,
    float threshold,
    size_t count,
    bool keep_greater = true,
    cudaStream_t stream = 0
);

} // namespace MyEngine
