#include "scan.h"
#include "cuda_error.h"

namespace MyEngine {

// Forward declarations for template functions
template<typename Tdata, typename Toffset>
__global__ void addBlockOffsetKernel(
    Tdata* __restrict__ d_data,
    const Toffset* __restrict__ d_offsets,
    size_t n
);

// Block-level inclusive scan (Hillis-Steele)
template<typename T, ScanOp Op>
__device__ T blockScanInclusive(T* sdata, unsigned int tid) {
    // Phase 1: Upsweep
    for (int d = 0; d < 5; d++) {  // 32 elements -> 5 levels
        int stride = 1 << d;
        if (tid >= stride) {
            sdata[tid] = sdata[tid] + sdata[tid - stride];
        }
        __syncthreads();
    }

    // Phase 2: Downsweep
    for (int d = 4; d >= 0; d--) {
        int stride = 1 << d;
        if (tid + stride < blockDim.x) {
            sdata[tid + stride] = sdata[tid + stride] + sdata[tid];
        }
        __syncthreads();
    }

    if (Op == ScanOp::Inclusive) {
        return sdata[tid];
    } else {
        // Exclusive: return previous element's sum
        if (tid == 0) {
            return T(0);
        }
        return sdata[tid - 1];
    }
}

// Scan kernel for a single block
template<typename T, ScanOp Op>
__global__ void scanBlockKernel(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    size_t n
) {
    extern __shared__ float sdata[];
    T* sdata_t = reinterpret_cast<T*>(sdata);

    unsigned int tid = threadIdx.x;
    unsigned int idx = tid;

    // Load data
    if (idx < n) {
        sdata_t[tid] = d_input[idx];
    } else {
        sdata_t[tid] = T(0);
    }
    __syncthreads();

    // Perform scan
    T result = blockScanInclusive<T, Op>(sdata_t, tid);

    // Store result
    if (idx < n) {
        d_output[idx] = result;
    }
}

// Multi-block scan kernel - d_block_sums is int (writable for storing block sums)
template<typename T, ScanOp Op>
__global__ void scanMultiBlockKernel(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    size_t n,
    int* __restrict__ d_block_sums
) {
    extern __shared__ float sdata[];
    T* sdata_t = reinterpret_cast<T*>(sdata);

    unsigned int tid = threadIdx.x;
    unsigned int block_start = blockIdx.x * blockDim.x;
    unsigned int idx = block_start + tid;

    // Load data for this block
    if (idx < n) {
        sdata_t[tid] = d_input[idx];
    } else {
        sdata_t[tid] = T(0);
    }
    __syncthreads();

    // Scan within block
    T block_result = blockScanInclusive<T, ScanOp::Inclusive>(sdata_t, tid);

    // Store block sum
    if (tid == blockDim.x - 1) {
        // This is the last element in the block (or padded)
        unsigned int block_end = block_start + blockDim.x;
        if (block_end > n) block_end = (unsigned int)n;
        if (block_end > 0 && block_start < n) {
            unsigned int last_valid = block_end - 1 - block_start;
            d_block_sums[blockIdx.x] = (int)sdata_t[last_valid];
        } else {
            d_block_sums[blockIdx.x] = 0;
        }
    }
    __syncthreads();

    // Get offset from block sums
    T offset = (blockIdx.x > 0) ? d_block_sums[blockIdx.x - 1] : T(0);

    // Apply offset and do final scan
    if (idx < n) {
        T val = sdata_t[tid];
        T result = (Op == ScanOp::Inclusive) ? (val + offset) : offset;
        d_output[idx] = result;
    }
}

// Compact kernel - scatter valid elements
template<typename T>
__global__ void compactKernel(
    const T* __restrict__ d_input,
    const int* __restrict__ d_flags,
    T* __restrict__ d_output,
    size_t n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    if (d_flags[tid]) {
        // d_flags contains the write position from scan
        d_output[d_flags[tid]] = d_input[tid];
    }
}

// Compact count kernel - count valid elements (no template needed)
__global__ void compactCountKernel(
    const int* __restrict__ d_flags,
    int* __restrict__ d_count,
    size_t n
) {
    extern __shared__ int compact_sdata[];
    int* sdata_i = compact_sdata;

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int val = (idx < n) ? d_flags[idx] : 0;
    sdata_i[tid] = val;
    __syncthreads();

    // Sum reduction for count
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_i[tid] = sdata_i[tid] + sdata_i[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_count[blockIdx.x] = sdata_i[0];
    }
}

// CUDAScan implementation
CUDAScan::CUDAScan()
    : _max_elements(0), _element_size(0), _temp_storage_bytes(0),
      _d_temp_storage(nullptr), _stream(0) {
    cudaStreamCreate(&_stream);
}

CUDAScan::~CUDAScan() {
    if (_d_temp_storage) {
        cudaFree(_d_temp_storage);
    }
    if (_stream) {
        cudaStreamDestroy(_stream);
    }
}

CUDAScan::CUDAScan(CUDAScan&& other) noexcept
    : _max_elements(other._max_elements), _element_size(other._element_size),
      _temp_storage_bytes(other._temp_storage_bytes),
      _d_temp_storage(other._d_temp_storage), _stream(other._stream) {
    other._d_temp_storage = nullptr;
    other._stream = 0;
}

CUDAScan& CUDAScan::operator=(CUDAScan&& other) noexcept {
    if (this != &other) {
        if (_d_temp_storage) cudaFree(_d_temp_storage);
        if (_stream) cudaStreamDestroy(_stream);

        _max_elements = other._max_elements;
        _element_size = other._element_size;
        _temp_storage_bytes = other._temp_storage_bytes;
        _d_temp_storage = other._d_temp_storage;
        _stream = other._stream;

        other._d_temp_storage = nullptr;
        other._stream = 0;
    }
    return *this;
}

void CUDAScan::initialize(size_t max_elements, size_t element_size) {
    _max_elements = max_elements;
    _element_size = element_size;

    // Allocate temp storage for block sums (1 int per block)
    int block_size = 256;
    int max_blocks = (max_elements + block_size - 1) / block_size;
    _temp_storage_bytes = max_blocks * sizeof(int);

    cudaError_t err = cudaMalloc(&_d_temp_storage, _temp_storage_bytes);
    if (err != cudaSuccess) {
        CUDA_LOG_ERROR_WITH_CODE("Failed to allocate scan temp storage", err);
    }
}

template<typename T>
CUDAScan CUDAScan::create(size_t max_elements) {
    CUDAScan scan;
    scan.initialize(max_elements, sizeof(T));
    return scan;
}

template<typename T>
void CUDAScan::executeScan(const T* d_input, T* d_output, size_t count,
                           bool inclusive, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    size_t smem_size = block_size * sizeof(T);
    cudaStream_t use_stream = stream ? stream : _stream;

    if (grid_size == 1) {
        // Single block scan
        if (inclusive) {
            scanBlockKernel<T, ScanOp::Inclusive><<<1, block_size, smem_size, use_stream>>>(
                d_input, d_output, count);
        } else {
            scanBlockKernel<T, ScanOp::Exclusive><<<1, block_size, smem_size, use_stream>>>(
                d_input, d_output, count);
        }
    } else {
        // Multi-block scan
        int* d_block_sums = static_cast<int*>(_d_temp_storage);

        // First pass: scan each block and compute block sums
        if (inclusive) {
            scanMultiBlockKernel<T, ScanOp::Inclusive><<<grid_size, block_size, smem_size, use_stream>>>(
                d_input, d_output, count, d_block_sums);
        } else {
            scanMultiBlockKernel<T, ScanOp::Exclusive><<<grid_size, block_size, smem_size, use_stream>>>(
                d_input, d_output, count, d_block_sums);
        }

        // Scan the block sums
        int block_sums_count = grid_size;
        if (block_sums_count <= block_size) {
            // Single block for block sums
            int* d_scanned_sums;
            cudaMalloc(&d_scanned_sums, block_sums_count * sizeof(int));
            cudaMemcpy(d_scanned_sums, d_block_sums, block_sums_count * sizeof(int), cudaMemcpyDeviceToDevice);

            size_t sum_smem = block_sums_count * sizeof(int);
            if (block_sums_count > 0) {
                scanBlockKernel<int, ScanOp::Inclusive><<<1, block_sums_count, sum_smem, use_stream>>>(
                    d_block_sums, d_scanned_sums, block_sums_count);
            }

            // Add scanned sum as offset to each block (except first)
            dim3 add_grid(grid_size - 1);
            dim3 add_block(block_size);
            addBlockOffsetKernel<<<add_grid, add_block, 0, use_stream>>>(
                d_output + block_size, d_scanned_sums + 1, (grid_size - 1) * block_size);

            cudaFree(d_scanned_sums);
        }

        // Note: Simplified - real implementation would be more complex
        // For production, use thrust orcub
    }
}

template<typename T>
ScanResult<T> CUDAScan::scanInclusive(const T* d_input, size_t count, cudaStream_t stream) {
    ScanResult<T> result;
    result.count = count;

    cudaError_t err = cudaMalloc(&result.d_output, count * sizeof(T));
    if (err != cudaSuccess) {
        result.error = err;
        return result;
    }

    // Time the operation
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);

    executeScan(d_input, result.d_output, count, true, stream);

    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&result.execution_time_ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    result.error = cudaPeekAtLastError();
    return result;
}

template<typename T>
ScanResult<T> CUDAScan::scanExclusive(const T* d_input, size_t count, cudaStream_t stream) {
    ScanResult<T> result;
    result.count = count;

    cudaError_t err = cudaMalloc(&result.d_output, count * sizeof(T));
    if (err != cudaSuccess) {
        result.error = err;
        return result;
    }

    // Time the operation
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);

    executeScan(d_input, result.d_output, count, false, stream);

    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&result.execution_time_ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    result.error = cudaPeekAtLastError();
    return result;
}

template<typename T>
ScanResult<T> CUDAScan::compact(const T* d_input, const int* d_predicate,
                                 size_t count, cudaStream_t stream) {
    ScanResult<T> result;
    result.count = count;

    cudaStream_t use_stream = stream ? stream : _stream;

    // First, scan the predicate to get write positions
    int* d_scan_output;
    cudaError_t err = cudaMalloc(&d_scan_output, count * sizeof(int));
    if (err != cudaSuccess) {
        result.error = err;
        return result;
    }

    CUDAScan scan_int = CUDAScan::create<int>(count);
    auto scan_result = scan_int.scanInclusive(d_predicate, count, use_stream);

    if (!scan_result.ok()) {
        cudaFree(d_scan_output);
        result.error = scan_result.error;
        return result;
    }

    // Copy scan result to our buffer
    cudaMemcpy(d_scan_output, scan_result.d_output, count * sizeof(int), cudaMemcpyDeviceToDevice);

    // Count total valid elements (last element of scan)
    int total_count = 0;
    if (count > 0) {
        cudaMemcpy(&total_count, scan_result.d_output + count - 1, sizeof(int), cudaMemcpyDeviceToDevice);
        // Add last element value if it was valid
        int last_flag = 0;
        cudaMemcpy(&last_flag, d_predicate + count - 1, sizeof(int), cudaMemcpyDeviceToDevice);
        total_count += last_flag;
    }

    // Allocate output buffer
    err = cudaMalloc(&result.d_output, total_count * sizeof(T));
    if (err != cudaSuccess) {
        cudaFree(d_scan_output);
        result.error = err;
        return result;
    }

    // Scatter valid elements
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    compactKernel<T><<<grid, block, 0, use_stream>>>(
        d_input, d_scan_output, result.d_output, count);

    cudaFree(d_scan_output);

    result.count = total_count;
    result.error = cudaPeekAtLastError();
    return result;
}

// Add block offset kernel
template<typename Tdata, typename Toffset>
__global__ void addBlockOffsetKernel(
    Tdata* __restrict__ d_data,
    const Toffset* __restrict__ d_offsets,
    size_t n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    d_data[tid] = d_data[tid] + (Tdata)d_offsets[blockIdx.x];
}

// Explicit instantiations
template CUDAScan CUDAScan::create<float>(size_t);
template CUDAScan CUDAScan::create<int>(size_t);
template CUDAScan CUDAScan::create<double>(size_t);

template ScanResult<float> CUDAScan::scanInclusive(const float*, size_t, cudaStream_t);
template ScanResult<float> CUDAScan::scanExclusive(const float*, size_t, cudaStream_t);
template ScanResult<float> CUDAScan::compact(const float*, const int*, size_t, cudaStream_t);

template ScanResult<int> CUDAScan::scanInclusive(const int*, size_t, cudaStream_t);
template ScanResult<int> CUDAScan::scanExclusive(const int*, size_t, cudaStream_t);
template ScanResult<int> CUDAScan::compact(const int*, const int*, size_t, cudaStream_t);

// Forward declaration
__global__ void createPredicateKernel(
    const float* __restrict__ d_values,
    int* __restrict__ d_predicate,
    size_t n,
    float threshold,
    bool keep_greater
);

// Utility function implementation
inline int* createPredicateFromCondition(
    const float* d_values,
    float threshold,
    size_t count,
    bool keep_greater,
    cudaStream_t stream
) {
    int* d_predicate;
    cudaMalloc(&d_predicate, count * sizeof(int));

    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);

    createPredicateKernel<<<grid, block, 0, stream>>>(
        d_values, d_predicate, count, threshold, keep_greater);

    return d_predicate;
}

__global__ void createPredicateKernel(
    const float* __restrict__ d_values,
    int* __restrict__ d_predicate,
    size_t n,
    float threshold,
    bool keep_greater
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    if (keep_greater) {
        d_predicate[tid] = (d_values[tid] > threshold) ? 1 : 0;
    } else {
        d_predicate[tid] = (d_values[tid] <= threshold) ? 1 : 0;
    }
}

} // namespace MyEngine
