#include "scan.h"
#include "cuda_error.h"

namespace MyEngine {

// Forward declarations for template functions
template<typename T>
__global__ void scanBlockKernel(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    size_t n
);

template<typename T>
__global__ void scanMultiBlockKernel(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    size_t n,
    T* __restrict__ d_block_sums  // Output: block sums
);

template<typename T>
__global__ void addBlockOffsetKernel(
    T* __restrict__ d_data,
    const T* __restrict__ d_block_offsets,
    size_t n
);

template<typename T>
__global__ void shiftExclusiveKernel(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    size_t n
);

// Block-level inclusive scan (single-phase Hillis-Steele)
// Fixed: explicit read/write phases to prevent compiler optimization issues
template<typename T, ScanOp Op>
__device__ T blockScanInclusive(T* sdata, unsigned int tid, unsigned int blockSize) {
    // Single-phase Hillis-Steele scan with explicit synchronization
    for (int stride = 1; stride < blockSize; stride *= 2) {
        // Read phase - store values in registers
        T val = sdata[tid];
        T other = (tid >= stride) ? sdata[tid - stride] : T(0);
        __syncthreads();
        // Write phase
        if (tid >= stride) {
            sdata[tid] = val + other;
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
    unsigned int blockSize = blockDim.x;

    // Load data
    if (tid < n) {
        sdata_t[tid] = d_input[tid];
    } else {
        sdata_t[tid] = T(0);
    }
    __syncthreads();

    // Perform scan
    T result = blockScanInclusive<T, Op>(sdata_t, tid, blockSize);

    // Store result
    if (tid < n) {
        d_output[tid] = result;
    }
}

// Multi-block scan kernel - computes block scan and stores block sums
template<typename T>
__global__ void scanMultiBlockKernel(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    size_t n,
    T* __restrict__ d_block_sums
) {
    extern __shared__ float sdata[];
    T* sdata_t = reinterpret_cast<T*>(sdata);

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int block_start = blockIdx.x * blockSize;
    unsigned int idx = block_start + tid;

    // Calculate valid count first
    unsigned int valid_count = (n > block_start) ? min((size_t)blockSize, n - block_start) : 0;

    // Compute block sum using thread 0 - sum of INPUT values
    if (tid == 0 && valid_count > 0) {
        T sum = T(0);
        for (unsigned int i = 0; i < valid_count; i++) {
            sum = sum + d_input[block_start + i];
        }
        d_block_sums[blockIdx.x] = sum;
    }
    __syncthreads();

    // Load data for this block
    if (idx < n) {
        sdata_t[tid] = d_input[idx];
    } else {
        sdata_t[tid] = T(0);
    }
    __syncthreads();

    // Scan within block (inclusive)
    T result = blockScanInclusive<T, ScanOp::Inclusive>(sdata_t, tid, blockSize);

    // Store result
    if (idx < n) {
        d_output[idx] = result;
    }
}

// Add block offset kernel - adds the scanned block sum to each element
template<typename T>
__global__ void addBlockOffsetKernel(
    T* __restrict__ d_data,
    const T* __restrict__ d_block_offsets,
    size_t n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Block 0 has no offset, block i has offset d_block_offsets[i-1]
    T offset = (blockIdx.x > 0) ? d_block_offsets[blockIdx.x - 1] : T(0);
    d_data[tid] = d_data[tid] + offset;
}

// Compact kernel - scatter valid elements
// For inclusive scan, we need to subtract 1 from the scan result to get 0-indexed positions
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
        // d_flags contains the write position from scan (inclusive)
        // Subtract 1 to get 0-indexed position
        d_output[d_flags[tid] - 1] = d_input[tid];
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

    // Allocate temp storage for block sums (1 element per block)
    int block_size = 256;
    int max_blocks = (max_elements + block_size - 1) / block_size;

    // Need storage for: block sums array + possibly intermediate arrays
    // Use float as base type for simplicity
    _temp_storage_bytes = max_blocks * sizeof(float) * 2;  // Extra space for scanned sums

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
        // Multi-block scan - three pass algorithm
        T* d_block_sums = static_cast<T*>(_d_temp_storage);
        T* d_scanned_sums = d_block_sums + grid_size;

        // Pass 1: Scan each block and compute block sums
        if (inclusive) {
            scanMultiBlockKernel<T><<<grid_size, block_size, smem_size, use_stream>>>(
                d_input, d_output, count, d_block_sums);
        } else {
            scanMultiBlockKernel<T><<<grid_size, block_size, smem_size, use_stream>>>(
                d_input, d_output, count, d_block_sums);
        }
        CUDA_CHECK(cudaStreamSynchronize(use_stream));

        // Pass 2: Scan the block sums array
        size_t sum_smem = grid_size * sizeof(T);
        if (grid_size <= block_size) {
            // Single block can scan all block sums
            scanBlockKernel<T, ScanOp::Inclusive><<<1, grid_size, sum_smem, use_stream>>>(
                d_block_sums, d_scanned_sums, grid_size);
        } else {
            // Recursive case: scan block sums in batches
            int num_batches = (grid_size + block_size - 1) / block_size;
            // For simplicity, handle moderate sizes with multiple passes
            // In production, use thrust or CUB for this
            for (int batch = 0; batch < num_batches; batch++) {
                int batch_start = batch * block_size;
                int batch_count = min(block_size, grid_size - batch_start);
                if (batch_count > 0) {
                    scanBlockKernel<T, ScanOp::Inclusive><<<1, batch_count, batch_count * sizeof(T), use_stream>>>(
                        d_block_sums + batch_start, d_scanned_sums + batch_start, batch_count);
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(use_stream));

        // Pass 3: Add block offsets to each element
        dim3 add_grid(grid_size);
        dim3 add_block(block_size);
        addBlockOffsetKernel<T><<<add_grid, add_block, 0, use_stream>>>(
            d_output, d_scanned_sums, count);

        // Handle exclusive scan by shifting
        if (!inclusive) {
            // Shift left by one: output[i] = input[i-1] (or 0 for i=0)
            // We need to move data: output[i] = output[i-1] for i >= 1
            // Use a simple kernel for this
            // For now, just note that exclusive isn't fully implemented
        }
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

    // For exclusive, we scan inclusive first then shift
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);

    // First do an inclusive scan to temp buffer
    T* d_temp;
    cudaMalloc(&d_temp, count * sizeof(T));
    executeScan(d_input, d_temp, count, true, stream);

    // Shift: output[0] = 0, output[i] = temp[i-1]
    int block = 256;
    int grid = (count + block - 1) / block;

    // Simple shift kernel
    shiftExclusiveKernel<<<grid, block, 0, stream>>>(d_temp, result.d_output, count);

    cudaFree(d_temp);

    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&result.execution_time_ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    result.error = cudaPeekAtLastError();
    return result;
}

// Shift kernel for exclusive scan
template<typename T>
__global__ void shiftExclusiveKernel(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    size_t n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    if (tid == 0) {
        d_output[tid] = T(0);
    } else {
        d_output[tid] = d_input[tid - 1];
    }
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

    // Copy scan result
    cudaMemcpy(d_scan_output, scan_result.d_output, count * sizeof(int), cudaMemcpyDeviceToDevice);

    // Count total valid elements: last element of scan is the count
    int total_count = 0;
    if (count > 0) {
        cudaMemcpy(&total_count, scan_result.d_output + count - 1, sizeof(int), cudaMemcpyDeviceToHost);
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
