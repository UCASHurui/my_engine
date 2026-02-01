#include "reduction.h"
#include "cuda_error.h"
#include "cuda_math.h"
#include <cfloat>

namespace MyEngine {

// Warp-level reduction using shuffle
template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warpReduceMin(T val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction
template<typename T, ReductionOp Op>
__device__ T blockReduce(T* sdata, unsigned int tid) {
    const unsigned int blockSize = blockDim.x;

    // Warp reduction
    T val = sdata[tid];
    switch (Op) {
        case ReductionOp::Sum:
            val = warpReduceSum(val);
            break;
        case ReductionOp::Min:
            val = warpReduceMin(val);
            break;
        case ReductionOp::Max:
            val = warpReduceMax(val);
            break;
        case ReductionOp::Average:
            val = warpReduceSum(val);
            break;
    }

    // Final warp reduction at tid == 0
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = val;
    }
    __syncthreads();

    // Reduce across warps
    const unsigned int warpCount = (blockSize + warpSize - 1) / warpSize;
    if (tid < warpCount) {
        val = sdata[tid];
        switch (Op) {
            case ReductionOp::Sum:
                val = warpReduceSum(val);
                break;
            case ReductionOp::Min:
                val = warpReduceMin(val);
                break;
            case ReductionOp::Max:
                val = warpReduceMax(val);
                break;
            case ReductionOp::Average:
                val = warpReduceSum(val);
                break;
        }
        sdata[tid] = val;
    }
    __syncthreads();

    return sdata[0];
}

// Reduction kernel for sum
template<typename T>
__global__ void reduceSumKernel(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    size_t n
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    T val = (idx < n) ? d_input[idx] : T(0);
    T* sdata_t = reinterpret_cast<T*>(sdata);

    sdata_t[tid] = val;
    __syncthreads();

    T result = blockReduce<T, ReductionOp::Sum>(sdata_t, tid);

    if (tid == 0) {
        d_output[blockIdx.x] = result;
    }
}

// Reduction kernel for min
template<typename T>
__global__ void reduceMinKernel(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    size_t n
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    T val = (idx < n) ? d_input[idx] : T(FLT_MAX);
    T* sdata_t = reinterpret_cast<T*>(sdata);

    sdata_t[tid] = val;
    __syncthreads();

    T result = blockReduce<T, ReductionOp::Min>(sdata_t, tid);

    if (tid == 0) {
        d_output[blockIdx.x] = result;
    }
}

// Reduction kernel for max
template<typename T>
__global__ void reduceMaxKernel(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    size_t n
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    T val = (idx < n) ? d_input[idx] : T(-FLT_MAX);
    T* sdata_t = reinterpret_cast<T*>(sdata);

    sdata_t[tid] = val;
    __syncthreads();

    T result = blockReduce<T, ReductionOp::Max>(sdata_t, tid);

    if (tid == 0) {
        d_output[blockIdx.x] = result;
    }
}

// Final reduction kernel (single block)
template<typename T, ReductionOp Op>
__global__ void reduceFinalKernel(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    size_t n
) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    T* sdata_t = reinterpret_cast<T*>(sdata);
    sdata_t[tid] = (tid < n) ? d_input[tid] : T(0);
    __syncthreads();

    T result = blockReduce<T, Op>(sdata_t, tid);

    if (tid == 0) {
        d_output[0] = result;
    }
}

// CUDAReduction implementation
template<typename T>
CUDAReduction<T>::CUDAReduction(ReductionOp operation, size_t max_elements)
    : _operation(operation), _max_elements(max_elements), _stream(0),
      _d_temp_storage(nullptr), _temp_storage_bytes(0) {
    cudaError_t err = cudaStreamCreate(&_stream);
    if (err != cudaSuccess) {
        CUDA_LOG_ERROR_WITH_CODE("Failed to create stream for reduction", err);
    }
}

template<typename T>
CUDAReduction<T>::~CUDAReduction() {
    freeStorage();
    if (_stream) {
        cudaStreamDestroy(_stream);
    }
}

template<typename T>
CUDAReduction<T>::CUDAReduction(CUDAReduction&& other) noexcept
    : _operation(other._operation), _max_elements(other._max_elements),
      _stream(other._stream), _d_temp_storage(other._d_temp_storage),
      _temp_storage_bytes(other._temp_storage_bytes) {
    other._stream = 0;
    other._d_temp_storage = nullptr;
}

template<typename T>
CUDAReduction<T>& CUDAReduction<T>::operator=(CUDAReduction&& other) noexcept {
    if (this != &other) {
        freeStorage();
        if (_stream) cudaStreamDestroy(_stream);

        _operation = other._operation;
        _max_elements = other._max_elements;
        _stream = other._stream;
        _d_temp_storage = other._d_temp_storage;
        _temp_storage_bytes = other._temp_storage_bytes;

        other._stream = 0;
        other._d_temp_storage = nullptr;
    }
    return *this;
}

template<typename T>
void CUDAReduction<T>::allocateStorage(size_t max_elements) {
    freeStorage();

    int block_size = 256;
    int grid_size = (max_elements + block_size - 1) / block_size;
    _temp_storage_bytes = grid_size * sizeof(T);

    cudaError_t err = cudaMalloc(&_d_temp_storage, _temp_storage_bytes);
    if (err != cudaSuccess) {
        CUDA_LOG_ERROR_WITH_CODE("Failed to allocate reduction temp storage", err);
    }
}

template<typename T>
void CUDAReduction<T>::freeStorage() {
    if (_d_temp_storage) {
        cudaFree(_d_temp_storage);
        _d_temp_storage = nullptr;
        _temp_storage_bytes = 0;
    }
}

template<typename T>
int CUDAReduction<T>::getOptimalBlockCount(size_t count, int block_size) {
    int max_blocks = 1024;  // Maximum blocks to launch
    int grid_size = (count + block_size - 1) / block_size;
    return min(grid_size, max_blocks);
}

template<typename T>
cudaError_t CUDAReduction<T>::reduceAsync(
    const T* d_input,
    T* d_output,
    size_t count,
    cudaStream_t stream
) {
    if (count == 0) {
        d_output[0] = T(0);
        return cudaSuccess;
    }

    if (count == 1) {
        d_output[0] = d_input[0];
        return cudaSuccess;
    }

    int block_size = 256;
    int grid_size = getOptimalBlockCount(count, block_size);
    size_t smem_size = block_size * sizeof(T);

    cudaStream_t use_stream = stream ? stream : _stream;

    switch (_operation) {
        case ReductionOp::Sum:
            reduceSumKernel<<<grid_size, block_size, smem_size, use_stream>>>(
                d_input, _d_temp_storage, count);
            break;
        case ReductionOp::Min:
            reduceMinKernel<<<grid_size, block_size, smem_size, use_stream>>>(
                d_input, _d_temp_storage, count);
            break;
        case ReductionOp::Max:
            reduceMaxKernel<<<grid_size, block_size, smem_size, use_stream>>>(
                d_input, _d_temp_storage, count);
            break;
        case ReductionOp::Average:
            reduceSumKernel<<<grid_size, block_size, smem_size, use_stream>>>(
                d_input, _d_temp_storage, count);
            break;
    }

    // Final reduction if more than one block
    if (grid_size > 1) {
        int final_block_size = min((int)count, 256);
        size_t final_smem_size = final_block_size * sizeof(T);

        switch (_operation) {
            case ReductionOp::Sum:
            case ReductionOp::Average:
                reduceFinalKernel<T, ReductionOp::Sum><<<1, final_block_size, final_smem_size, use_stream>>>(
                    _d_temp_storage, d_output, grid_size);
                break;
            case ReductionOp::Min:
                reduceFinalKernel<T, ReductionOp::Min><<<1, final_block_size, final_smem_size, use_stream>>>(
                    _d_temp_storage, d_output, grid_size);
                break;
            case ReductionOp::Max:
                reduceFinalKernel<T, ReductionOp::Max><<<1, final_block_size, final_smem_size, use_stream>>>(
                    _d_temp_storage, d_output, grid_size);
                break;
        }
    } else {
        // Single block, result is already in temp storage
        cudaMemcpy(d_output, _d_temp_storage, sizeof(T), cudaMemcpyDeviceToDevice);
    }

    // For average, divide by count
    if (_operation == ReductionOp::Average) {
        T result;
        cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToDevice);
        result = result / (T)count;
        cudaMemcpy(d_output, &result, sizeof(T), cudaMemcpyDeviceToDevice);
    }

    return cudaPeekAtLastError();
}

template<typename T>
ReductionResult<T> CUDAReduction<T>::reduce(
    const T* d_input,
    size_t count,
    cudaStream_t stream
) {
    // Allocate output if not provided
    T* d_output;
    cudaError_t err = cudaMalloc(&d_output, sizeof(T));
    if (err != cudaSuccess) {
        return ReductionResult<T>(T{}, err, 0.0f);
    }

    // Time the operation
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);

    err = reduceAsync(d_input, d_output, count, stream);

    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // Read result
    T result = T{};
    if (err == cudaSuccess) {
        err = cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_output);
    return ReductionResult<T>(result, err, elapsed_ms);
}

// Explicit instantiations for CUDAReduction class
template class CUDAReduction<float>;
template class CUDAReduction<int>;
template class CUDAReduction<double>;

} // namespace MyEngine
