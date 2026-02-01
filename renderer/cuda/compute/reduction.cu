#include "reduction.h"
#include "cuda_error.h"
#include "cuda_math.h"
#include <cfloat>

namespace MyEngine {

// Block-level reduction using sequential addressing (no warp shuffle)
template<typename T, ReductionOp Op>
__device__ T blockReduce(T* sdata, unsigned int tid, unsigned int blockSize) {
    // Tree reduction in shared memory
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            switch (Op) {
                case ReductionOp::Sum:
                case ReductionOp::Average:
                    sdata[tid] = sdata[tid] + sdata[tid + s];
                    break;
                case ReductionOp::Min:
                    sdata[tid] = min(sdata[tid], sdata[tid + s]);
                    break;
                case ReductionOp::Max:
                    sdata[tid] = max(sdata[tid], sdata[tid + s]);
                    break;
            }
        }
        __syncthreads();
    }
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

    T result = blockReduce<T, ReductionOp::Sum>(sdata_t, tid, blockDim.x);

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

    T result = blockReduce<T, ReductionOp::Min>(sdata_t, tid, blockDim.x);

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

    T result = blockReduce<T, ReductionOp::Max>(sdata_t, tid, blockDim.x);

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

    T result = blockReduce<T, Op>(sdata_t, tid, blockDim.x);

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
    // Allocate temp storage for reduction
    allocateStorage(max_elements);
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
    // Don't cap the grid size - process all elements
    return (count + block_size - 1) / block_size;
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
        // For large numbers of partial results, do a two-level reduction
        // First, reduce all grid_size partial results to fewer partial results

        // Calculate how many blocks we need to reduce all grid_size results
        int reduce_block_size = 256;
        int reduce_grid_size = (grid_size + reduce_block_size - 1) / reduce_block_size;
        size_t reduce_smem_size = reduce_block_size * sizeof(T);

        // First level: reduce grid_size partial sums to reduce_grid_size partial sums
        switch (_operation) {
            case ReductionOp::Sum:
            case ReductionOp::Average:
                reduceSumKernel<<<reduce_grid_size, reduce_block_size, reduce_smem_size, use_stream>>>(
                    _d_temp_storage, _d_temp_storage, grid_size);
                break;
            case ReductionOp::Min:
                reduceMinKernel<<<reduce_grid_size, reduce_block_size, reduce_smem_size, use_stream>>>(
                    _d_temp_storage, _d_temp_storage, grid_size);
                break;
            case ReductionOp::Max:
                reduceMaxKernel<<<reduce_grid_size, reduce_block_size, reduce_smem_size, use_stream>>>(
                    _d_temp_storage, _d_temp_storage, grid_size);
                break;
        }
        // Synchronize to ensure all writes are visible
        CUDA_CHECK(cudaStreamSynchronize(use_stream));

        // Second level: reduce reduce_grid_size partial sums to one result
        if (reduce_grid_size > 1) {
            // Use a full block size (256) for the final reduction
            int final_block_size = 256;
            size_t final_smem_size = final_block_size * sizeof(T);

            // The reduceFinalKernel expects n = number of elements to reduce
            // which is reduce_grid_size
            switch (_operation) {
                case ReductionOp::Sum:
                case ReductionOp::Average:
                    reduceFinalKernel<T, ReductionOp::Sum><<<1, final_block_size, final_smem_size, use_stream>>>(
                        _d_temp_storage, d_output, reduce_grid_size);
                    break;
                case ReductionOp::Min:
                    reduceFinalKernel<T, ReductionOp::Min><<<1, final_block_size, final_smem_size, use_stream>>>(
                        _d_temp_storage, d_output, reduce_grid_size);
                    break;
                case ReductionOp::Max:
                    reduceFinalKernel<T, ReductionOp::Max><<<1, final_block_size, final_smem_size, use_stream>>>(
                        _d_temp_storage, d_output, reduce_grid_size);
                    break;
            }
        } else {
            // Single partial result left
            cudaMemcpy(d_output, _d_temp_storage, sizeof(T), cudaMemcpyDeviceToDevice);
        }
    } else {
        // Single block, result is already in temp storage
        cudaMemcpy(d_output, _d_temp_storage, sizeof(T), cudaMemcpyDeviceToDevice);
    }

    // For average, divide by count
    if (_operation == ReductionOp::Average) {
        T result;
        cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost);
        result = result / (T)count;
        cudaMemcpy(d_output, &result, sizeof(T), cudaMemcpyHostToDevice);
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

    // Read result - use cudaMemcpy to ensure synchronization
    T result = T{};
    if (err == cudaSuccess) {
        // Ensure all GPU operations are complete
        cudaDeviceSynchronize();
        err = cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_output);
    return ReductionResult<T>(result, err, elapsed_ms);
}

// Convenience function implementations
template<typename T>
ReductionResult<T> reduceSum(const T* d_input, size_t count, cudaStream_t stream) {
    CUDAReduction<T> reducer(ReductionOp::Sum, count);
    return reducer.reduce(d_input, count, stream);
}

template<typename T>
ReductionResult<T> reduceMin(const T* d_input, size_t count, cudaStream_t stream) {
    CUDAReduction<T> reducer(ReductionOp::Min, count);
    return reducer.reduce(d_input, count, stream);
}

template<typename T>
ReductionResult<T> reduceMax(const T* d_input, size_t count, cudaStream_t stream) {
    CUDAReduction<T> reducer(ReductionOp::Max, count);
    return reducer.reduce(d_input, count, stream);
}

template<typename T>
ReductionResult<T> reduceAverage(const T* d_input, size_t count, cudaStream_t stream) {
    CUDAReduction<T> reducer(ReductionOp::Average, count);
    return reducer.reduce(d_input, count, stream);
}

// Explicit instantiations for CUDAReduction class
template class CUDAReduction<float>;
template class CUDAReduction<int>;
template class CUDAReduction<double>;

// Explicit template instantiations for convenience functions
template ReductionResult<float> reduceSum(const float*, size_t, cudaStream_t);
template ReductionResult<float> reduceMin(const float*, size_t, cudaStream_t);
template ReductionResult<float> reduceMax(const float*, size_t, cudaStream_t);
template ReductionResult<float> reduceAverage(const float*, size_t, cudaStream_t);

template ReductionResult<int> reduceSum(const int*, size_t, cudaStream_t);
template ReductionResult<int> reduceMin(const int*, size_t, cudaStream_t);
template ReductionResult<int> reduceMax(const int*, size_t, cudaStream_t);
template ReductionResult<int> reduceAverage(const int*, size_t, cudaStream_t);

} // namespace MyEngine
