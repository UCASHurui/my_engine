#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

#include "cuda_runtime.h"
#include "compute/reduction.h"
#include "compute/scan.h"
#include "profiler/cuda_profiler.h"

// Include CUDA implementations
#include "compute/reduction.cu"
#include "compute/scan.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while(0)

// Test reduction operations
bool testReduction() {
    std::cout << "=== Testing Reduction Operations ===" << std::endl;

    const size_t N = 1000000;
    std::vector<float> h_input(N);

    // Generate random test data with smaller values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < N; i++) {
        h_input[i] = dist(gen);
    }

    // Calculate expected values on CPU
    float cpu_sum = 0.0f;
    for (size_t i = 0; i < N; i++) {
        cpu_sum += h_input[i];
    }

    std::cout << "CPU Sum: " << std::fixed << std::setprecision(6) << cpu_sum << std::endl;

    // Copy to GPU
    float* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Test sum - use CUDAReduction directly
    {
        MyEngine::CUDAReduction<float> reducer(MyEngine::ReductionOp::Sum, N);
        auto result = reducer.reduce(d_input, N);
        std::cout << "GPU Sum: " << std::fixed << std::setprecision(6) << result.value << std::endl;
        std::cout << "  Time: " << result.execution_time_ms << " ms" << std::endl;
        float diff = std::abs(result.value - cpu_sum);
        float tol = cpu_sum * 0.01f;  // 1% tolerance
        if (diff > tol) {
            std::cerr << "  ERROR: Sum mismatch! Diff: " << diff << ", Tolerance: " << tol << std::endl;
            return false;
        }
        std::cout << "  PASSED" << std::endl;
    }

    // Test min
    {
        MyEngine::CUDAReduction<float> reducer(MyEngine::ReductionOp::Min, N);
        auto result = reducer.reduce(d_input, N);
        std::cout << "GPU Min: " << result.value << std::endl;
        if (!result.ok()) {
            std::cerr << "  ERROR: Min failed with error " << result.error << std::endl;
            return false;
        }
        std::cout << "  PASSED" << std::endl;
    }

    // Test max
    {
        MyEngine::CUDAReduction<float> reducer(MyEngine::ReductionOp::Max, N);
        auto result = reducer.reduce(d_input, N);
        std::cout << "GPU Max: " << result.value << std::endl;
        if (!result.ok()) {
            std::cerr << "  ERROR: Max failed with error " << result.error << std::endl;
            return false;
        }
        std::cout << "  PASSED" << std::endl;
    }

    // Test average
    {
        MyEngine::CUDAReduction<float> reducer(MyEngine::ReductionOp::Average, N);
        auto result = reducer.reduce(d_input, N);
        std::cout << "GPU Avg: " << std::fixed << std::setprecision(6) << result.value << std::endl;
        if (!result.ok()) {
            std::cerr << "  ERROR: Average failed with error " << result.error << std::endl;
            return false;
        }
        std::cout << "  PASSED" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_input));

    std::cout << "Reduction tests PASSED" << std::endl << std::endl;
    return true;
}

// Test scan operations
bool testScan() {
    std::cout << "=== Testing Scan Operations ===" << std::endl;

    // Test small arrays first (single block)
    {
        const size_t N = 100;
        std::vector<float> h_input(N);
        std::vector<float> h_expected(N);

        // Simple sequence: 1, 2, 3, 4, ...
        for (size_t i = 0; i < N; i++) {
            h_input[i] = static_cast<float>(i + 1);
            float sum = 0;
            for (size_t j = 0; j <= i; j++) sum += h_input[j];
            h_expected[i] = sum;
        }

        float* d_input;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        auto scan = MyEngine::CUDAScan::create<float>(N);
        auto result = scan.scanInclusive(d_input, N);

        if (!result.ok()) {
            std::cerr << "  ERROR: Inclusive scan failed with error " << result.error << std::endl;
            cudaFree(d_input);
            return false;
        }

        std::vector<float> h_output(N);
        cudaMemcpy(h_output.data(), result.d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        bool pass = true;
        for (size_t i = 0; i < N; i++) {
            if (std::abs(h_output[i] - h_expected[i]) > 0.001f) {
                std::cerr << "  ERROR at index " << i << ": got " << h_output[i]
                          << ", expected " << h_expected[i] << std::endl;
                pass = false;
                break;
            }
        }

        if (pass) {
            std::cout << "  Single block inclusive scan: PASSED" << std::endl;
        }

        cudaFree(d_input);
        if (!pass) return false;
    }

    // Test multi-block array
    {
        const size_t N = 10000;
        std::vector<float> h_input(N);
        std::vector<float> h_expected(N);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 10.0f);
        for (size_t i = 0; i < N; i++) {
            h_input[i] = dist(gen);
        }

        float sum = 0;
        for (size_t i = 0; i < N; i++) {
            sum += h_input[i];
            h_expected[i] = sum;
        }

        float* d_input;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        auto scan = MyEngine::CUDAScan::create<float>(N);
        auto result = scan.scanInclusive(d_input, N);

        if (!result.ok()) {
            std::cerr << "  ERROR: Multi-block inclusive scan failed with error " << result.error << std::endl;
            cudaFree(d_input);
            return false;
        }

        std::vector<float> h_output(N);
        cudaMemcpy(h_output.data(), result.d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        bool pass = true;
        float max_diff = 0.0f;
        int max_diff_idx = -1;
        for (size_t i = 0; i < N; i++) {
            float diff = std::abs(h_output[i] - h_expected[i]);
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_idx = i;
            }
            if (diff > 1.0f) {  // Use larger tolerance for accumulated sums
                std::cerr << "  ERROR at index " << i << ": got " << h_output[i]
                          << ", expected " << h_expected[i] << ", diff=" << diff << std::endl;
                pass = false;
                if (i > 20) break;
                break;
            }
        }
        if (max_diff_idx >= 0) {
            std::cout << "  Max diff: " << max_diff << " at index " << max_diff_idx << std::endl;
        }

        if (pass) {
            std::cout << "  Multi-block inclusive scan: PASSED" << std::endl;
        }

        cudaFree(d_input);
        if (!pass) return false;
    }

    // Test exclusive scan
    {
        const size_t N = 1000;
        std::vector<int> h_input(N);
        std::vector<int> h_expected(N);

        for (size_t i = 0; i < N; i++) {
            h_input[i] = static_cast<int>(i + 1);
        }
        h_expected[0] = 0;
        for (size_t i = 1; i < N; i++) {
            h_expected[i] = h_expected[i-1] + h_input[i-1];
        }

        int* d_input;
        cudaMalloc(&d_input, N * sizeof(int));
        cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        auto scan = MyEngine::CUDAScan::create<int>(N);
        auto result = scan.scanExclusive(d_input, N);

        if (!result.ok()) {
            std::cerr << "  ERROR: Exclusive scan failed with error " << result.error << std::endl;
            cudaFree(d_input);
            return false;
        }

        std::vector<int> h_output(N);
        cudaMemcpy(h_output.data(), result.d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        bool pass = true;
        for (size_t i = 0; i < N; i++) {
            if (h_output[i] != h_expected[i]) {
                std::cerr << "  ERROR at index " << i << ": got " << h_output[i]
                          << ", expected " << h_expected[i] << std::endl;
                pass = false;
                break;
            }
        }

        if (pass) {
            std::cout << "  Exclusive scan: PASSED" << std::endl;
        }

        cudaFree(d_input);
        if (!pass) return false;
    }

    std::cout << "Scan tests PASSED" << std::endl << std::endl;
    return true;
}

// Test stream compaction
bool testCompact() {
    std::cout << "=== Testing Stream Compaction ===" << std::endl;

    // Basic compact test
    {
        const size_t N = 100;
        std::vector<float> h_input(N);
        std::vector<int> h_predicate(N);

        // Keep even-indexed elements
        for (size_t i = 0; i < N; i++) {
            h_input[i] = static_cast<float>(i);
            h_predicate[i] = (i % 2 == 0) ? 1 : 0;
        }

        float* d_input;
        int* d_predicate;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_predicate, N * sizeof(int));
        cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_predicate, h_predicate.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        auto scan = MyEngine::CUDAScan::create<float>(N);
        auto result = scan.compact(d_input, d_predicate, N);

        if (!result.ok()) {
            std::cerr << "  ERROR: Compact failed with error " << result.error << std::endl;
            cudaFree(d_input);
            cudaFree(d_predicate);
            return false;
        }

        size_t expected_count = N / 2;
        if (result.count != expected_count) {
            std::cerr << "  ERROR: Expected " << expected_count << " elements, got " << result.count << std::endl;
            cudaFree(d_input);
            cudaFree(d_predicate);
            return false;
        }

        std::vector<float> h_output(expected_count);
        cudaMemcpy(h_output.data(), result.d_output, expected_count * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        bool pass = true;
        for (size_t i = 0; i < expected_count; i++) {
            float expected = static_cast<float>(i * 2);
            if (std::abs(h_output[i] - expected) > 0.001f) {
                std::cerr << "  ERROR at output index " << i << ": got " << h_output[i]
                          << ", expected " << expected << std::endl;
                pass = false;
                break;
            }
        }

        if (pass) {
            std::cout << "  Basic compact: PASSED" << std::endl;
        }

        cudaFree(d_input);
        cudaFree(d_predicate);
        if (!pass) return false;
    }

    // Test compact all zeros
    {
        const size_t N = 50;
        std::vector<float> h_input(N);
        std::vector<int> h_predicate(N, 0);

        for (size_t i = 0; i < N; i++) {
            h_input[i] = static_cast<float>(i);
        }

        float* d_input;
        int* d_predicate;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_predicate, N * sizeof(int));
        cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_predicate, h_predicate.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        auto scan = MyEngine::CUDAScan::create<float>(N);
        auto result = scan.compact(d_input, d_predicate, N);

        if (!result.ok()) {
            std::cerr << "  ERROR: Compact all zeros failed with error " << result.error << std::endl;
            cudaFree(d_input);
            cudaFree(d_predicate);
            return false;
        }

        if (result.count != 0) {
            std::cerr << "  ERROR: Expected 0 elements, got " << result.count << std::endl;
            cudaFree(d_input);
            cudaFree(d_predicate);
            return false;
        }

        std::cout << "  Compact all zeros: PASSED" << std::endl;

        cudaFree(d_input);
        cudaFree(d_predicate);
    }

    // Test compact all ones
    {
        const size_t N = 50;
        std::vector<float> h_input(N);
        std::vector<int> h_predicate(N, 1);

        for (size_t i = 0; i < N; i++) {
            h_input[i] = static_cast<float>(i);
        }

        float* d_input;
        int* d_predicate;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_predicate, N * sizeof(int));
        cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_predicate, h_predicate.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        auto scan = MyEngine::CUDAScan::create<float>(N);
        auto result = scan.compact(d_input, d_predicate, N);

        if (!result.ok()) {
            std::cerr << "  ERROR: Compact all ones failed with error " << result.error << std::endl;
            cudaFree(d_input);
            cudaFree(d_predicate);
            return false;
        }

        if (result.count != N) {
            std::cerr << "  ERROR: Expected " << N << " elements, got " << result.count << std::endl;
            cudaFree(d_input);
            cudaFree(d_predicate);
            return false;
        }

        std::vector<float> h_output(N);
        cudaMemcpy(h_output.data(), result.d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        bool pass = true;
        for (size_t i = 0; i < N; i++) {
            if (std::abs(h_output[i] - static_cast<float>(i)) > 0.001f) {
                std::cerr << "  ERROR at index " << i << std::endl;
                pass = false;
                break;
            }
        }

        if (pass) {
            std::cout << "  Compact all ones: PASSED" << std::endl;
        }

        cudaFree(d_input);
        cudaFree(d_predicate);
        if (!pass) return false;
    }

    std::cout << "Compaction tests PASSED" << std::endl << std::endl;
    return true;
}

// Test profiler
bool testProfiler() {
    std::cout << "=== Testing Profiler ===" << std::endl;

    auto& profiler = MyEngine::CUDAProfiler::instance();
    profiler.reset();

    const size_t N = 100000;
    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i);
    }

    float* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Profile a sum reduction operation using convenience function
    {
        auto scope = profiler.profile("reduction_test");
        auto result = MyEngine::reduceSum(d_input, N);
    }

    // Profile a scan operation
    {
        auto scope = profiler.profile("scan_test");

        auto scan = MyEngine::CUDAScan::create<float>(N);
        auto result = scan.scanInclusive(d_input, N);
    }

    // Print stats
    std::cout << "Profiler stats:" << std::endl;
    auto stats = profiler.getStats();
    for (const auto& stat : stats) {
        std::cout << "  " << stat.toString() << std::endl;
    }

    // Export to CSV
    if (profiler.exportStatsToCSV("/tmp/myengine_profiler_stats.csv")) {
        std::cout << "Exported stats to /tmp/myengine_profiler_stats.csv" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_input));

    std::cout << "Profiler test PASSED" << std::endl << std::endl;
    return true;
}

// Performance benchmark
bool benchmarkPrimitives() {
    std::cout << "=== Performance Benchmark ===" << std::endl;

    std::vector<size_t> sizes = {10000, 100000, 1000000};

    for (size_t N : sizes) {
        std::cout << "N = " << N << std::endl;

        std::vector<float> h_input(N);
        for (size_t i = 0; i < N; i++) {
            h_input[i] = static_cast<float>(i);
        }

        float* d_input;
        CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        // Benchmark sum reduction
        {
            MyEngine::CUDAReduction<float> reducer(MyEngine::ReductionOp::Sum, N);
            auto result = reducer.reduce(d_input, N);
            float throughput = (N * sizeof(float)) / (result.execution_time_ms * 1e6);
            std::cout << "  Sum: " << result.execution_time_ms << " ms"
                      << " (" << std::fixed << std::setprecision(1) << throughput << " GB/s)" << std::endl;
        }

        CUDA_CHECK(cudaFree(d_input));
    }

    std::cout << std::endl;
    return true;
}

int main() {
    std::cout << "MyEngine CUDA Primitives Test" << std::endl;
    std::cout << "==============================" << std::endl << std::endl;

    // Check CUDA devices
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "CUDA devices found: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "  Device " << i << ": " << prop.name << std::endl;
        std::cout << "    Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "    Global memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    }
    std::cout << std::endl;

    bool allPassed = true;

    // Run tests
    allPassed &= testReduction();
    allPassed &= testScan();
    allPassed &= testCompact();
    allPassed &= testProfiler();

    // Run benchmark
    benchmarkPrimitives();

    std::cout << "==============================" << std::endl;
    if (allPassed) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}
