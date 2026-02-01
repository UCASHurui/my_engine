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

    // Generate random test data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    for (size_t i = 0; i < N; i++) {
        h_input[i] = dist(gen);
    }

    // Calculate expected values on CPU
    float cpu_sum = 0.0f;
    float cpu_min = h_input[0];
    float cpu_max = h_input[0];
    for (size_t i = 0; i < N; i++) {
        cpu_sum += h_input[i];
        cpu_min = std::min(cpu_min, h_input[i]);
        cpu_max = std::max(cpu_max, h_input[i]);
    }
    float cpu_avg = cpu_sum / N;

    std::cout << "CPU Reference:" << std::endl;
    std::cout << "  Sum: " << std::fixed << std::setprecision(6) << cpu_sum << std::endl;
    std::cout << "  Min: " << cpu_min << std::endl;
    std::cout << "  Max: " << cpu_max << std::endl;
    std::cout << "  Avg: " << cpu_avg << std::endl;

    // Copy to GPU
    float* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Test sum
    {
        auto result = MyEngine::reduceSum(d_input, N);
        std::cout << "GPU Sum: " << std::fixed << std::setprecision(6) << result.value << std::endl;
        std::cout << "  Time: " << result.execution_time_ms << " ms" << std::endl;
        if (std::abs(result.value - cpu_sum) > cpu_sum * 0.001f) {
            std::cerr << "  ERROR: Sum mismatch!" << std::endl;
            return false;
        }
        std::cout << "  PASSED" << std::endl;
    }

    // Test min
    {
        auto result = MyEngine::reduceMin(d_input, N);
        std::cout << "GPU Min: " << result.value << std::endl;
        if (std::abs(result.value - cpu_min) > 0.0001f) {
            std::cerr << "  ERROR: Min mismatch!" << std::endl;
            return false;
        }
        std::cout << "  PASSED" << std::endl;
    }

    // Test max
    {
        auto result = MyEngine::reduceMax(d_input, N);
        std::cout << "GPU Max: " << result.value << std::endl;
        if (std::abs(result.value - cpu_max) > 0.0001f) {
            std::cerr << "  ERROR: Max mismatch!" << std::endl;
            return false;
        }
        std::cout << "  PASSED" << std::endl;
    }

    // Test average
    {
        auto result = MyEngine::reduceAverage(d_input, N);
        std::cout << "GPU Avg: " << std::fixed << std::setprecision(6) << result.value << std::endl;
        if (std::abs(result.value - cpu_avg) > cpu_avg * 0.001f) {
            std::cerr << "  ERROR: Average mismatch!" << std::endl;
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

    const size_t N = 100000;
    std::vector<float> h_input(N);

    // Generate test data
    for (size_t i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }

    // Calculate expected values on CPU
    std::vector<float> h_expected_inclusive(N);
    std::vector<float> h_expected_exclusive(N);
    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) {
        h_expected_inclusive[i] = sum + h_input[i];
        h_expected_exclusive[i] = sum;
        sum += h_input[i];
    }

    std::cout << "Input range: [1, " << N << "], Expected sum: " << sum << std::endl;

    // Copy to GPU
    float* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Test inclusive scan
    {
        auto scan = MyEngine::CUDAScan::create<float>(N);
        auto result = scan.scanInclusive(d_input, N);

        std::cout << "Inclusive Scan:" << std::endl;
        std::cout << "  Time: " << result.execution_time_ms << " ms" << std::endl;

        // Verify first and last elements
        float first, last;
        CUDA_CHECK(cudaMemcpy(&first, result.d_output, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last, result.d_output + N - 1, sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "  First: " << first << " (expected: " << h_expected_inclusive[0] << ")" << std::endl;
        std::cout << "  Last:  " << last << " (expected: " << h_expected_inclusive[N-1] << ")" << std::endl;

        // Verify a few random elements
        bool allCorrect = true;
        std::vector<float> h_output(N);
        CUDA_CHECK(cudaMemcpy(h_output.data(), result.d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < N; i++) {
            if (std::abs(h_output[i] - h_expected_inclusive[i]) > 0.001f) {
                if (allCorrect) {
                    std::cerr << "  First mismatch at index " << i
                              << ": got " << h_output[i] << ", expected " << h_expected_inclusive[i] << std::endl;
                }
                allCorrect = false;
            }
        }

        if (allCorrect) {
            std::cout << "  All elements verified correctly" << std::endl;
            std::cout << "  PASSED" << std::endl;
        } else {
            std::cerr << "  FAILED" << std::endl;
            return false;
        }
    }

    // Test exclusive scan
    {
        auto scan = MyEngine::CUDAScan::create<float>(N);
        auto result = scan.scanExclusive(d_input, N);

        std::cout << "Exclusive Scan:" << std::endl;
        std::cout << "  Time: " << result.execution_time_ms << " ms" << std::endl;

        float first, last;
        CUDA_CHECK(cudaMemcpy(&first, result.d_output, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last, result.d_output + N - 1, sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "  First: " << first << " (expected: " << h_expected_exclusive[0] << ")" << std::endl;
        std::cout << "  Last:  " << last << " (expected: " << h_expected_exclusive[N-1] << ")" << std::endl;

        bool allCorrect = true;
        std::vector<float> h_output(N);
        CUDA_CHECK(cudaMemcpy(h_output.data(), result.d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < N; i++) {
            if (std::abs(h_output[i] - h_expected_exclusive[i]) > 0.001f) {
                if (allCorrect) {
                    std::cerr << "  First mismatch at index " << i
                              << ": got " << h_output[i] << ", expected " << h_expected_exclusive[i] << std::endl;
                }
                allCorrect = false;
            }
        }

        if (allCorrect) {
            std::cout << "  All elements verified correctly" << std::endl;
            std::cout << "  PASSED" << std::endl;
        } else {
            std::cerr << "  FAILED" << std::endl;
            return false;
        }
    }

    CUDA_CHECK(cudaFree(d_input));

    std::cout << "Scan tests PASSED" << std::endl << std::endl;
    return true;
}

// Test stream compaction
bool testCompact() {
    std::cout << "=== Testing Stream Compaction ===" << std::endl;

    const size_t N = 1000;
    std::vector<float> h_input(N);
    std::vector<int> h_predicate(N);

    // Generate test data: keep values > 50
    for (size_t i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i % 100);
        h_predicate[i] = (h_input[i] > 50.0f) ? 1 : 0;
    }

    // Calculate expected result on CPU
    std::vector<float> h_expected;
    for (size_t i = 0; i < N; i++) {
        if (h_predicate[i]) {
            h_expected.push_back(h_input[i]);
        }
    }

    std::cout << "Input: " << N << " elements, keeping values > 50" << std::endl;
    std::cout << "Expected output: " << h_expected.size() << " elements" << std::endl;

    // Copy to GPU
    float* d_input;
    int* d_predicate;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predicate, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_predicate, h_predicate.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Test compact
    {
        auto scan = MyEngine::CUDAScan::create<float>(N);
        auto result = scan.compact(d_input, d_predicate, N);

        std::cout << "Compact:" << std::endl;
        std::cout << "  Time: " << result.execution_time_ms << " ms" << std::endl;
        std::cout << "  Output count: " << result.count << std::endl;

        if (result.count != h_expected.size()) {
            std::cerr << "  ERROR: Count mismatch! Got " << result.count
                      << ", expected " << h_expected.size() << std::endl;
            return false;
        }

        // Verify elements
        std::vector<float> h_output(result.count);
        CUDA_CHECK(cudaMemcpy(h_output.data(), result.d_output,
                              result.count * sizeof(float), cudaMemcpyDeviceToHost));

        bool allCorrect = true;
        for (size_t i = 0; i < result.count; i++) {
            if (std::abs(h_output[i] - h_expected[i]) > 0.001f) {
                std::cerr << "  ERROR at index " << i
                          << ": got " << h_output[i] << ", expected " << h_expected[i] << std::endl;
                allCorrect = false;
            }
        }

        if (allCorrect) {
            std::cout << "  All elements verified correctly" << std::endl;
            std::cout << "  PASSED" << std::endl;
        } else {
            std::cerr << "  FAILED" << std::endl;
            return false;
        }
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_predicate));

    std::cout << "Compact test PASSED" << std::endl << std::endl;
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

    std::vector<size_t> sizes = {10000, 100000, 1000000, 10000000};

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
            auto result = MyEngine::reduceSum(d_input, N);
            float throughput = (N * sizeof(float)) / (result.execution_time_ms * 1e6);
            std::cout << "  Sum: " << result.execution_time_ms << " ms"
                      << " (" << std::fixed << std::setprecision(1) << throughput << " GB/s)" << std::endl;
        }

        // Benchmark inclusive scan
        {
            auto scan = MyEngine::CUDAScan::create<float>(N);
            auto result = scan.scanInclusive(d_input, N);
            float throughput = (N * sizeof(float)) / (result.execution_time_ms * 1e6);
            std::cout << "  Inclusive Scan: " << result.execution_time_ms << " ms"
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
