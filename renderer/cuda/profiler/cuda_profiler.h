#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "cuda_runtime.h"

namespace MyEngine {

// Forward declarations
class CUDAProfiler;

/**
 * RAII scope for timing GPU operations
 * Automatically records timing when destroyed
 */
class CUDAProfileScope {
public:
    CUDAProfileScope(const char* name, CUDAProfiler& profiler);
    ~CUDAProfileScope();

    // Non-copyable, movable
    CUDAProfileScope(const CUDAProfileScope&) = delete;
    CUDAProfileScope& operator=(const CUDAProfileScope&) = delete;
    CUDAProfileScope(CUDAProfileScope&& other) noexcept;
    CUDAProfileScope& operator=(CUDAProfileScope&& other) noexcept;

private:
    const char* _name;
    CUDAProfiler& _profiler;
    cudaEvent_t _startEvent;
    cudaEvent_t _endEvent;
    bool _active;
};

// Timing result for a single measurement
struct TimingResult {
    std::string name;
    float gpu_time_ms;
    std::chrono::microseconds cpu_time_us;
    size_t bytes_transferred;
    size_t operations;
    cudaStream_t stream;

    TimingResult()
        : name(""), gpu_time_ms(0.0f), cpu_time_us(0),
          bytes_transferred(0), operations(0), stream(nullptr) {}

    // Internal timing helpers
    void startTimingInternal() {
        _cpuStart = std::chrono::high_resolution_clock::now();
    }

    void endTimingInternal() {
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        cpu_time_us = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - _cpuStart);
    }

private:
    std::chrono::high_resolution_clock::time_point _cpuStart;
};

// Aggregated statistics for repeated measurements
struct TimingStats {
    std::string name;
    int count;
    float min_ms;
    float max_ms;
    float avg_ms;
    float total_ms;
    std::chrono::microseconds total_cpu_us;

    TimingStats()
        : name(""), count(0), min_ms(0.0f), max_ms(0.0f),
          avg_ms(0.0f), total_ms(0.0f), total_cpu_us(0) {}

    void aggregate(const TimingResult& result);
    std::string toString() const;
};

// Performance profiler singleton
class CUDAProfiler {
public:
    static CUDAProfiler& instance();

    // Profile a named operation (auto-creates scope)
    CUDAProfileScope profile(const char* name);

    // Manual timing
    TimingResult startTiming(const char* name);
    void endTiming(TimingResult& result);

    // Get/clear current timing result
    TimingResult getLastTiming() const { return _lastTiming; }

    // Statistics
    void updateStats(const TimingResult& result);
    std::vector<TimingStats> getStats() const;
    TimingStats getStatsFor(const char* name) const;

    // Export
    bool exportToCSV(const std::string& filepath);
    bool exportStatsToCSV(const std::string& filepath);

    // Configuration
    void setEnabled(bool enabled) { _enabled = enabled; }
    bool isEnabled() const { return _enabled; }

    void setTrackBytes(bool track) { _trackBytes = track; }
    bool shouldTrackBytes() const { return _trackBytes; }

    void setTrackOperations(bool track) { _trackOperations = track; }
    bool shouldTrackOperations() const { return _trackOperations; }

    // Reset
    void reset();
    void resetStats();

    // Destructor
    ~CUDAProfiler();

private:
    CUDAProfiler();
    CUDAProfiler(const CUDAProfiler&) = delete;
    CUDAProfiler& operator=(const CUDAProfiler&) = delete;

    bool _enabled;
    bool _trackBytes;
    bool _trackOperations;

    TimingResult _lastTiming;
    std::vector<TimingResult> _results;
    mutable std::mutex _resultsMutex;
    std::vector<TimingStats> _stats;
    mutable std::mutex _statsMutex;

    cudaEvent_t _globalStartEvent;
    cudaEvent_t _globalEndEvent;

    void updateStatsInternal(const TimingResult& result);
};

// Convenience macro for profiling
#ifdef ENABLE_CUDA_PROFILING
    #define CUDA_PROFILE_SCOPE(name) \
        CUDAProfileScope CUDA_MACRO_CONCAT(_profile_scope_, __LINE__)(name, MyEngine::CUDAProfiler::instance())
    #define CUDA_PROFILE_FUNCTION() \
        CUDAProfileScope CUDA_MACRO_CONCAT(_profile_func_, __LINE__)(__PRETTY_FUNCTION__, MyEngine::CUDAProfiler::instance())
    #define CUDA_MACRO_CONCAT(a, b) CUDA_MACRO_CONCAT_IMPL(a, b)
    #define CUDA_MACRO_CONCAT_IMPL(a, b) a##b
#else
    #define CUDA_PROFILE_SCOPE(name) ((void)0)
    #define CUDA_PROFILE_FUNCTION() ((void)0)
#endif

// Inline implementation of CUDAProfileScope
inline CUDAProfileScope::CUDAProfileScope(const char* name, CUDAProfiler& profiler)
    : _name(name), _profiler(profiler), _active(false) {
    if (!_profiler.isEnabled()) return;

    cudaError_t err = cudaEventCreate(&_startEvent);
    if (err != cudaSuccess) return;
    err = cudaEventCreate(&_endEvent);
    if (err != cudaSuccess) {
        cudaEventDestroy(_startEvent);
        return;
    }
    _active = true;
    cudaEventRecord(_startEvent);
}

inline CUDAProfileScope::~CUDAProfileScope() {
    if (!_active) return;

    cudaEventRecord(_endEvent);
    cudaEventSynchronize(_endEvent);

    TimingResult result = _profiler.startTiming(_name);
    cudaEventElapsedTime(&result.gpu_time_ms, _startEvent, _endEvent);
    _profiler.endTiming(result);

    cudaEventDestroy(_startEvent);
    cudaEventDestroy(_endEvent);
}

inline CUDAProfileScope::CUDAProfileScope(CUDAProfileScope&& other) noexcept
    : _name(other._name), _profiler(other._profiler),
      _startEvent(other._startEvent), _endEvent(other._endEvent),
      _active(other._active) {
    other._active = false;
}

inline CUDAProfileScope& CUDAProfileScope::operator=(CUDAProfileScope&& other) noexcept {
    if (this != &other) {
        _name = other._name;
        _startEvent = other._startEvent;
        _endEvent = other._endEvent;
        _active = other._active;
        other._active = false;
    }
    return *this;
}

// Inline implementation of TimingStats
inline void TimingStats::aggregate(const TimingResult& result) {
    if (count == 0) {
        name = result.name;
        min_ms = result.gpu_time_ms;
        max_ms = result.gpu_time_ms;
        avg_ms = result.gpu_time_ms;
        total_ms = result.gpu_time_ms;
        total_cpu_us = result.cpu_time_us;
    } else {
        min_ms = std::min(min_ms, result.gpu_time_ms);
        max_ms = std::max(max_ms, result.gpu_time_ms);
        total_ms += result.gpu_time_ms;
        avg_ms = total_ms / (count + 1);
        total_cpu_us += result.cpu_time_us;
    }
    count++;
}

inline std::string TimingStats::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << name << ": " << avg_ms << " ms ("
        << min_ms << "-" << max_ms << " ms, "
        << count << " samples)";
    return oss.str();
}

// Inline implementation of CUDAProfiler
inline CUDAProfiler& CUDAProfiler::instance() {
    static CUDAProfiler instance;
    return instance;
}

inline CUDAProfileScope CUDAProfiler::profile(const char* name) {
    return CUDAProfileScope(name, *this);
}

} // namespace MyEngine
