#include "cuda_profiler.h"
#include "cuda_error.h"

namespace MyEngine {

CUDAProfiler::CUDAProfiler()
    : _enabled(true), _trackBytes(false), _trackOperations(false) {
    cudaEventCreate(&_globalStartEvent);
    cudaEventCreate(&_globalEndEvent);
}

CUDAProfiler::~CUDAProfiler() {
    cudaEventDestroy(_globalStartEvent);
    cudaEventDestroy(_globalEndEvent);
}

TimingResult CUDAProfiler::startTiming(const char* name) {
    TimingResult result;
    result.name = name;
    result.startTimingInternal();
    return result;
}

void CUDAProfiler::endTiming(TimingResult& result) {
    result.endTimingInternal();

    std::lock_guard<std::mutex> lock(_resultsMutex);
    _results.push_back(result);

    updateStatsInternal(result);
    _lastTiming = result;
}

void CUDAProfiler::updateStats(const TimingResult& result) {
    std::lock_guard<std::mutex> lock(_statsMutex);
    updateStatsInternal(result);
}

void CUDAProfiler::updateStatsInternal(const TimingResult& result) {
    for (auto& stat : _stats) {
        if (stat.name == result.name) {
            stat.aggregate(result);
            return;
        }
    }
    TimingStats newStat;
    newStat.aggregate(result);
    _stats.push_back(newStat);
}

std::vector<TimingStats> CUDAProfiler::getStats() const {
    std::lock_guard<std::mutex> lock(_statsMutex);
    return _stats;
}

TimingStats CUDAProfiler::getStatsFor(const char* name) const {
    std::lock_guard<std::mutex> lock(_statsMutex);
    for (const auto& stat : _stats) {
        if (stat.name == name) {
            return stat;
        }
    }
    return TimingStats();
}

bool CUDAProfiler::exportToCSV(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(_resultsMutex);

    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }

    file << "name,gpu_time_ms,cpu_time_us,bytes_transferred,operations\n";

    for (const auto& result : _results) {
        file << result.name << ","
             << std::fixed << std::setprecision(6) << result.gpu_time_ms << ","
             << result.cpu_time_us.count() << ","
             << result.bytes_transferred << ","
             << result.operations << "\n";
    }

    file.close();
    return true;
}

bool CUDAProfiler::exportStatsToCSV(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(_statsMutex);

    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }

    file << "name,count,min_ms,max_ms,avg_ms,total_ms,total_cpu_us\n";

    for (const auto& stat : _stats) {
        file << stat.name << ","
             << stat.count << ","
             << std::fixed << std::setprecision(6) << stat.min_ms << ","
             << stat.max_ms << ","
             << stat.avg_ms << ","
             << stat.total_ms << ","
             << stat.total_cpu_us.count() << "\n";
    }

    file.close();
    return true;
}

void CUDAProfiler::reset() {
    std::lock_guard<std::mutex> lock(_resultsMutex);
    _results.clear();
    _lastTiming = TimingResult();
}

void CUDAProfiler::resetStats() {
    std::lock_guard<std::mutex> lock(_statsMutex);
    _stats.clear();
}

} // namespace MyEngine
