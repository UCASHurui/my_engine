#include <iostream>
#include "cuda_runtime_engine.h"
#include "cuda_memory.h"
#include "cuda_stream.h"
#include "cuda_error.h"

int main() {
    std::cout << "=== MyEngine CUDA Test ===" << std::endl;

    // Test initialization
    if (!MyEngine::CUDARuntime::initialize()) {
        std::cerr << "CUDA initialization failed!" << std::endl;
        return 1;
    }

    std::cout << "CUDA Initialized: " << MyEngine::CUDARuntime::getDeviceName() << std::endl;
    std::cout << "Compute Capability: " << MyEngine::CUDARuntime::getComputeCapability() << std::endl;
    std::cout << "VRAM: " << (MyEngine::CUDARuntime::getTotalMemory() / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Available: " << (MyEngine::CUDARuntime::getAvailableMemory() / 1024 / 1024) << " MB" << std::endl;
    std::cout << "RTX Support: " << (MyEngine::CUDARuntime::supportsRTX() ? "Yes" : "No") << std::endl;
    std::cout << "Managed Memory: " << (MyEngine::CUDARuntime::supportsManagedMemory() ? "Yes" : "No") << std::endl;

    // Test memory allocation
    std::cout << "\n=== Memory Test ===" << std::endl;
    auto deviceMem = MyEngine::CUDAMemoryManager::allocDevice(1024 * 1024); // 1MB
    auto hostMem = MyEngine::CUDAMemoryManager::allocHost(1024 * 1024);     // 1MB
    auto managedMem = MyEngine::CUDAMemoryManager::allocManaged(1024 * 1024); // 1MB

    std::cout << "Device alloc: " << (deviceMem ? "OK" : "FAILED") << std::endl;
    std::cout << "Host alloc: " << (hostMem ? "OK" : "FAILED") << std::endl;
    std::cout << "Managed alloc: " << (managedMem ? "OK" : "FAILED") << std::endl;

    // Test stream creation
    std::cout << "\n=== Stream Test ===" << std::endl;
    MyEngine::CUDAStream stream;
    std::cout << "Stream created: " << (stream ? "OK" : "FAILED") << std::endl;

    // Test stream pool
    MyEngine::CUDAStreamPool pool(4);
    std::cout << "Stream pool (4 streams): OK" << std::endl;

    // Shutdown
    MyEngine::CUDARuntime::shutdown();
    std::cout << "\nCUDA shutdown complete." << std::endl;

    return 0;
}
