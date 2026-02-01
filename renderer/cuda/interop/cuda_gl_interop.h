#pragma once

#include "cuda_runtime.h"
#include <cstdint>

namespace MyEngine {

// Graphics resource registration flags
enum class CUDAGraphicsRegisterFlags {
    NONE = 0,
    READ_ONLY = cudaGraphicsRegisterFlagsReadOnly,
    WRITE_DISCARD = cudaGraphicsRegisterFlagsWriteDiscard,
    SURFACE_LDST = cudaGraphicsRegisterFlagsSurfaceLoadStore,
    TEXTURE_GATHER = cudaGraphicsRegisterFlagsTextureGather
};

// Graphics map flags
enum class CUDAGraphicsMapFlags {
    NONE = cudaGraphicsMapFlagsNone,
    READ_ONLY = cudaGraphicsMapFlagsReadOnly,
    WRITE_DISCARD = cudaGraphicsMapFlagsWriteDiscard
};

// Configuration for graphics resource registration
struct CUDAGraphicsConfig {
    CUDAGraphicsRegisterFlags register_flags = CUDAGraphicsRegisterFlags::NONE;
    CUDAGraphicsMapFlags map_flags = CUDAGraphicsMapFlags::NONE;
};

// Resource type enumeration
enum class CUDAGraphicsResourceType {
    BUFFER,
    ARRAY,
    PIXEL_BUFFER
};

// RAII wrapper for CUDA graphics resource
class CUDAGraphicsResource {
public:
    CUDAGraphicsResource();
    ~CUDAGraphicsResource();

    // Register OpenGL texture 2D (requires OpenGL)
    bool registerTexture2D(uint32_t gl_texture, int width, int height,
                           CUDAGraphicsConfig config = CUDAGraphicsConfig());

    // Register OpenGL buffer (requires OpenGL)
    bool registerBuffer(uint32_t gl_buffer, size_t size,
                        CUDAGraphicsConfig config = CUDAGraphicsConfig());

    // Unregister and release
    void unregister();

    // Map for CUDA access - returns device pointer
    void* map(cudaStream_t stream = 0);

    // Unmap
    void unmap(cudaStream_t stream = 0);

    // Get CUDA array for texture resources
    cudaArray_t getArray() const;

    // Get device pointer for buffer resources
    void* getDevicePointer() const;

    // Get mapped size
    size_t getSize() const { return _size; }

    // Accessors
    cudaGraphicsResource_t get() const { return _resource; }
    explicit operator bool() const { return _resource != nullptr; }

    // Check if GL interop is available
    bool isGLInteropAvailable() const { return _gl_handle != 0; }

    // Move semantics
    CUDAGraphicsResource(CUDAGraphicsResource&& other) noexcept;
    CUDAGraphicsResource& operator=(CUDAGraphicsResource&& other) noexcept;

    // Copy disabled
    CUDAGraphicsResource(const CUDAGraphicsResource&) = delete;
    CUDAGraphicsResource& operator=(const CUDAGraphicsResource&) = delete;

private:
    cudaGraphicsResource_t _resource = nullptr;
    CUDAGraphicsResourceType _type;
    uint32_t _gl_handle = 0;
    size_t _size = 0;
    bool _is_mapped = false;
};

// OpenGL interop manager
class CUDAGLInterop {
public:
    // Initialize interop (call once at startup)
    static bool initialize();

    // Shutdown interop (call at shutdown)
    static void shutdown();

    // Check if initialized
    static bool isInitialized() { return _initialized; }

    // Check if interop is supported
    static bool isSupported() { return _interop_supported; }

    // Register a texture for CUDA access
    static CUDAGraphicsResource registerTexture(
        uint32_t gl_texture, int width, int height,
        CUDAGraphicsRegisterFlags flags = CUDAGraphicsRegisterFlags::NONE);

    // Register a buffer for CUDA access
    static CUDAGraphicsResource registerBuffer(
        uint32_t gl_buffer, size_t size,
        CUDAGraphicsRegisterFlags flags = CUDAGraphicsRegisterFlags::NONE);

    // Synchronize CUDA and OpenGL
    static void synchronize();

private:
    static bool _initialized;
    static bool _interop_supported;
};

} // namespace MyEngine
