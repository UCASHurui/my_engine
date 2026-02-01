#include "cuda_gl_interop.h"
#include "cuda_error.h"
#include "cuda_runtime_engine.h"
#include <iostream>

namespace MyEngine {

// Static initialization
bool CUDAGLInterop::_initialized = false;
bool CUDAGLInterop::_interop_supported = false;

bool CUDAGLInterop::initialize() {
    if (_initialized) return true;

    // Check if CUDA runtime is initialized
    if (!CUDARuntime::isInitialized()) {
        std::cerr << "CUDA runtime not initialized for GL interop" << std::endl;
        return false;
    }

    // Check CUDA-GL interop capability
    // Note: cudaDevAttrGLDevIsSupported may not be available in all CUDA versions
    // Try to detect by attempting basic interop check
    int cuda_version;
    cudaDriverGetVersion(&cuda_version);

    if (cuda_version >= 11000) { // CUDA 11.0+ should have better support
        _interop_supported = true;
    } else {
        _interop_supported = false;
    }

    _initialized = true;
    std::cout << "CUDA-OpenGL interop initialized (supported: "
              << (_interop_supported ? "yes" : "limited") << ")" << std::endl;
    return true;
}

void CUDAGLInterop::shutdown() {
    if (_initialized) {
        synchronize();
        _initialized = false;
        _interop_supported = false;
        std::cout << "CUDA-OpenGL interop shutdown complete" << std::endl;
    }
}

CUDAGraphicsResource CUDAGLInterop::registerTexture(
    uint32_t gl_texture, int width, int height,
    CUDAGraphicsRegisterFlags flags) {

    CUDAGraphicsResource resource;
    if (!resource.registerTexture2D(gl_texture, width, height,
                                     CUDAGraphicsConfig{flags, CUDAGraphicsMapFlags::NONE})) {
        return CUDAGraphicsResource();
    }
    return resource;
}

CUDAGraphicsResource CUDAGLInterop::registerBuffer(
    uint32_t gl_buffer, size_t size,
    CUDAGraphicsRegisterFlags flags) {

    CUDAGraphicsResource resource;
    if (!resource.registerBuffer(gl_buffer, size,
                                  CUDAGraphicsConfig{flags, CUDAGraphicsMapFlags::NONE})) {
        return CUDAGraphicsResource();
    }
    return resource;
}

void CUDAGLInterop::synchronize() {
    cudaStreamSynchronize(0);
    cudaDeviceSynchronize();
}

// CUDAGraphicsResource implementation
CUDAGraphicsResource::CUDAGraphicsResource()
    : _type(CUDAGraphicsResourceType::BUFFER) {}

CUDAGraphicsResource::~CUDAGraphicsResource() {
    unregister();
}

bool CUDAGraphicsResource::registerTexture2D(uint32_t gl_texture, int width, int height,
                                              CUDAGraphicsConfig config) {
    if (gl_texture == 0) {
        std::cerr << "Cannot register null OpenGL texture" << std::endl;
        return false;
    }

    // Only attempt registration if interop is supported
    if (!CUDAGLInterop::isSupported()) {
        std::cerr << "OpenGL interop not supported in this configuration" << std::endl;
        return false;
    }

#ifdef CUDA_GL_INTEROP_AVAILABLE
    // Full OpenGL interop implementation
    cudaError_t err = cudaGraphicsGLRegister2(
        &_resource,
        gl_texture,
        static_cast<unsigned int>(config.register_flags));

    if (err != cudaSuccess) {
        std::cerr << "Failed to register OpenGL texture: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    _gl_handle = gl_texture;
    _type = CUDAGraphicsResourceType::ARRAY;
    _size = static_cast<size_t>(width) * height * 4;
    return true;
#else
    // Stub implementation - GL interop not available at compile time
    std::cerr << "OpenGL interop not available (CUDA_GL_INTEROP_AVAILABLE not defined)" << std::endl;
    (void)width;
    (void)height;
    (void)config;
    return false;
#endif
}

bool CUDAGraphicsResource::registerBuffer(uint32_t gl_buffer, size_t size,
                                           CUDAGraphicsConfig config) {
    if (gl_buffer == 0) {
        std::cerr << "Cannot register null OpenGL buffer" << std::endl;
        return false;
    }

    if (!CUDAGLInterop::isSupported()) {
        std::cerr << "OpenGL interop not supported in this configuration" << std::endl;
        return false;
    }

#ifdef CUDA_GL_INTEROP_AVAILABLE
    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &_resource,
        gl_buffer,
        static_cast<unsigned int>(config.register_flags));

    if (err != cudaSuccess) {
        std::cerr << "Failed to register OpenGL buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    _gl_handle = gl_buffer;
    _type = CUDAGraphicsResourceType::BUFFER;
    _size = size;
    return true;
#else
    std::cerr << "OpenGL interop not available (CUDA_GL_INTEROP_AVAILABLE not defined)" << std::endl;
    (void)size;
    (void)config;
    return false;
#endif
}

void CUDAGraphicsResource::unregister() {
    if (_resource) {
        if (_is_mapped) {
            cudaGraphicsUnmapResources(1, &_resource, 0);
            _is_mapped = false;
        }

#ifdef CUDA_GL_INTEROP_AVAILABLE
        cudaError_t err = cudaGraphicsUnregisterResource(_resource);
        if (err != cudaSuccess) {
            std::cerr << "Failed to unregister graphics resource: " << cudaGetErrorString(err) << std::endl;
        }
#endif
        _resource = nullptr;
    }
    _gl_handle = 0;
    _size = 0;
}

void* CUDAGraphicsResource::map(cudaStream_t stream) {
    if (!_resource) return nullptr;

    if (_is_mapped) {
        std::cerr << "Warning: Resource already mapped" << std::endl;
        return getDevicePointer();
    }

#ifdef CUDA_GL_INTEROP_AVAILABLE
    cudaError_t err = cudaGraphicsMapResources(1, &_resource, stream);
    if (err != cudaSuccess) {
        std::cerr << "Failed to map graphics resource: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
#endif
    _is_mapped = true;

    if (_type == CUDAGraphicsResourceType::BUFFER) {
        void* device_ptr = nullptr;
        size_t size = 0;
#ifdef CUDA_GL_INTEROP_AVAILABLE
        cudaError_t err = cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, _resource);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get mapped pointer: " << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
#endif
        return device_ptr;
    }

    return nullptr;
}

void CUDAGraphicsResource::unmap(cudaStream_t stream) {
    if (_resource && _is_mapped) {
#ifdef CUDA_GL_INTEROP_AVAILABLE
        cudaGraphicsUnmapResources(1, &_resource, stream);
#endif
        _is_mapped = false;
    }
}

cudaArray_t CUDAGraphicsResource::getArray() const {
    if (!_resource || _type != CUDAGraphicsResourceType::ARRAY) {
        return nullptr;
    }

#ifdef CUDA_GL_INTEROP_AVAILABLE
    cudaArray_t array = nullptr;
    cudaError_t err = cudaGraphicsSubResourceGetMappedArray(&array, _resource, 0, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get mapped array: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return array;
#else
    return nullptr;
#endif
}

void* CUDAGraphicsResource::getDevicePointer() const {
    if (!_resource || _type != CUDAGraphicsResourceType::BUFFER) {
        return nullptr;
    }

#ifdef CUDA_GL_INTEROP_AVAILABLE
    void* ptr = nullptr;
    size_t size = 0;
    cudaError_t err = cudaGraphicsResourceGetMappedPointer(&ptr, &size, _resource);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get mapped pointer: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return ptr;
#else
    return nullptr;
#endif
}

CUDAGraphicsResource::CUDAGraphicsResource(CUDAGraphicsResource&& other) noexcept
    : _resource(other._resource), _type(other._type),
      _gl_handle(other._gl_handle), _size(other._size),
      _is_mapped(other._is_mapped) {
    other._resource = nullptr;
    other._gl_handle = 0;
    other._is_mapped = false;
}

CUDAGraphicsResource& CUDAGraphicsResource::operator=(CUDAGraphicsResource&& other) noexcept {
    if (this != &other) {
        unregister();
        _resource = other._resource;
        _type = other._type;
        _gl_handle = other._gl_handle;
        _size = other._size;
        _is_mapped = other._is_mapped;
        other._resource = nullptr;
        other._gl_handle = 0;
        other._is_mapped = false;
    }
    return *this;
}

} // namespace MyEngine
