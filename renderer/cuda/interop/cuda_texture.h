#pragma once

#include "cuda_runtime.h"
#include "cuda_error.h"
#include <cstdint>

namespace MyEngine {

// Texture channel format
enum class CUDATextureChannelFormat {
    R8_UNORM,
    R16_FLOAT,
    R32_FLOAT,
    RG8_UNORM,
    RG16_FLOAT,
    RG32_FLOAT,
    RGBA8_UNORM,
    RGBA16_FLOAT,
    RGBA32_FLOAT
};

// Texture filter mode
enum class CUDATextureFilter {
    NEAREST,
    LINEAR
};

// Texture address mode
enum class CUDATextureAddress {
    WRAP,
    CLAMP_TO_EDGE,
    MIRRORED_REPEAT
};

// Sampler configuration
struct CUDASamplerConfig {
    CUDATextureFilter filter = CUDATextureFilter::LINEAR;
    CUDATextureAddress address_u = CUDATextureAddress::CLAMP_TO_EDGE;
    CUDATextureAddress address_v = CUDATextureAddress::CLAMP_TO_EDGE;
    CUDATextureAddress address_w = CUDATextureAddress::CLAMP_TO_EDGE;
    float max_anisotropy = 1.0f;
    bool normalized_coords = true;

    // Conversion to cudaTextureDesc
    cudaTextureDesc toCudaDesc() const;
};

// RAII wrapper for CUDA texture object
class CUDATextureObject {
public:
    CUDATextureObject();
    ~CUDATextureObject();

    // Create from existing CUDA array
    bool create(cudaArray_t array, const CUDASamplerConfig& config = CUDASamplerConfig());

    // Destroy and release
    void destroy();

    // Move semantics
    CUDATextureObject(CUDATextureObject&& other) noexcept;
    CUDATextureObject& operator=(CUDATextureObject&& other) noexcept;

    // Copy disabled
    CUDATextureObject(const CUDATextureObject&) = delete;
    CUDATextureObject& operator=(const CUDATextureObject&) = delete;

    // Accessors
    cudaTextureObject_t get() const { return _texture; }
    operator cudaTextureObject_t() const { return _texture; }
    explicit operator bool() const { return _texture != 0; }

private:
    cudaTextureObject_t _texture = 0;
    cudaArray_t _array = nullptr;
};

// RAII wrapper for CUDA surface object (writable)
class CUDASurfaceObject {
public:
    CUDASurfaceObject();
    ~CUDASurfaceObject();

    // Create from existing CUDA array
    bool create(cudaArray_t array);

    // Recreate with new array
    bool update(cudaArray_t array);

    // Destroy
    void destroy();

    // Move semantics
    CUDASurfaceObject(CUDASurfaceObject&& other) noexcept;
    CUDASurfaceObject& operator=(CUDASurfaceObject&& other) noexcept;

    // Copy disabled
    CUDASurfaceObject(const CUDASurfaceObject&) = delete;
    CUDASurfaceObject& operator=(const CUDASurfaceObject&) = delete;

    // Accessors
    cudaSurfaceObject_t get() const { return _surface; }
    operator cudaSurfaceObject_t() const { return _surface; }
    explicit operator bool() const { return _surface != 0; }

private:
    cudaSurfaceObject_t _surface = 0;
    cudaArray_t _array = nullptr;
};

// High-level CUDA texture wrapper
class CUDATexture {
public:
    CUDATexture();
    ~CUDATexture();

    // Create 2D texture
    bool create2D(int width, int height, CUDATextureChannelFormat format,
                  const CUDASamplerConfig& sampler = CUDASamplerConfig());

    // Destroy
    void destroy();

    // Write data to texture (from device memory)
    void write(const void* data, size_t pitch, cudaStream_t stream = 0);

    // Read data from texture (to device memory)
    void read(void* data, size_t pitch, cudaStream_t stream = 0);

    // Clear texture to a color
    void clear(float4 color, cudaStream_t stream = 0);

    // Accessors
    int getWidth() const { return _width; }
    int getHeight() const { return _height; }
    cudaTextureObject_t getTexture() const { return _texture_obj.get(); }
    cudaSurfaceObject_t getSurface() const { return _surface_obj.get(); }
    cudaArray_t getArray() const { return _array; }

    bool isValid() const { return _array != nullptr; }

private:
    int _width = 0;
    int _height = 0;
    CUDATextureChannelFormat _format;
    cudaChannelFormatDesc _channel_desc;
    cudaArray_t _array = nullptr;
    CUDATextureObject _texture_obj;
    CUDASurfaceObject _surface_obj;
    CUDASamplerConfig _sampler;
    bool _owns_array = true;
};

// Texture data layout helpers
struct CUDATextureLayout {
    size_t element_size;      // Size of one pixel in bytes
    size_t pitch;             // Row pitch in bytes
    int width;
    int height;

    static CUDATextureLayout getLayout(int width, int height, CUDATextureChannelFormat format);
};

// Convert channel format to cudaChannelFormatDesc
cudaChannelFormatDesc getChannelFormatDesc(CUDATextureChannelFormat format);

} // namespace MyEngine
