#include "cuda_texture.h"
#include "cuda_error.h"

namespace MyEngine {

// GPU kernel: Clear texture buffer (defined before use in anonymous namespace)
namespace {
    __global__ void clearKernel(
        float4* buffer,
        int width,
        int height,
        size_t pitch,
        float4 color
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        float4* row = (float4*)((char*)buffer + y * pitch);
        row[x] = color;
    }
}

// CUDASamplerConfig implementation
cudaTextureDesc CUDASamplerConfig::toCudaDesc() const {
    cudaTextureDesc desc = {};
    desc.filterMode = (filter == CUDATextureFilter::LINEAR)
        ? cudaFilterModeLinear : cudaFilterModePoint;

    // Address modes
    desc.addressMode[0] = (address_u == CUDATextureAddress::WRAP) ? cudaAddressModeWrap :
                          (address_u == CUDATextureAddress::MIRRORED_REPEAT) ? cudaAddressModeMirror :
                          cudaAddressModeClamp;

    desc.addressMode[1] = (address_v == CUDATextureAddress::WRAP) ? cudaAddressModeWrap :
                          (address_v == CUDATextureAddress::MIRRORED_REPEAT) ? cudaAddressModeMirror :
                          cudaAddressModeClamp;

    desc.addressMode[2] = (address_w == CUDATextureAddress::WRAP) ? cudaAddressModeWrap :
                          (address_w == CUDATextureAddress::MIRRORED_REPEAT) ? cudaAddressModeMirror :
                          cudaAddressModeClamp;

    desc.normalizedCoords = normalized_coords ? 1 : 0;
    desc.maxAnisotropy = (int)max_anisotropy;
    desc.mipmapFilterMode = cudaFilterModeLinear;
    desc.mipmapLevelBias = 0.0f;
    desc.minMipmapLevelClamp = 0.0f;
    desc.maxMipmapLevelClamp = 255.0f;

    return desc;
}

// CUDATextureObject implementation
CUDATextureObject::CUDATextureObject() = default;

CUDATextureObject::~CUDATextureObject() {
    destroy();
}

bool CUDATextureObject::create(cudaArray_t array, const CUDASamplerConfig& config) {
    if (array == nullptr) {
        CUDA_LOG_ERROR("Cannot create texture object from null array");
        return false;
    }

    cudaTextureDesc tex_desc = config.toCudaDesc();
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    cudaError_t err = cudaCreateTextureObject(&_texture, &res_desc, &tex_desc, nullptr);
    if (err != cudaSuccess) {
        CUDA_LOG_ERROR_WITH_CODE("Failed to create texture object", err);
        return false;
    }

    _array = array;
    return true;
}

void CUDATextureObject::destroy() {
    if (_texture != 0) {
        cudaDestroyTextureObject(_texture);
        _texture = 0;
    }
    _array = nullptr;
}

CUDATextureObject::CUDATextureObject(CUDATextureObject&& other) noexcept
    : _texture(other._texture), _array(other._array) {
    other._texture = 0;
    other._array = nullptr;
}

CUDATextureObject& CUDATextureObject::operator=(CUDATextureObject&& other) noexcept {
    if (this != &other) {
        destroy();
        _texture = other._texture;
        _array = other._array;
        other._texture = 0;
        other._array = nullptr;
    }
    return *this;
}

// CUDASurfaceObject implementation
CUDASurfaceObject::CUDASurfaceObject() = default;

CUDASurfaceObject::~CUDASurfaceObject() {
    destroy();
}

bool CUDASurfaceObject::create(cudaArray_t array) {
    if (array == nullptr) {
        CUDA_LOG_ERROR("Cannot create surface object from null array");
        return false;
    }

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    cudaError_t err = cudaCreateSurfaceObject(&_surface, &res_desc);
    if (err != cudaSuccess) {
        CUDA_LOG_ERROR_WITH_CODE("Failed to create surface object", err);
        return false;
    }

    _array = array;
    return true;
}

bool CUDASurfaceObject::update(cudaArray_t array) {
    if (array == nullptr) {
        CUDA_LOG_ERROR("Cannot update surface object with null array");
        return false;
    }

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    cudaError_t err = cudaDestroySurfaceObject(_surface);
    if (err != cudaSuccess) {
        CUDA_LOG_ERROR_WITH_CODE("Failed to destroy old surface object", err);
    }

    err = cudaCreateSurfaceObject(&_surface, &res_desc);
    if (err != cudaSuccess) {
        CUDA_LOG_ERROR_WITH_CODE("Failed to create new surface object", err);
        return false;
    }

    _array = array;
    return true;
}

void CUDASurfaceObject::destroy() {
    if (_surface != 0) {
        cudaDestroySurfaceObject(_surface);
        _surface = 0;
    }
    _array = nullptr;
}

CUDASurfaceObject::CUDASurfaceObject(CUDASurfaceObject&& other) noexcept
    : _surface(other._surface), _array(other._array) {
    other._surface = 0;
    other._array = nullptr;
}

CUDASurfaceObject& CUDASurfaceObject::operator=(CUDASurfaceObject&& other) noexcept {
    if (this != &other) {
        destroy();
        _surface = other._surface;
        _array = other._array;
        other._surface = 0;
        other._array = nullptr;
    }
    return *this;
}

// CUDATexture implementation
CUDATexture::CUDATexture() = default;

CUDATexture::~CUDATexture() {
    destroy();
}

bool CUDATexture::create2D(int width, int height, CUDATextureChannelFormat format,
                           const CUDASamplerConfig& sampler) {
    if (width <= 0 || height <= 0) {
        CUDA_LOG_ERROR("Invalid texture dimensions");
        return false;
    }

    _width = width;
    _height = height;
    _format = format;
    _sampler = sampler;

    // Convert format to cudaChannelFormatDesc
    _channel_desc = getChannelFormatDesc(format);
    if (_channel_desc.x == 0 && _channel_desc.y == 0 &&
        _channel_desc.z == 0 && _channel_desc.w == 0) {
        CUDA_LOG_ERROR("Unsupported texture format");
        return false;
    }

    cudaError_t err = cudaMallocArray(&_array, &_channel_desc, width, height);
    if (err != cudaSuccess) {
        CUDA_LOG_ERROR_WITH_CODE("Failed to allocate CUDA array", err);
        return false;
    }

    // Create texture and surface objects
    if (!_texture_obj.create(_array, sampler)) {
        cudaFreeArray(_array);
        _array = nullptr;
        return false;
    }

    if (!_surface_obj.create(_array)) {
        _texture_obj.destroy();
        cudaFreeArray(_array);
        _array = nullptr;
        return false;
    }

    _owns_array = true;
    return true;
}

void CUDATexture::destroy() {
    _texture_obj.destroy();
    _surface_obj.destroy();
    if (_array && _owns_array) {
        cudaFreeArray(_array);
    }
    _array = nullptr;
    _width = 0;
    _height = 0;
}

void CUDATexture::write(const void* data, size_t pitch, cudaStream_t stream) {
    if (!_array) return;

    CUDATextureLayout layout = CUDATextureLayout::getLayout(_width, _height, _format);
    cudaError_t err = cudaMemcpy2DToArray(
        _array, 0, 0,
        data, pitch,
        layout.pitch,
        _height,
        cudaMemcpyDeviceToDevice
    );

    if (err != cudaSuccess) {
        CUDA_LOG_ERROR_WITH_CODE("Failed to write to texture", err);
    }
    (void)stream; // Unused when copying to array
}

void CUDATexture::read(void* data, size_t pitch, cudaStream_t stream) {
    if (!_array) return;

    CUDATextureLayout layout = CUDATextureLayout::getLayout(_width, _height, _format);
    cudaError_t err = cudaMemcpy2DFromArray(
        data, pitch,
        _array, 0, 0,
        layout.pitch,
        _height,
        cudaMemcpyDeviceToDevice
    );

    if (err != cudaSuccess) {
        CUDA_LOG_ERROR_WITH_CODE("Failed to read from texture", err);
    }
    (void)stream; // Unused when copying from array
}

void CUDATexture::clear(float4 color, cudaStream_t stream) {
    if (!_array) return;

    CUDATextureLayout layout = CUDATextureLayout::getLayout(_width, _height, _format);

    // Allocate temporary buffer for fill
    size_t buffer_size = layout.pitch * _height;
    float4* fill_buffer;
    cudaError_t err = cudaMalloc(&fill_buffer, buffer_size);
    if (err != cudaSuccess) {
        CUDA_LOG_ERROR_WITH_CODE("Failed to allocate fill buffer", err);
        return;
    }

    // Launch kernel to fill buffer
    dim3 block(16, 16);
    dim3 grid((_width + 15) / 16, (_height + 15) / 16);

    clearKernel<<<grid, block, 0, stream>>>(fill_buffer, _width, _height, layout.pitch / sizeof(float4), color);

    // Copy to texture
    err = cudaMemcpy2DToArray(
        _array, 0, 0,
        fill_buffer, layout.pitch,
        layout.pitch,
        _height,
        cudaMemcpyDeviceToDevice
    );

    cudaFree(fill_buffer);

    if (err != cudaSuccess) {
        CUDA_LOG_ERROR_WITH_CODE("Failed to clear texture", err);
    }
}

CUDATextureLayout CUDATextureLayout::getLayout(int width, int height, CUDATextureChannelFormat format) {
    CUDATextureLayout layout = {};
    layout.width = width;
    layout.height = height;

    switch (format) {
        case CUDATextureChannelFormat::R8_UNORM:
            layout.element_size = 1;
            break;
        case CUDATextureChannelFormat::R16_FLOAT:
        case CUDATextureChannelFormat::R32_FLOAT:
            layout.element_size = 4;
            break;
        case CUDATextureChannelFormat::RG8_UNORM:
            layout.element_size = 2;
            break;
        case CUDATextureChannelFormat::RG16_FLOAT:
        case CUDATextureChannelFormat::RG32_FLOAT:
            layout.element_size = 8;
            break;
        case CUDATextureChannelFormat::RGBA8_UNORM:
            layout.element_size = 4;
            break;
        case CUDATextureChannelFormat::RGBA16_FLOAT:
            layout.element_size = 8;
            break;
        case CUDATextureChannelFormat::RGBA32_FLOAT:
            layout.element_size = 16;
            break;
        default:
            layout.element_size = 4;
    }

    // Calculate pitch (aligned to 256 bytes for optimal performance)
    layout.pitch = ((width * layout.element_size + 255) / 256) * 256;

    return layout;
}

cudaChannelFormatDesc getChannelFormatDesc(CUDATextureChannelFormat format) {
    switch (format) {
        case CUDATextureChannelFormat::R8_UNORM:
            return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        case CUDATextureChannelFormat::R16_FLOAT:
            return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
        case CUDATextureChannelFormat::R32_FLOAT:
            return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        case CUDATextureChannelFormat::RG8_UNORM:
            return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
        case CUDATextureChannelFormat::RG16_FLOAT:
            return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindFloat);
        case CUDATextureChannelFormat::RG32_FLOAT:
            return cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
        case CUDATextureChannelFormat::RGBA8_UNORM:
            return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        case CUDATextureChannelFormat::RGBA16_FLOAT:
            return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
        case CUDATextureChannelFormat::RGBA32_FLOAT:
            return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        default:
            return cudaChannelFormatDesc{};
    }
}

} // namespace MyEngine
