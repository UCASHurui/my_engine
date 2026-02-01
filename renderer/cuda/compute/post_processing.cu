#include "post_processing.h"

namespace MyEngine {

// GPU kernel: Tone mapping
__global__ void toneMapKernel(
    float3* input,
    float3* output,
    int width,
    int height,
    ToneMapOperator op,
    float exposure,
    float gamma,
    float brightness,
    float contrast,
    float saturation
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float3 color = input[idx];

    // Apply exposure
    color = vmul(color, exposure);

    // Tone mapping
    switch (op) {
        case ToneMapOperator::ACESFilmic: {
            float3 c = color;
            float3 result = vmul(c, make_float3(2.51f, 2.51f, 2.51f));
            result = vadd(result, make_float3(0.03f, 0.03f, 0.03f));
            float3 denom = vmul(c, make_float3(0.03f, 0.03f, 0.03f));
            denom = vadd(denom, make_float3(0.59f, 0.59f, 0.59f));
            denom = vadd(denom, make_float3(0.14f, 0.14f, 0.14f));
            color = vdiv(result, denom);
            break;
        }
        case ToneMapOperator::Reinhard:
            color = vdiv(color, vadd(color, make_float3(1, 1, 1)));
            break;
        case ToneMapOperator::Exposure:
            color = make_float3(
                1.0f - expf(-color.x),
                1.0f - expf(-color.y),
                1.0f - expf(-color.z)
            );
            break;
    }

    // Apply gamma
    float inv_gamma = 1.0f / gamma;
    color = make_float3(
        powf(color.x, inv_gamma),
        powf(color.y, inv_gamma),
        powf(color.z, inv_gamma)
    );

    // Color grading
    float lum = dot(color, make_float3(0.2126f, 0.7152f, 0.0722f));
    color = vadd(vmul(make_float3(lum, lum, lum), 1.0f - saturation),
                 vmul(color, saturation));

    color = vadd(make_float3(0.5f, 0.5f, 0.5f),
                 vmul(vsub(color, make_float3(0.5f, 0.5f, 0.5f)), contrast));
    color = vadd(color, make_float3(brightness, brightness, brightness));

    output[idx] = clamp3(color, 0.0f, 1.0f);
}

// GPU kernel: Vignette
__global__ void vignetteKernel(
    float3* input,
    float3* output,
    int width,
    int height,
    float strength
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float nx = ((float)x / width - 0.5f) * 2.0f;
    float ny = ((float)y / height - 0.5f) * 2.0f;
    float dist = sqrtf(nx * nx + ny * ny);

    float vignette = 1.0f - clamp(dist * strength, 0.0f, 1.0f);
    output[idx] = vmul(input[idx], vignette);
}

// GPU kernel: Bloom threshold
__global__ void bloomThresholdKernel(
    const float3* input,
    float3* output,
    int width,
    int height,
    float threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float3 color = input[idx];
    float lum = dot(color, make_float3(0.2126f, 0.7152f, 0.0722f));

    if (lum > threshold) {
        output[idx] = color;
    } else {
        output[idx] = make_float3(0, 0, 0);
    }
}

// GPU kernel: Simple box blur
__global__ void blurKernel(
    const float3* input,
    float3* output,
    int width,
    int height,
    int radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float3 sum = make_float3(0, 0, 0);
    int count = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int sx = x + dx;
            int sy = y + dy;
            if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                sum = vadd(sum, input[sy * width + sx]);
                count++;
            }
        }
    }

    output[idx] = vdiv(sum, (float)count);
}

// GPU kernel: Composite
__global__ void compositeKernel(
    const float3* base,
    const float3* overlay,
    float3* output,
    int width,
    int height,
    float intensity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    output[idx] = vadd(base[idx], vmul(overlay[idx], intensity));
}

// Post-processor implementation
PostProcessor::PostProcessor()
    : _width(0), _height(0), _temp_buffer(nullptr), _stream(0) {}

PostProcessor::~PostProcessor() {
    shutdown();
}

bool PostProcessor::initialize(int width, int height) {
    _width = width;
    _height = height;

    size_t size = width * height * sizeof(float3);
    cudaError_t err = cudaMalloc(&_temp_buffer, size);
    if (err != cudaSuccess) return false;

    err = cudaStreamCreate(&_stream);
    return err == cudaSuccess;
}

void PostProcessor::shutdown() {
    if (_stream) {
        cudaStreamDestroy(_stream);
        _stream = 0;
    }
    if (_temp_buffer) {
        cudaFree(_temp_buffer);
        _temp_buffer = nullptr;
    }
    _width = 0;
    _height = 0;
}

bool PostProcessor::applyToneMap(float3* data, int width, int height, const PostProcessConfig& config) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    toneMapKernel<<<grid, block, 0, _stream>>>(
        data, data, width, height,
        config.tone_map, config.exposure, config.gamma,
        config.brightness, config.contrast, config.saturation
    );

    return cudaGetLastError() == cudaSuccess;
}

bool PostProcessor::applyVignette(float3* data, int width, int height, float strength) {
    if (strength <= 0) return true;

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    vignetteKernel<<<grid, block, 0, _stream>>>(
        data, data, width, height, strength
    );

    return cudaGetLastError() == cudaSuccess;
}

bool PostProcessor::applyBloom(float3* data, int width, int height, float threshold, float intensity) {
    if (intensity <= 0) return true;

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    // Extract bright spots
    bloomThresholdKernel<<<grid, block, 0, _stream>>>(
        data, _temp_buffer, width, height, threshold
    );

    // Blur
    blurKernel<<<grid, block, 0, _stream>>>(
        _temp_buffer, _temp_buffer, width, height, 2
    );

    // Composite
    compositeKernel<<<grid, block, 0, _stream>>>(
        data, _temp_buffer, data, width, height, intensity
    );

    return cudaGetLastError() == cudaSuccess;
}

} // namespace MyEngine
