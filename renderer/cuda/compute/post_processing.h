#pragma once

#include "cuda_runtime.h"
#include "cuda_math.h"

namespace MyEngine {

// Tone mapping operators
enum class ToneMapOperator {
    Reinhard = 0,
    ACESFilmic = 1,
    Exposure = 2
};

// Post-processing configuration
struct PostProcessConfig {
    ToneMapOperator tone_map;
    float exposure;
    float gamma;
    float brightness;
    float contrast;
    float saturation;
    float bloom_threshold;
    float bloom_intensity;
    bool vignette_enabled;
    float vignette_strength;

    __host__ __device__ PostProcessConfig()
        : tone_map(ToneMapOperator::ACESFilmic)
        , exposure(1.0f)
        , gamma(2.2f)
        , brightness(0.0f)
        , contrast(1.0f)
        , saturation(1.0f)
        , bloom_threshold(1.0f)
        , bloom_intensity(0.0f)
        , vignette_enabled(true)
        , vignette_strength(0.5f) {}
};

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
);

// GPU kernel: Vignette
__global__ void vignetteKernel(
    float3* input,
    float3* output,
    int width,
    int height,
    float strength
);

// GPU kernel: Bloom threshold
__global__ void bloomThresholdKernel(
    const float3* input,
    float3* output,
    int width,
    int height,
    float threshold
);

// GPU kernel: Simple blur
__global__ void blurKernel(
    const float3* input,
    float3* output,
    int width,
    int height,
    int radius
);

// GPU kernel: Composite
__global__ void compositeKernel(
    const float3* base,
    const float3* overlay,
    float3* output,
    int width,
    int height,
    float intensity
);

// Post-processor class
class PostProcessor {
public:
    PostProcessor();
    ~PostProcessor();

    bool initialize(int width, int height);
    void shutdown();

    // Apply tone mapping
    bool applyToneMap(float3* data, int width, int height, const PostProcessConfig& config);

    // Apply vignette
    bool applyVignette(float3* data, int width, int height, float strength);

    // Apply bloom
    bool applyBloom(float3* data, int width, int height, float threshold, float intensity);

private:
    int _width;
    int _height;
    float3* _temp_buffer;
    cudaStream_t _stream;
};

// Inline helpers
inline __device__ float3 clamp3(float3 v, float min_val, float max_val) {
    return make_float3(
        fminf(fmaxf(v.x, min_val), max_val),
        fminf(fmaxf(v.y, min_val), max_val),
        fminf(fmaxf(v.z, min_val), max_val)
    );
}

inline __device__ float clamp(float v, float min_val, float max_val) {
    return fminf(fmaxf(v, min_val), max_val);
}

inline __device__ float luminance(float3 c) {
    return dot(c, make_float3(0.2126f, 0.7152f, 0.0722f));
}

} // namespace MyEngine
