# CUDA Integration Architecture

## Overview

This document describes the CUDA integration plan for MyEngine to enable GPU-accelerated ray tracing, compute shaders, and general-purpose GPU computing.

## System Requirements

### Hardware
- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM, Compute Capability 8.9)
- Supports: CUDA 12.x, RT Cores, Tensor Cores

### Required Software
1. **CUDA Toolkit 12.x** - CUDA compiler (nvcc) and runtime
2. **NVIDIA Driver 535+** - Already installed via WSL
3. **CMake 3.18+** - Build system with CUDA support
4. **OptiX 8.0+** (optional) - NVIDIA ray tracing SDK

### Installation Status
| Software | Status | Notes |
|----------|--------|-------|
| NVIDIA Driver | ✅ Installed | Version via nvidia-smi |
| CUDA Toolkit | ❌ Not installed | Need to install |
| CMake | ⚠️ Check | Need to verify |

## Architecture Design

### Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│              (Game Code, Scenes, Scripts)                    │
├─────────────────────────────────────────────────────────────┤
│                   Engine Core Layer                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Scripting / Game Logic                      ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                  Rendering Backend                           │
│  ┌──────────────────┐  ┌──────────────────────────────────┐│
│  │  OpenGL Render   │  │     CUDA Compute Backend         ││
│  │  (Legacy/Web)    │  │  ┌────────────────────────────┐  ││
│  │                  │  │  │   CUDA Ray Tracer          │  ││
│  │                  │  │  │   - Path Tracer            │  ││
│  │                  │  │  │   - RT Core Integration    │  ││
│  │                  │  │  └────────────────────────────┘  ││
│  │                  │  │  ┌────────────────────────────┐  ││
│  │                  │  │  │   Compute Kernels          │  ││
│  │                  │  │  │   - Particle Simulation    │  ││
│  │                  │  │  │   - Physics                │  ││
│  │                  │  │  │   - Post-Processing        │  ││
│  │                  │  │  └────────────────────────────┘  ││
│  └──────────────────┘  └──────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    Memory Layer                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │         Unified Memory Manager (UMA/PMR)                ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ ││
│  │  │  CPU Memory │  │  GPU Memory │  │  Unified Memory │ ││
│  │  │  (Host)     │  │  (Device)   │  │  (Managed)      │ ││
│  │  └─────────────┘  └─────────────┘  └─────────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. CUDA Runtime Layer (`renderer/cuda/`)

```
renderer/cuda/
├── cuda_runtime.h       # CUDA runtime initialization
├── cuda_context.h       # Context management
├── cuda_memory.h        # Memory allocation (pinned, managed)
├── cuda_stream.h        # Stream management for async ops
└── cuda_error.h         # Error handling utilities
```

#### 2. Ray Tracing Module (`renderer/cuda/rt/`)

```
renderer/cuda/rt/
├── ray_tracer.h         # Main ray tracing interface
├── bvh_builder.h        # BVH construction (SAH)
├── bvh_node.h           # BVH node structures
├── ray_generator.h      # CUDA ray generation kernels
├── ray_intersect.h      # Ray-triangle intersection (RT Core)
├── shading_kernel.h     # Material evaluation on GPU
├── denoise_kernel.h     # AI denoising (Tensor Core)
└── path_tracer.h        # Path tracing accumulation
```

#### 3. Compute Module (`renderer/cuda/compute/`)

```
renderer/cuda/compute/
├── compute_scheduler.h  # Dispatch compute work
├── particle_kernel.h    # GPU particle simulation
├── physics_kernel.h     # GPU physics (rigid body)
├── blur_kernel.h        # Gaussian blur for effects
├── reduction_kernel.h   # Parallel reduction ops
└── sort_kernel.h        # Radix sort for BVH
```

#### 4. Hybrid Renderer (`renderer/`)

```
renderer/
├── hybrid_renderer.h    # Selects GPU path at runtime
├── rasterizer.h         # Standard rasterization
├── rt_hybridizer.h      # Combines raster + ray trace
├── frame_buffer.h       # Accumulation buffer
└── gbuffer.h            # Geometry buffer for hybrid
```

### Data Structures

#### BVH (Bounding Volume Hierarchy)

```cpp
// Compact BVH for GPU traversal
struct BVHNode {
    float3 min_bounds;    // Bounding box min
    uint32_t child_right; // Right child index or primitive count
    float3 max_bounds;    // Bounding box max
    uint32_t child_left;  // Left child index
};

struct BVHPrimitive {
    float3 position;      // Centroid for SAH
    uint32_t triangle_idx;// Index into triangle buffer
};
```

#### Ray Structure for RT Cores

```cpp
struct Ray {
    float3 origin;
    float t_min;
    float3 direction;
    float t_max;
};

struct HitRecord {
    float3 position;
    float3 normal;
    float2 uv;
    uint32_t material_id;
    float t;
};
```

### Memory Management

#### Unified Memory Strategy

```cpp
class CUDAMemoryManager {
public:
    // Allocate managed memory (accessible by CPU and GPU)
    template<typename T>
    T* alloc_managed(size_t count, cudaMemoryAdvise adv = cudaMemAdviseSetPreferred);

    // Prefetch for access pattern optimization
    void prefetch(void* ptr, size_t size, int device);

    // Async memory copy
    void memcpy_async(void* dst, const void* src, size_t size, cudaStream_t stream);

    // Memory pool for frequent allocations
    MemoryPool* create_pool(size_t block_size, size_t block_count);
};
```

### Shader Integration

#### CUDA-OpenGL Interop

```cpp
class CUDAOpenGLInterop {
public:
    // Register OpenGL texture for CUDA access
    cudaGraphicsResource* register_texture(GLuint texture_id);

    // Map graphics resource for CUDA kernels
    void* map_resource(cudaGraphicsResource* resource);

    // Unmap and sync
    void unmap_resource(cudaGraphicsResource* resource);

    // Surface reference for read/write
    cudaSurfaceObject_t create_surface(cudaGraphicsResource* resource);
};
```

### Rendering Pipeline

#### Hybrid Rendering Approach

```
┌────────────────────────────────────────────────────────────────┐
│                    Frame Rendering Pipeline                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Rasterization Pass                                          │
│     ├── Draw opaque geometry (rasterizer)                       │
│     └── Write to G-Buffer (albedo, normal, depth)              │
│                     ↓                                           │
│  2. CUDA Ray Tracing Pass                                       │
│     ├── Generate rays from G-Buffer                             │
│     ├── Trace rays (RT Core)                                    │
│     ├── Shade hits (PBR)                                        │
│     └── Accumulate indirect light                               │
│                     ↓                                           │
│  3. AI Denoising Pass (Tensor Core)                             │
│     ├── Apply DLSS/OMPTAI denoising                             │
│     └── Temporal accumulation                                   │
│                     ↓                                           │
│  4. Post-Processing (CUDA)                                      │
│     ├── Bloom (Gaussian blur)                                   │
│     ├── Color grading                                           │
│     └── FXAA/SMAA                                               │
│                     ↓                                           │
│  5. Display                                                     │
│     └── Present to screen                                       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Performance Optimization

#### Asynchronous Operations

```cpp
// Multi-stream architecture
cudaStream_t raster_stream;    // OpenGL synchronization
cudaStream_t compute_stream;   // Physics/particles
cudaStream_t rt_stream;        // Ray tracing
cudaStream_t copy_stream;      // Memory operations
```

#### BVH Updates

- **Static geometry**: Build BVH once, cache
- **Dynamic geometry**: Update bounding boxes only, partial rebuild
- **Instance motion**: Motion blur BVH with velocity bounds

### Build System Integration

```cmake
# CMakeLists.txt additions
if(ENABLE_CUDA)
    enable_language(CUDA)

    # Find CUDA
    find_package(CUDA REQUIRED)

    # CUDA compile flags
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -arch=sm_89          # RTX 4060 (Ada)
        -O3                  # Optimization
        --use_fast_math      # Fast math operations
        --extra-device-vectorization
    )

    # Include directories
    include_directories(${CUDA_INCLUDE_DIRS})

    # Compile CUDA sources
    cuda_add_library(cuda_core
        renderer/cuda/cuda_runtime.cu
        renderer/cuda/rt/ray_tracer.cu
        renderer/cuda/compute/compute_kernels.cu
    )
endif()
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Install CUDA Toolkit 12.x
- [ ] Create CUDA runtime wrapper layer
- [ ] Implement memory manager (managed memory)
- [ ] Set up OpenGL-CUDA interop

### Phase 2: Ray Tracing Core (Week 2)
- [ ] Implement BVH builder (SAH)
- [ ] Create ray-triangle intersection (RT Core)
- [ ] Build path tracer kernel
- [ ] Integrate with existing material system

### Phase 3: Compute Kernels (Week 3)
- [ ] Particle simulation on GPU
- [ ] Physics acceleration
- [ ] Post-processing effects (blur, bloom)

### Phase 4: Hybrid Renderer (Week 4)
- [ ] G-Buffer generation
- [ ] Hybrid raster + ray trace pipeline
- [ ] AI denoising integration
- [ ] Performance profiling and tuning

## API Usage Example

```cpp
// Initialize CUDA
CUDARuntime::initialize();

// Create ray tracer
auto ray_tracer = RayTracer::create();
ray_tracer->set_max_bounces(4);
ray_tracer->set_samples_per_pixel(128);

// Load scene
ray_tracer->load_scene(scene);

// Render frame
auto output = ray_tracer->render_frame(camera, delta_time);

// Display result
renderer->blit_texture(output);
```

## Compatibility Notes

- **WSL2**: Requires WSLg for display
- **Driver**: NVIDIA driver 535+ for CUDA 12.x
- **Fallback**: Graceful degradation to CPU rendering if CUDA unavailable
